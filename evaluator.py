import torch
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm

class DistributedMultiModalEvaluator:
    def __init__(self, device, ks=[1, 10, 30]):

        self.device = device
        self.ks = ks
        # 定义检索关系：用 queries 去匹配 anchors
        self.anchor_keys = ["text_embed", "image_embed"]
        self.query_keys = ["audio_embed", "depth_embed", "tactile_embed", "spatial_embed"]
    
    def gather_features(self, tensor):
        if tensor is None or not dist.is_initialized(): 
            return tensor

        world_size = dist.get_world_size()
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
        #gathered[dist.get_rank()] = tensor
        return torch.cat(gathered, dim=0)

    @torch.no_grad()
    def evaluate(self, model, val_loader):
        model.eval()
        
        all_keys = self.anchor_keys + self.query_keys
        local_storage = {k: [] for k in all_keys}

        pbar = tqdm(val_loader, desc="DDP Validating", disable=(dist.get_rank() != 0))
        for batch in pbar:
            batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=(self.device.type == "cuda")):
                outputs = model(batch)
            
            for k in all_keys:
                if outputs.get(k) is not None:
                    local_storage[k].append(outputs[k].detach())

        global_storage = {}
        for k in all_keys:
            if not local_storage[k]:
                global_storage[k] = None
                continue
            
            local_feat = torch.cat(local_storage[k], dim=0)
            global_storage[k] = self.gather_features(local_feat)

        metrics = {}
        if dist.get_rank() == 0:
            metrics = self._calculate_metrics(global_storage)
            
        dist.barrier()
        return metrics

    def _calculate_metrics(self, storage):
        results = {}
        # 遍历查询模态 (如 Audio)
        for q_key in self.query_keys:
            q_feat = storage[q_key]
            if q_feat is None: 
                continue
            
            # 遍历锚点模态 (如 Image/Text)
            for a_key in self.anchor_keys:
                a_feat = storage[a_key]
                if a_feat is None: continue
                
                # 计算相似度矩阵
                # 如果数据集很大，这里可以使用分块乘法防止 OOM
                sim = q_feat @ a_feat.T
                
                q_name = q_key.split('_')[0]
                a_name = a_key.split('_')[0]
                prefix = f"val/{q_name}_to_{a_name}"
                
                results.update(self._top_k_accuracy(sim, prefix))

        return results

    def _top_k_accuracy(self, sim_matrix, prefix):
        n = sim_matrix.size(0)
        targets = torch.arange(n, device=sim_matrix.device)
        
        max_k = max(self.ks)
        _, indices = sim_matrix.topk(max_k, dim=1)
        
        accs = {}
        for k in self.ks:
            correct = indices[:, :k].eq(targets.view(-1, 1)).any(dim=1)
            acc = correct.float().mean().item()
            accs[f"{prefix}_R{k}"] = acc
        return accs
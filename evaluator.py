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
            global_storage[k] = self._all_gather_and_clean(local_feat, val_loader)

        metrics = {}
        if dist.get_rank() == 0:
            metrics = self._calculate_metrics(global_storage)
            
        dist.barrier()
        return metrics

    def _all_gather_and_clean(self, local_feat, loader):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        # 获取各卡样本数
        local_size = torch.tensor([local_feat.size(0)], device=self.device)
        size_list = [torch.zeros_like(local_size) for _ in range(world_size)]
        dist.all_gather(size_list, local_size)
        
        # 为了 all_gather 成功，必须补齐到最大长度
        max_size = max([s.item() for s in size_list])
        dim = local_feat.size(1)
        
        padded_local = F.pad(local_feat, (0, 0, 0, max_size - local_feat.size(0)))
        gather_list = [torch.zeros_like(padded_local) for _ in range(world_size)]
        dist.all_gather(gather_list, padded_local)
        
        # 合并并根据实际 dataset 长度裁剪 (剔除 DistributedSampler 的 padding)
        all_feat = torch.cat([gather_list[i][:size_list[i]] for i in range(world_size)], dim=0)
        
        dataset_len = len(loader.dataset)
        return all_feat[:dataset_len]

    def _calculate_metrics(self, storage):
        results = {}
        # 遍历查询模态 (如 Audio)
        for q_key in self.query_keys:
            q_feat = storage[q_key]
            if q_feat is None: continue
            
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
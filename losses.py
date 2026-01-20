import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class DistributedContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None

    def gather_features(self, tensor):
        """将所有 GPU 上的特征收集起来并保持梯度流"""
        if not dist.is_initialized():
            return tensor
        
        world_size = dist.get_world_size()
        gathered_tensor = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered_tensor, tensor)
        
        # 这一步很关键：为了让梯度传回当前 rank 的特征，
        # 我们用当前 rank 的 tensor 替换掉 gather 列表中的对应部分
        rank = dist.get_rank()
        gathered_tensor[rank] = tensor
        return torch.cat(gathered_tensor, dim=0)

    def forward(self, outputs):
        audio_embed = outputs.get('audio_embed')
        image_embed = outputs.get('image_embed')
        logit_scale = outputs['logit_scale']
        
        device = (audio_embed if audio_embed is not None else image_embed).device
        local_batch_size = (audio_embed if audio_embed is not None else image_embed).size(0)

        # 生成全局标签
        rank = dist.get_rank() if dist.is_initialized() else 0
        labels = torch.arange(local_batch_size, device=device) + rank * local_batch_size

        # 分布式特征收集 (Global features for negative sampling)
        audio_all = self.gather_features(audio_embed) if audio_embed is not None else None
        image_all = self.gather_features(image_embed) if image_embed is not None else None

        loss_dict = {}
        total_loss = 0.0

        # 音频 <-> 图像
        if audio_embed is not None and image_embed is not None:
            # 当前 rank 的音频与全局图像对比
            logits_a2i = logit_scale * audio_embed @ image_all.t()
            # 当前 rank 的图像与全局音频对比
            logits_i2a = logit_scale * image_embed @ audio_all.t()
            
            loss_a2i = (F.cross_entropy(logits_a2i, labels) + F.cross_entropy(logits_i2a, labels)) / 2
            total_loss += loss_a2i
            
            with torch.no_grad():
                pred = torch.argmax(logits_a2i, dim=-1)
                acc = (pred == labels).float().mean() * 100
                loss_dict['train/acc_audio_image'] = acc

        loss_dict['loss'] = total_loss
        return loss_dict
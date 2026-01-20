import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class DualAnchorContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def gather_features(self, tensor):
        if tensor is None or not dist.is_initialized(): return tensor
        world_size = dist.get_world_size()
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
        gathered[dist.get_rank()] = tensor
        return torch.cat(gathered, dim=0)

    def contrastive_pair(self, feat1, feat2_all, logit_scale, labels):
        """计算单向对比损失"""
        logits = logit_scale * feat1 @ feat2_all.t()
        return F.cross_entropy(logits, labels)

    def forward(self, outputs):
        # 提取双核
        text_emb = outputs.get('text_embed')    # Anchor 1
        img_emb = outputs.get('image_embed')    # Anchor 2
        logit_scale = outputs['logit_scale']
        
        device = (text_emb if text_emb is not None else img_emb).device
        local_bs = (text_emb if text_emb is not None else img_emb).size(0)
        rank = dist.get_rank() if dist.is_initialized() else 0
        labels = torch.arange(local_bs, device=device) + rank * local_bs

        # Gather 全局 Anchor
        text_all = self.gather_features(text_emb)
        img_all = self.gather_features(img_emb)

        loss_dict = {"loss": 0.0}
        
        # 遍历所有需要对齐的模态
        other_modalities = ['audio_embed', 'depth_embed', 'tactile_embed', 'spatial_embed']
        
        for m_key in other_modalities:
            m_emb = outputs.get(m_key)
            if m_emb is not None:
                m_all = self.gather_features(m_emb)
                
                # 1. 与 Text 对齐
                if text_emb is not None:
                    l_m2t = self.contrastive_pair(m_emb, text_all, logit_scale, labels)
                    l_t2m = self.contrastive_pair(text_emb, m_all, logit_scale, labels)
                    loss_dict[f'loss_{m_key}_to_text'] = (l_m2t + l_t2m) / 2
                    loss_dict['loss'] += loss_dict[f'loss_{m_key}_to_text']
                
                # 2. 与 Image 对齐
                if img_emb is not None:
                    l_m2i = self.contrastive_pair(m_emb, img_all, logit_scale, labels)
                    l_i2m = self.contrastive_pair(img_emb, m_all, logit_scale, labels)
                    loss_dict[f'loss_{m_key}_to_img'] = (l_m2i + l_i2m) / 2
                    loss_dict['loss'] += loss_dict[f'loss_{m_key}_to_img']

        # 3. 双核互相对齐 (Text <-> Image)
        if text_emb is not None and img_emb is not None:
            l_t2i = self.contrastive_pair(text_emb, img_all, logit_scale, labels)
            l_i2t = self.contrastive_pair(img_emb, text_all, logit_scale, labels)
            loss_dict['loss_text_img'] = (l_t2i + l_i2t) / 2
            loss_dict['loss'] += loss_dict['loss_text_img']

        return loss_dict
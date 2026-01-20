import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, Qwen3OmniMoeAudioEncoder
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import _get_feat_extract_output_lengths

class UnifiedAligner(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # 视觉/文本核
        self.clip = CLIPModel.from_pretrained(cfg.clip_path)
        for param in self.clip.parameters():
            param.requires_grad = False
        self.clip.logit_scale.requires_grad = True
        
        # 音频核
        from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeAudioEncoderConfig
        audio_config = Qwen3OmniMoeAudioEncoderConfig.from_pretrained(cfg.audio_encoder_path)
        if cfg.use_flash_attention:
            audio_config._attn_implementation = "flash_attention_2"
            
        self.audio_encoder = Qwen3OmniMoeAudioEncoder.from_pretrained(
            cfg.audio_encoder_path,
            config=audio_config,
        )
        
        self.embed_dim = self.clip.config.projection_dim
        self.audio_proj = nn.Linear(self.audio_encoder.config.output_dim, self.embed_dim)

    def forward(self, batch):
        outputs = {}
        
        # 1. Image Embeddings
        if "pixel_values" in batch:
            # DDP 模式下通过 self 访问，若在外部访问 logit_scale 需注意 .module
            img_feat = self.clip.get_image_features(pixel_values=batch["pixel_values"])
            outputs["image_embed"] = F.normalize(img_feat, dim=-1)

        # 2. Audio Embeddings
        if "audio_values" in batch:
            audio_values = batch["audio_values"]
            mask = batch.get("audio_attention_mask")
            B, _, T = audio_values.shape
            if mask is None: mask = torch.ones((B, T), device=audio_values.device)
            
            feat_lens = mask.sum(-1)
            packed = audio_values.permute(0, 2, 1)[mask.bool()].permute(1, 0)
            
            audio_out = self.audio_encoder(input_features=packed, feature_lens=feat_lens)
            audio_hidden = audio_out.last_hidden_state
            
            out_lens = _get_feat_extract_output_lengths(feat_lens)
            vectors, start = [], 0
            for l in out_lens.tolist():
                vectors.append(audio_hidden[start : start + l].mean(dim=0))
                start += l
            
            aud_feat = self.audio_proj(torch.stack(vectors, dim=0))
            outputs["audio_embed"] = F.normalize(aud_feat, dim=-1)

        # 3. Logit Scale (Temperature)
        # 统一由 CLIP 的 logit_scale 控制
        outputs["logit_scale"] = self.clip.logit_scale.exp().clamp(max=100)
        
        return outputs
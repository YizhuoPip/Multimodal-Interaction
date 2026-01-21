import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder,
    _get_feat_extract_output_lengths,
)
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeAudioEncoderConfig

class UnifiedAligner(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # --- 核心双核：CLIP (Text & RGB) ---
        self.clip = CLIPModel.from_pretrained(cfg.clip_path)
        for param in self.clip.parameters():
            param.requires_grad = False
        self.clip.logit_scale.requires_grad = True
        self.embed_dim = self.clip.config.projection_dim
        
        # --- 音频核：Qwen3-Omni ---
        audio_config = Qwen3OmniMoeAudioEncoderConfig.from_pretrained(cfg.audio_encoder_path)
        if cfg.use_flash_attention:
            audio_config._attn_implementation = "flash_attention_2"
        self.audio_encoder = Qwen3OmniMoeAudioEncoder.from_pretrained(
            cfg.audio_encoder_path,
            config=audio_config,
            attn_implementation="flash_attention_2" if cfg.use_flash_attention else None,
            dtype=torch.float32,
        )
        self.audio_proj = nn.Linear(self.audio_encoder.config.output_dim, self.embed_dim)

        # --- Depth, Tactile, Spatial ---
        self.depth_proj = nn.Linear(768, self.embed_dim)
        self.tactile_proj = nn.Linear(768, self.embed_dim)
        self.spatial_proj = nn.Linear(768, self.embed_dim)

    def forward(self, batch):
        outputs = {
            "text_embed": None, "image_embed": None, "audio_embed": None,
            "depth_embed": None, "tactile_embed": None, "spatial_embed": None
        }
        
        # 1. Text (CLIP Text Encoder)
        if "input_ids" in batch:
            text_outputs = self.clip.get_text_features(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("text_attention_mask")
            )
            outputs["text_embed"] = F.normalize(text_outputs, dim=-1)

        # 2. RGB (CLIP Vision Encoder)
        if "pixel_values" in batch:
            img_feat = self.clip.get_image_features(pixel_values=batch["pixel_values"])
            outputs["image_embed"] = F.normalize(img_feat, dim=-1)

        # 3. Audio
        if "audio_values" in batch:
            audio_values = batch["audio_values"]
            mask = batch.get("audio_attention_mask")
            B, _, T = audio_values.shape
            if mask is None or mask.shape[-1] != T:
                mask = torch.ones((B, T), dtype=torch.long, device=audio_values.device)
            feat_lens = mask.sum(-1)
            packed = audio_values.permute(0, 2, 1)[mask.bool()].permute(1, 0)
            
            audio_out = self.audio_encoder(input_features=packed, feature_lens=feat_lens)
            out_lens = _get_feat_extract_output_lengths(feat_lens)
            vectors, start = [], 0
            for l in out_lens.tolist():
                vectors.append(audio_out.last_hidden_state[start : start + l].mean(dim=0))
                start += l
            outputs["audio_embed"] = F.normalize(self.audio_proj(torch.stack(vectors, dim=0)), dim=-1)

        # 4-6. Depth, Tactile, Spatial
        
        outputs["logit_scale"] = self.clip.logit_scale.exp().clamp(max=100)
        return outputs
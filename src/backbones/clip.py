"""CLIP ViT-B/32 visual encoder wrapper for face verification."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "checkpoints")

# CLIP (OpenAI) normalization — ImageNet stats
_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def _resolve_pretrained(model_name, pretrained):
    """Return local checkpoint path if cached, else return pretrained tag."""
    fname = f"clip-{model_name.lower().replace('-', '')}-{pretrained}.bin"
    local = os.path.join(_CACHE_DIR, fname)
    if os.path.exists(local):
        return local
    return pretrained


class CLIPBackbone(nn.Module):
    """Frozen CLIP visual encoder.

    Accepts [B, 3, 112, 112] in [-1, 1] (pipeline convention).
    Internally resizes to 224×224 and applies CLIP normalization.
    Returns [B, embedding_dim] (not L2-normalized — done externally).
    """

    def __init__(self, model_name="ViT-B-32", pretrained="openai"):
        super().__init__()
        resolved = _resolve_pretrained(model_name, pretrained)
        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=resolved,
        )
        self.visual = model.visual
        self.visual.requires_grad_(False)
        self.register_buffer("clip_mean", _CLIP_MEAN)
        self.register_buffer("clip_std", _CLIP_STD)

    def forward(self, x):
        # Undo pipeline norm [-1,1] → [0,1]
        x = x * 0.5 + 0.5
        # Resize 112 → 224
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        # Apply CLIP normalization
        x = (x - self.clip_mean) / self.clip_std
        return self.visual(x)


def build_clip(cfg):
    model_name = cfg["backbone"].get("clip_model", "ViT-B-32")
    pretrained = cfg["backbone"].get("clip_pretrained", "openai")
    backbone = CLIPBackbone(model_name, pretrained)

    lora_cfg = cfg["backbone"].get("lora")
    if lora_cfg:
        from src.backbones.lora import inject_lora
        n = inject_lora(
            backbone.visual, lora_cfg["targets"],
            r=lora_cfg["rank"], alpha=lora_cfg["alpha"],
        )
        print(f"  LoRA: injected {n} adapters (rank={lora_cfg['rank']})")
        trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        total = sum(p.numel() for p in backbone.parameters())
        print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return backbone

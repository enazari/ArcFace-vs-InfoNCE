"""DINOv2 ViT-B/14 wrapper for face verification."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "checkpoints")
_HUB_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "hub")

# ImageNet normalization
_IN_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IN_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class DINOv2Backbone(nn.Module):
    """Frozen DINOv2 ViT-B/14 visual encoder.

    Accepts [B, 3, 112, 112] in [-1, 1] (pipeline convention).
    Internally resizes to 224x224 and applies ImageNet normalization.
    Returns [B, 768] (not L2-normalized).
    """

    def __init__(self):
        super().__init__()
        torch.hub.set_dir(_HUB_DIR)
        local_weights = os.path.join(_CACHE_DIR, "dinov2-vitb14.pth")
        if os.path.exists(local_weights):
            self.dino = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14",
                pretrained=False, verbose=False,
            )
            state = torch.load(local_weights, map_location="cpu", weights_only=True)
            self.dino.load_state_dict(state)
        else:
            self.dino = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14", verbose=False,
            )
        self.dino.requires_grad_(False)
        self.register_buffer("img_mean", _IN_MEAN)
        self.register_buffer("img_std", _IN_STD)

    def forward(self, x):
        x = x * 0.5 + 0.5  # [-1,1] -> [0,1]
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self.img_mean) / self.img_std
        return self.dino(x)


def build_dino(cfg):
    backbone = DINOv2Backbone()

    lora_cfg = cfg["backbone"].get("lora")
    if lora_cfg:
        from src.backbones.lora import inject_lora
        n = inject_lora(
            backbone.dino, lora_cfg["targets"],
            r=lora_cfg["rank"], alpha=lora_cfg["alpha"],
        )
        print(f"  LoRA: injected {n} adapters (rank={lora_cfg['rank']})")
        trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        total = sum(p.numel() for p in backbone.parameters())
        print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return backbone

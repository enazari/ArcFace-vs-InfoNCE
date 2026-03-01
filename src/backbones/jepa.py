"""I-JEPA ViT-H/14 wrapper for face verification."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.backbones.vit import VisionTransformer

_IJEPA_URL = "https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar"
_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "checkpoints")

# ImageNet normalization
_IN_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IN_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _download_checkpoint():
    os.makedirs(_CACHE_DIR, exist_ok=True)
    fname = os.path.join(_CACHE_DIR, "ijepa-vith14-300e.pth.tar")
    if not os.path.exists(fname):
        print(f"  Downloading I-JEPA ViT-H/14 checkpoint...")
        torch.hub.download_url_to_file(_IJEPA_URL, fname)
    return fname


class IJEPABackbone(nn.Module):
    """Frozen I-JEPA ViT-H/14 visual encoder.

    Accepts [B, 3, 112, 112] in [-1, 1] (pipeline convention).
    Internally resizes to 224x224 and applies ImageNet normalization.
    Returns [B, 1280] (not L2-normalized).
    """

    def __init__(self):
        super().__init__()
        self.vit = VisionTransformer(
            img_size=224, patch_size=14,
            embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4.0,
        )
        ckpt_path = _download_checkpoint()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt["target_encoder"]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        self.vit.load_state_dict(state)
        self.vit.requires_grad_(False)
        self.register_buffer("img_mean", _IN_MEAN)
        self.register_buffer("img_std", _IN_STD)

    def forward(self, x):
        x = x * 0.5 + 0.5  # [-1,1] -> [0,1]
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self.img_mean) / self.img_std
        return self.vit(x)


def build_jepa(cfg):
    backbone = IJEPABackbone()

    lora_cfg = cfg["backbone"].get("lora")
    if lora_cfg:
        from src.backbones.lora import inject_lora
        n = inject_lora(
            backbone.vit, lora_cfg["targets"],
            r=lora_cfg["rank"], alpha=lora_cfg["alpha"],
        )
        print(f"  LoRA: injected {n} adapters (rank={lora_cfg['rank']})")
        trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        total = sum(p.numel() for p in backbone.parameters())
        print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return backbone

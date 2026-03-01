"""Build backbone from config."""

import torch.nn as nn

from src.backbones.resnet import iresnet50


def build_backbone(cfg: dict) -> nn.Module:
    name = cfg["backbone"]["name"]
    emb  = cfg["backbone"]["embedding_dim"]
    drop = cfg["backbone"]["dropout"]

    if name == "resnet50":
        return iresnet50(embedding_dim=emb, dropout=drop)

    elif name == "clip":
        from src.backbones.clip import build_clip   # Phase 2
        return build_clip(cfg)

    elif name == "dino":
        from src.backbones.dino import build_dino   # Phase 2
        return build_dino(cfg)

    elif name == "jepa":
        from src.backbones.jepa import build_jepa   # Phase 2
        return build_jepa(cfg)

    else:
        raise ValueError(f"Unknown backbone: {name}")

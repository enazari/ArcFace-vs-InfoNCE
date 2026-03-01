"""Build loss from config."""

import torch.nn as nn

from src.losses.arcface import ArcFaceLoss


def build_loss(cfg: dict) -> nn.Module:
    name = cfg["loss"]["name"]

    if name == "infonce":
        from src.losses.contrastive import InfoNCELoss
        return InfoNCELoss(temperature=cfg["loss"]["temperature"])

    scale  = cfg["loss"]["scale"]
    margin = cfg["loss"]["margin"]

    if name == "arcface":
        return ArcFaceLoss(scale=scale, margin=margin)

    elif name == "arcface_infonce":
        from src.losses.infonce import ArcFaceInfoNCELoss
        return ArcFaceInfoNCELoss(
            scale=scale, margin=margin,
            lam=cfg["loss"]["lambda_contrastive"],
            temperature=cfg["loss"]["temperature"],
        )

    elif name == "arcface_repulsion":
        from src.losses.repulsion import ArcFaceRepulsionLoss
        return ArcFaceRepulsionLoss(
            scale=scale, margin=margin,
            lam=cfg["loss"]["lambda_repulsion"],
        )

    else:
        raise ValueError(f"Unknown loss: {name}")

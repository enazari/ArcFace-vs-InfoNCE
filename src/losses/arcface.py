"""ArcFace margin loss."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    """
    Additive angular margin loss (ArcFace).

    Expects cosine logits from ArcFaceHead (no scale applied yet).
    Applies angular margin to the target class, scales all logits by s,
    then computes cross-entropy.
    """

    def __init__(self, scale: float, margin: float):
        super().__init__()
        self.scale  = scale
        self.margin = margin
        self.cos_m  = math.cos(margin)
        self.sin_m  = math.sin(margin)
        self.th     = math.cos(math.pi - margin)  # cos(pi - m) threshold
        self.mm     = math.sin(math.pi - margin) * margin

    def forward(self, cosine: torch.Tensor, labels: torch.Tensor,
                embeddings: torch.Tensor = None) -> torch.Tensor:
        idx = torch.arange(cosine.size(0), device=cosine.device)

        # Target cosine values, clamped for numerical safety
        cos_theta = cosine[idx, labels].clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # cos(theta + m) = cos*cos_m - sin*sin_m
        sin_theta = torch.sqrt(1.0 - cos_theta ** 2)
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m

        # Avoid gradient issues when theta + m > pi
        cos_theta_m = torch.where(cos_theta > self.th, cos_theta_m,
                                  cos_theta - self.mm)

        # Replace target logits
        logits = cosine.clone()
        logits[idx, labels] = cos_theta_m.to(logits.dtype)

        return F.cross_entropy(logits * self.scale, labels)

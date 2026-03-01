"""ArcFace + InfoNCE (CLIP-style paired contrastive loss).

Loss = L_arcface + lambda_contrastive * L_infonce

Batch structure: P identities × 2 images (PairSampler).
For each image i, its positive is the paired image of the same identity.
All other 2P-2 images are negatives.

L_infonce operates on L2-normalized embeddings before the ArcFace head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.arcface import ArcFaceLoss


class ArcFaceInfoNCELoss(nn.Module):

    def __init__(self, scale: float, margin: float, lam: float, temperature: float):
        super().__init__()
        self.arcface = ArcFaceLoss(scale, margin)
        self.lam = lam
        self.temperature = temperature

    def forward(self, cosine: torch.Tensor, labels: torch.Tensor,
                embeddings: torch.Tensor = None) -> torch.Tensor:
        arc_loss = self.arcface(cosine, labels)
        if embeddings is None:
            return arc_loss

        B = embeddings.size(0)
        sim = embeddings @ embeddings.T / self.temperature  # [2P, 2P]

        # Mask self-similarity
        sim.fill_diagonal_(float('-inf'))

        # Positive partner indices: 0↔1, 2↔3, 4↔5, ...
        targets = torch.arange(B, device=sim.device)
        targets[0::2] = torch.arange(1, B, 2, device=sim.device)
        targets[1::2] = torch.arange(0, B, 2, device=sim.device)

        infonce = F.cross_entropy(sim, targets)
        return arc_loss + self.lam * infonce

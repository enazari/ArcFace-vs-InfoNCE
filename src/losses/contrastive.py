"""Pure InfoNCE (CLIP-style) contrastive loss.

No classification head — operates directly on L2-normalized embeddings.
Batch structure: P identities × 2 images (PairSampler).
Positive pairs at adjacent indices: (0,1), (2,3), ...
All other 2P-2 images serve as negatives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):

    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor,
                labels: torch.Tensor = None) -> torch.Tensor:
        B = embeddings.size(0)
        sim = embeddings @ embeddings.T / self.temperature
        sim.fill_diagonal_(float('-inf'))

        targets = torch.arange(B, device=sim.device)
        targets[0::2] = torch.arange(1, B, 2, device=sim.device)
        targets[1::2] = torch.arange(0, B, 2, device=sim.device)

        return F.cross_entropy(sim, targets)

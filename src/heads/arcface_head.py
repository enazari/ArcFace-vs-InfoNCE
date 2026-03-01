"""ArcFace classification head."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    """
    FC layer with L2-normalized weights.

    Returns cosine logits (scale and margin applied separately in the loss).
    Input embeddings must already be L2-normalized (output of IResNet.features).
    """

    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=1)
        weight = F.normalize(self.weight, dim=1)
        return F.linear(x, weight)  # cosine similarity logits

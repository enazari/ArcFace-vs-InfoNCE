"""LoRA (Low-Rank Adaptation) injection for nn.Linear layers.

Replaces target nn.Linear modules with LoRALinear wrappers that add a
trainable low-rank path while keeping the original weights frozen.
"""

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with a parallel low-rank adapter."""

    def __init__(self, original: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.original = original
        self.original.requires_grad_(False)
        self.A = nn.Linear(original.in_features, r, bias=False)
        self.B = nn.Linear(r, original.out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight)
        nn.init.zeros_(self.B.weight)
        self.scale = alpha / r

    def forward(self, x):
        return self.original(x) + self.B(self.A(x)) * self.scale


def inject_lora(model: nn.Module, target_names: list[str], r: int, alpha: float):
    """Replace matching nn.Linear layers with LoRALinear wrappers.

    Args:
        model: The model to inject LoRA into.
        target_names: List of substrings to match against module names
                      (e.g. ["out_proj", "c_fc", "c_proj"]).
        r: LoRA rank.
        alpha: LoRA scaling factor.

    Returns:
        Number of layers replaced.
    """
    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(t in name for t in target_names):
            continue
        # Navigate to the parent module and replace the attribute
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], LoRALinear(module, r, alpha))
        replaced += 1
    return replaced

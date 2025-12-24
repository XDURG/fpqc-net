from __future__ import annotations
import torch
from torch import nn

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(-ce)
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

def masked_bce_with_logits(quality_logits: torch.Tensor,
                           quality_target: torch.Tensor,
                           quality_mask: torch.Tensor) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(
        quality_logits, quality_target.float(), reduction="none"
    )  # [B, 5]
    masked = bce * quality_mask.float()
    denom = torch.clamp(quality_mask.float().sum(), min=1.0)
    return masked.sum() / denom

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn
import torchvision

from .attention import CBAM

@dataclass
class FPQCNetConfig:
    num_planes: int = 6
    quality_heads: int = 5
    image_size: int = 224
    backbone: str = "mobilenet_v3_small"
    pretrained: bool = False
    dropout: float = 0.2

class FPQCNet(nn.Module):
    def __init__(self, cfg: FPQCNetConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.backbone != "mobilenet_v3_small":
            raise ValueError("FPQCNet uses MobileNetV3-Small as default backbone. Use baselines.py for others.")

        weights = None
        if cfg.pretrained:
            weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT

        m = torchvision.models.mobilenet_v3_small(weights=weights)
        self.backbone = m.features
        self.backbone_out_ch = m.classifier[0].in_features

        self.attn = CBAM(self.backbone_out_ch, reduction=16, sa_kernel=7)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(cfg.dropout),
        )

        self.plane_head = nn.Linear(self.backbone_out_ch, cfg.num_planes)

        self.quality_heads = nn.ModuleList([
            nn.Linear(self.backbone_out_ch, 1) for _ in range(cfg.quality_heads)
        ])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.backbone(x)
        feat = self.attn(feat)
        pooled = self.pool(feat)
        z = self.shared(pooled)

        plane_logits = self.plane_head(z)
        quality_logits = torch.cat([h(z) for h in self.quality_heads], dim=1)

        return {
            "plane_logits": plane_logits,
            "quality_logits": quality_logits,
        }

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.forward(x)
        plane_prob = torch.softmax(out["plane_logits"], dim=1)
        quality_prob = torch.sigmoid(out["quality_logits"])
        return {
            "plane_prob": plane_prob,
            "quality_prob": quality_prob,
        }

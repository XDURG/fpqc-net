from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import torch
from torch import nn
import torchvision

@dataclass
class BaselineConfig:
    backbone: str = "resnet34"
    num_planes: int = 6
    quality_heads: int = 5
    pretrained: bool = False
    dropout: float = 0.2

class MultiTaskBaseline(nn.Module):
    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.cfg = cfg

        self.backbone, out_dim = self._build_backbone(cfg.backbone, cfg.pretrained)
        self.dropout = nn.Dropout(cfg.dropout)
        self.plane_head = nn.Linear(out_dim, cfg.num_planes)
        self.quality_heads = nn.ModuleList([nn.Linear(out_dim, 1) for _ in range(cfg.quality_heads)])

    def _build_backbone(self, name: str, pretrained: bool):
        name = name.lower()
        weights = None

        if name == "densenet121":
            if pretrained: weights = torchvision.models.DenseNet121_Weights.DEFAULT
            m = torchvision.models.densenet121(weights=weights)
            out_dim = m.classifier.in_features
            m.classifier = nn.Identity()
            return m, out_dim

        if name == "efficientnet_b0":
            if pretrained: weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
            m = torchvision.models.efficientnet_b0(weights=weights)
            out_dim = m.classifier[1].in_features
            m.classifier = nn.Identity()
            return m, out_dim

        if name == "resnet34":
            if pretrained: weights = torchvision.models.ResNet34_Weights.DEFAULT
            m = torchvision.models.resnet34(weights=weights)
            out_dim = m.fc.in_features
            m.fc = nn.Identity()
            return m, out_dim

        if name == "xception":
            m = _XceptionLite(out_dim=512)
            return m, 512

        raise ValueError(f"Unsupported backbone: {name}")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.backbone(x)
        z = self.dropout(z)

        plane_logits = self.plane_head(z)
        quality_logits = torch.cat([h(z) for h in self.quality_heads], dim=1)
        return {"plane_logits": plane_logits, "quality_logits": quality_logits}

class _XceptionLite(nn.Module):
    def __init__(self, out_dim: int = 512):
        super().__init__()
        def dw_sep(in_ch, out_ch, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            dw_sep(32, 64, 1),
            dw_sep(64, 128, 2),
            dw_sep(128, 256, 2),
            dw_sep(256, 512, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Identity()
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return x

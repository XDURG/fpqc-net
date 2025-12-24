from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import FocalLoss, masked_bce_with_logits

@dataclass
class TrainConfig:
    epochs: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cpu"
    focal_gamma: float = 2.0

def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    cfg: TrainConfig) -> Dict[str, float]:
    model.train()
    device = torch.device(cfg.device)

    focal = FocalLoss(gamma=cfg.focal_gamma)
    total_loss = 0.0
    total_plane = 0.0
    total_quality = 0.0
    n_steps = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        x = batch["image"].to(device)
        plane_y = batch["plane_label"].to(device)
        q_y = batch["quality_target"].to(device)       # [B,5]
        q_mask = batch["quality_mask"].to(device)      # [B,5]

        out = model(x)
        plane_logits = out["plane_logits"]
        quality_logits = out["quality_logits"]

        loss_plane = focal(plane_logits, plane_y)
        loss_quality = masked_bce_with_logits(quality_logits, q_y, q_mask)
        loss = loss_plane + loss_quality

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_plane += float(loss_plane.item())
        total_quality += float(loss_quality.item())
        n_steps += 1

        pbar.set_postfix({
            "loss": total_loss / n_steps,
            "plane": total_plane / n_steps,
            "qual": total_quality / n_steps,
        })

    return {
        "loss": total_loss / max(n_steps, 1),
        "loss_plane": total_plane / max(n_steps, 1),
        "loss_quality": total_quality / max(n_steps, 1),
    }

@torch.no_grad()
def eval_one_epoch(model: nn.Module,
                   loader: DataLoader,
                   cfg: TrainConfig) -> Dict[str, float]:
    model.eval()
    device = torch.device(cfg.device)

    focal = FocalLoss(gamma=cfg.focal_gamma)
    total_loss = 0.0
    total_plane = 0.0
    total_quality = 0.0
    n_steps = 0

    for batch in tqdm(loader, desc="eval", leave=False):
        x = batch["image"].to(device)
        plane_y = batch["plane_label"].to(device)
        q_y = batch["quality_target"].to(device)
        q_mask = batch["quality_mask"].to(device)

        out = model(x)
        loss_plane = focal(out["plane_logits"], plane_y)
        loss_quality = masked_bce_with_logits(out["quality_logits"], q_y, q_mask)
        loss = loss_plane + loss_quality

        total_loss += float(loss.item())
        total_plane += float(loss_plane.item())
        total_quality += float(loss_quality.item())
        n_steps += 1

    return {
        "loss": total_loss / max(n_steps, 1),
        "loss_plane": total_plane / max(n_steps, 1),
        "loss_quality": total_quality / max(n_steps, 1),
    }

def fit(model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: TrainConfig) -> Dict[str, Any]:
    device = torch.device(cfg.device)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = {"train": [], "val": []}
    for epoch in range(1, cfg.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, cfg)
        va = eval_one_epoch(model, val_loader, cfg)
        history["train"].append(tr)
        history["val"].append(va)
        print(f"[Epoch {epoch}/{cfg.epochs}] "
              f"train loss={tr['loss']:.4f} | val loss={va['loss']:.4f}")
    return history

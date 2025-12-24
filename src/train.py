import argparse
import torch
from torch.utils.data import DataLoader

from fpqcnet.models.fpqcnet import FPQCNet, FPQCNetConfig
from fpqcnet.data.dataset import FPQCDataset
from fpqcnet.engine.trainer import fit, TrainConfig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"

    model = FPQCNet(FPQCNetConfig(pretrained=False))
    train_ds = FPQCDataset(n=64)
    val_ds = FPQCDataset(n=32)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    cfg = TrainConfig(epochs=args.epochs, device=device)
    fit(model, train_loader, val_loader, cfg)

if __name__ == "__main__":
    main()

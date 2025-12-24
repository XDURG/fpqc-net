import argparse
import os
import csv
from glob import glob

import cv2
import torch

from fpqcnet.models.fpqcnet import FPQCNet, FPQCNetConfig
from fpqcnet.data.preprocess import preprocess_ultrasound
from fpqcnet.data.transforms import build_infer_transform, pil_from_bgr
from fpqcnet.utils.io import load_checkpoint_safely

PLANE_NAMES = ["AC", "HC", "FL", "4CH", "CL", "Other"]
QUALITY_NAMES = ["AC", "HC", "FL", "4CH", "CL"]
IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp")

def list_images(folder: str):
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob(os.path.join(folder, "**", ext), recursive=True))
    paths = sorted(set(paths))
    return paths

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True, help="Folder with images (recursive)")
    ap.add_argument("--output_csv", type=str, default="predictions.csv")
    ap.add_argument("--weights", type=str, default="", help="Optional local checkpoint path")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--image_size", type=int, default=224)
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    model = FPQCNet(FPQCNetConfig(pretrained=False, image_size=args.image_size)).to(device).eval()

    load_checkpoint_safely(model, args.weights, map_location=str(device), strict=False)

    tfm = build_infer_transform(args.image_size)

    paths = list_images(args.input_dir)
    if not paths:
        raise RuntimeError(f"No images found in {args.input_dir}")

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    header = ["path", "pred_plane", "pred_plane_prob"]
    header += [f"plane_prob_{n}" for n in PLANE_NAMES]
    header += [f"quality_prob_{n}" for n in QUALITY_NAMES]

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        for p in paths:
            img_bgr = cv2.imread(p, cv2.IMREAD_COLOR)
            if img_bgr is None:
                print(f"[WARN] Skip unreadable: {p}")
                continue

            img_bgr = preprocess_ultrasound(img_bgr)
            pil = pil_from_bgr(img_bgr)
            x = tfm(pil).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model.predict(x)
            plane_prob = pred["plane_prob"][0].detach().cpu()
            quality_prob = pred["quality_prob"][0].detach().cpu()

            top = int(torch.argmax(plane_prob).item())
            row = {
                "path": p,
                "pred_plane": PLANE_NAMES[top],
                "pred_plane_prob": float(plane_prob[top].item()),
            }
            for i, n in enumerate(PLANE_NAMES):
                row[f"plane_prob_{n}"] = float(plane_prob[i].item())
            for i, n in enumerate(QUALITY_NAMES):
                row[f"quality_prob_{n}"] = float(quality_prob[i].item())

            writer.writerow(row)

    print(f"[OK] Wrote: {args.output_csv} (n={len(paths)})")

if __name__ == "__main__":
    main()

from __future__ import annotations
import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T

def build_infer_transform(image_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

def pil_from_bgr(img_bgr: np.ndarray) -> Image.Image:
    img_rgb = img_bgr[..., ::-1].copy()
    return Image.fromarray(img_rgb)

from __future__ import annotations
import cv2
import numpy as np

def crop_ultrasound_sector(img_bgr: np.ndarray, min_area_ratio: float = 0.08) -> np.ndarray:
    if img_bgr is None:
        raise ValueError("img_bgr is None")

    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img_bgr

    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < (h * w * min_area_ratio):
        return img_bgr

    x, y, bw, bh = cv2.boundingRect(cnt)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + bw), min(h, y + bh)
    crop = img_bgr[y0:y1, x0:x1].copy()
    return crop

def mask_text_margins(img_bgr: np.ndarray, top: float = 0.12, bottom: float = 0.12, left: float = 0.10, right: float = 0.10) -> np.ndarray:
    out = img_bgr.copy()
    h, w = out.shape[:2]

    t = int(h * top)
    b = int(h * bottom)
    l = int(w * left)
    r = int(w * right)

    if t > 0: out[:t, :] = 0
    if b > 0: out[h-b:, :] = 0
    if l > 0: out[:, :l] = 0
    if r > 0: out[:, w-r:] = 0
    return out

def preprocess_ultrasound(img_bgr: np.ndarray) -> np.ndarray:
    x = crop_ultrasound_sector(img_bgr)
    x = mask_text_margins(x)
    return x

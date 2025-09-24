# src/bg/io_utils_bg.py
from __future__ import annotations
import os
import cv2
import numpy as np
from typing import Optional

def ensure_gray_u8(img: np.ndarray) -> np.ndarray:
    """Convert any image to single-channel uint8 [0..255]."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    mn, mx = float(np.min(img)), float(np.max(img))
    if mx > 255.0 or mn < 0.0:
        if mx - mn < 1e-6:
            img = np.zeros_like(img, dtype=np.float32)
        else:
            img = (img - mn) * (255.0 / (mx - mn))
    return img.astype(np.uint8)

def imread_any(path: str) -> np.ndarray:
    """Read common image formats or .npy, return gray uint8."""
    if path.lower().endswith(".npy"):
        return ensure_gray_u8(np.load(path))
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    return ensure_gray_u8(img)

def imsave_u8(path: str, img_u8: np.ndarray) -> None:
    """Save uint8 image, ensure parent dir exists."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img_u8)

def load_paired_npy_if_exists(img_path: str, suffix: str) -> Optional[np.ndarray]:
    """img.png + suffix -> img_suffix.npy가 있으면 로드해서 반환."""
    head, _ = os.path.splitext(img_path)
    cand = f"{head}_{suffix}.npy"
    return np.load(cand) if os.path.exists(cand) else None

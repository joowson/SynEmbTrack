# src/bg/simple_bg.py
from __future__ import annotations
import numpy as np
import cv2
from typing import Tuple, Optional

def generate_simple_background(
    shape: Tuple[int, int],
    base_value: float = 128.0,
    noise_std: float = 3.0,
    blur_ksize: int = 5,
    blur_sigma: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Create a simple background: constant brightness + Gaussian noise -> Gaussian blur.
    Default values mirror prior practice (mean≈128, std≈3, blur) used in your pipeline. :contentReference[oaicite:2]{index=2}
    """
    
    
    rng = rng or np.random.default_rng()
    h, w = shape
    bg = rng.normal(loc=float(base_value), scale=float(noise_std), size=(h, w)).astype(np.float32)
    bg = np.clip(bg, 0, 255).astype(np.uint8)
    if blur_ksize and blur_ksize > 1:
        bg = cv2.GaussianBlur(bg, (blur_ksize, blur_ksize), blur_sigma)
    return bg

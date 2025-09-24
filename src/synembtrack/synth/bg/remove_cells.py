# src/bg/remove_stopped.py
from __future__ import annotations
import numpy as np
import cv2
from typing import Optional

def adaptive_threshold_u8(gray_u8: np.ndarray, threshold_const: int = -4) -> np.ndarray:
    """
    Adaptive Gaussian thresholding; returns binary (0/255).
    """
    blurred = cv2.GaussianBlur(gray_u8, (5, 5), 0)
    block_size = 7
    thr = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, threshold_const
    )
    return thr

def synth_from_local_median(
    med_frame_u8: np.ndarray,
    std_frame_f32: Optional[np.ndarray] = None,
    sigma_fixed: float = 3.0
) -> np.ndarray:
    """
    Generate Normal(local_median, sigma).
    """
    local_mean = cv2.medianBlur(med_frame_u8, 13).astype(np.float32)
    if std_frame_f32 is not None:
        sigma = max(float(np.mean(std_frame_f32) / 2.0), 1.0)   # conservative lower bound
    else:
        sigma = float(sigma_fixed)                               # prior convention ~3 :contentReference[oaicite:5]{index=5}
    gen = np.random.default_rng().normal(loc=local_mean, scale=sigma).astype(np.float32)
    gen = np.clip(gen, 0, 255).astype(np.uint8)
    return gen

def remove_cells_background(
    med_frame_u8: np.ndarray,
    std_frame_f32: Optional[np.ndarray] = None,
    threshold_const: int = -4,
    dilate_iter: int = 4,
) -> np.ndarray:
    """
    1) adaptive threshold on median frame
    2) dilate mask (3×3, iter=dilate_iter)
    3) paste synthetic texture (Normal(local_median, σ)) over masked area
    
    """
    thr = adaptive_threshold_u8(med_frame_u8, threshold_const=threshold_const)
    # In prior data, bacteria may appear darker; pick mask polarity by global mean.
    mask = (thr < 10).astype(np.uint8) if np.mean(thr) > 127 else (thr > 245).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    dil = cv2.dilate(mask, kernel, iterations=max(0, dilate_iter))

    gen = synth_from_local_median(med_frame_u8, std_frame_f32)
    out = med_frame_u8.copy()
    out[dil.astype(bool)] = gen[dil.astype(bool)]
    return out

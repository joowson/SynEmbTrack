from __future__ import annotations
import os
from typing import Sequence, Tuple, Optional, List
import numpy as np

from .io_utils_bg import imread_any, imsave_u8, ensure_gray_u8, load_paired_npy_if_exists
from .simple_bg import generate_simple_background
from .remove_cells import remove_cells_background

def replace_patches_with_simple_bg(
    img_u8: np.ndarray,
    centers: Sequence[Tuple[int, int]],
    size: int = 60,
    use_image_median_for_mean_and_std: bool = True,
    blur_ksize: int = 5,
    blur_sigma: float = 0.0,
) -> np.ndarray:
    """
    Replace square patches centered at (cx, cy) with method-1 background.
    When use_image_median_for_mean_and_std=True:
      mean = median(img), noise std = median(img) 
    """
    h, w = img_u8.shape[:2]
    half = size // 2
    med = float(np.median(img_u8))
    mean = med if use_image_median_for_mean_and_std else 128.0
    std  = med if use_image_median_for_mean_and_std else 3.0

    out = img_u8.copy()
    for (cx, cy) in centers:
        x0, y0 = int(cx - half), int(cy - half)
        x1, y1 = x0 + size, y0 + size
        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(w, x1); y1 = min(h, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        ph, pw = (y1 - y0), (x1 - x0)
        patch = generate_simple_background(
            (ph, pw), base_value=mean, noise_std=std,
            blur_ksize=blur_ksize, blur_sigma=blur_sigma
        )
        out[y0:y1, x0:x1] = patch
    return out

def _paired(path: str, suffix: str) -> Optional[np.ndarray]:
    head, _ = os.path.splitext(path)
    cand = f"{head}_{suffix}.npy"
    return np.load(cand) if os.path.exists(cand) else None

def extract_background_for_paths(
    input_paths: Sequence[str],
    method: int,
    out_dir: str,
    # method-1 params
    base_value: float = 128.0,
    noise_std: float  = 3.0,
    blur_ksize: int   = 5,
    blur_sigma: float = 0.0,
    # method-3 params
    threshold_const: int = -4,
    dilate_iter: int     = 4,
    # patch option
    crop_centers: Optional[Sequence[Tuple[int, int]]] = None,
    crop_size: int = 60,
    patch_use_image_median: bool = True,
) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    saved: List[str] = []

    for p in input_paths:
        img = imread_any(p)  # uint8 gray
        h, w = img.shape[:2]

        if method == 1:
            out = generate_simple_background(
                (h, w), base_value=base_value, noise_std=noise_std,
                blur_ksize=blur_ksize, blur_sigma=blur_sigma
            )

        elif method == 2:
            med = _paired(p, "med_frame")
            std = _paired(p, "std_frame")
            med_u8 = ensure_gray_u8(med) if med is not None else img
            std_f32 = std.astype(np.float32) if std is not None else None
            out = remove_cells_background(
                med_frame_u8=med_u8,
                std_frame_f32=std_f32,
                threshold_const=threshold_const,
                dilate_iter=dilate_iter,   
            )
                
        elif method == 3:
            # z-stack median (not implemented)
            raise NotImplementedError("Method 2 is not implemented yet.")

        else:
            raise ValueError(f"Unknown method: {method}")

        if crop_centers:
            out = replace_patches_with_simple_bg(
                out, centers=crop_centers, size=crop_size,
                use_image_median_for_mean_and_std=patch_use_image_median,
                blur_ksize=blur_ksize, blur_sigma=blur_sigma,
            )

        stem = os.path.splitext(os.path.basename(p))[0]
        out_path = os.path.join(out_dir, f"{stem}_bg_m{method}.png")
        imsave_u8(out_path, out)
        saved.append(out_path)

    return saved

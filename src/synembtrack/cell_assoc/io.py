
# io.py
from pathlib import Path
from typing import List
import numpy as np
import tifffile as tiff

def list_mask_frames(masks_dir: Path) -> List[Path]:
    paths = sorted(masks_dir.glob("frame_*.tif"))
    if not paths:
        raise FileNotFoundError(f"No masks found under: {masks_dir}")
    return paths

def read_int_label_tif(path: Path) -> np.ndarray:
    arr = tiff.imread(str(path))
    if arr.ndim > 2:
        arr = arr.squeeze()
    if np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.uint16)
    return arr

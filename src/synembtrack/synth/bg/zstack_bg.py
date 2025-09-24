# src/bg/zstack_bg.py
from __future__ import annotations
import numpy as np

def background_from_zstack_median(frames: list[np.ndarray]) -> np.ndarray:
    """
    Use temporal (z) median to estimate static background / remove fixed parts.
    Intentionally left unimplemented until spec is finalized.
    """
    raise NotImplementedError("Method 2 (z-stack median) not implemented yet.")

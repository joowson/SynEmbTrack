from typing import Sequence
import numpy as np
import cv2
from PIL import Image
import colorsys

def stackBinaryMasks(masks: Sequence[Image.Image]) -> np.ndarray:
    if not masks:
        raise ValueError("masks is empty")
    h, w = np.array(masks[0]).shape
    stacked = np.zeros((h, w), dtype=np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    for m in masks:
        m = np.array(m)
        dil = cv2.dilate(m, kernel, iterations=1)
        stacked = np.where(dil != 0, 0, stacked)
        stacked = np.where(m != 0, 1, stacked)
    return stacked

def stackBinaryMasks_and_assignIntVal(masks: Sequence[Image.Image]) -> np.ndarray:
    if not masks:
        raise ValueError("masks is empty")
    h, w = np.array(masks[0]).shape
    stacked = np.zeros((h, w), dtype=np.uint8)
    interval = 2
    vals = list(range(255 - len(masks) * interval, 256, interval))
    for val, m in zip(vals, masks):
        m = np.array(m)
        stacked = np.where(m != 0, val, stacked)
    return stacked

def hsv_random_colors(n: int, seed: int = 0):
    import numpy as np
    np.random.seed(seed)
    hsv_colors = [(i / max(n,1), 1, 1) for i in range(max(n,1))]
    np.random.shuffle(hsv_colors)
    return [colorsys.hsv_to_rgb(*c) for c in hsv_colors]

def stackBinaryMasks_and_assignColor(masks: Sequence[Image.Image], seed: int = 0):
    import numpy as np
    if not masks:
        raise ValueError("masks is empty")
    h, w = np.array(masks[0]).shape
    color_mask = np.zeros((h, w, 3), dtype=np.float32)
    colors = hsv_random_colors(len(masks), seed=seed)
    for m, color in zip(masks, colors):
        mb = np.array(m, dtype=bool)
        for c in range(3):
            color_mask[:, :, c] = np.where(mb, color[c], color_mask[:, :, c])
    return color_mask  # float in [0,1]

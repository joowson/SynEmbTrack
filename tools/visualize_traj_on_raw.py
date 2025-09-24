# tools/visualize_traj_on_raw.py
# -*- coding: utf-8 -*-
"""
Visualize tracking trajectories on raw image frames.

Usage:
    python visualize_traj_on_raw.py /path/to/trajectory.csv /path/to/frames_dir /path/to/out_dir

Assumptions:
- Frames are named like "frame_0001.tif" (same pattern used in the tracking pipeline).
- Trajectory CSV has header: ['TIME_frame','TRACK_ID','X_(com)','Y_(com)', ...]
  which matches the writer in AssignID.ReID.  We only rely on TIME_frame, TRACK_ID, X_(com), Y_(com).
- Output images are written as PNG with the same frame index.

Notes:
- Comments are in English as requested.
- Keep CLI minimal (only three required arguments).
"""

import os
import sys
import glob
import math
import csv
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

from tqdm import tqdm


def _stable_color(id_int: int) -> Tuple[int, int, int]:
    """Return a stable BGR color for a given track id."""
    # Simple hashing → HSV → BGR conversion for wide color spread
    rng = (id_int * 2654435761) & 0xFFFFFFFF
    h = (rng % 180)  # OpenCV Hue range [0,180)
    s = 200
    v = 255
    hsv = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _draw_one(
    img: np.ndarray,
    items: List[Tuple[int, float, float]],
    side: int = 10,
) -> np.ndarray:
    """Draw one frame's overlays.

    Args:
        img: raw image (H,W[,3])
        items: list of (track_id, x, y)
        side: half side length for small square marker
    """
    if img.ndim == 2:
        canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        canvas = img.copy()

    for tid, x, y in items:
        color = _stable_color(int(tid))
        xi, yi = int(round(x)), int(round(y))
        # small diamond/square marker around centroid
        poly = np.array(
            [[xi + side, yi], [xi, yi - side], [xi - side, yi], [xi, yi + side]],
            dtype=np.int32,
        )
        cv2.polylines(canvas, [poly], isClosed=True, color=color, thickness=1)

        # ID label (similar to draw_assigned style)
        cv2.putText(
            canvas,
            f"ID{int(tid)}",
            (xi + side + 5, yi),
            cv2.FONT_HERSHEY_PLAIN,
            1.0,
            color,
            lineType=cv2.LINE_AA,
        )

    return canvas


# def _load_traj(traj_csv: str) -> Dict[int, List[Tuple[int, float, float]]]:
#     """Load trajectory CSV and group by frame.

#     Returns:
#         dict: frame_index -> list of (track_id, x_com, y_com)
#     """
#     # Be tolerant to column order; select by name.
#     df = pd.read_csv(traj_csv)
#     required = ["TIME_frame", "TRACK_ID", "X_(com)", "Y_(com)"]
#     for col in required:
#         if col not in df.columns:
#             raise ValueError(
#                 f"Column '{col}' is missing in trajectory CSV. Found: {list(df.columns)}"
#             )

#     # Ensure numeric
#     df = df.astype(
#         {
#             "TIME_frame": int,
#             "TRACK_ID": int,
#             "X_(com)": float,
#             "Y_(com)": float,
#         }
#     )

#     grouped: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
#     for row in df.itertuples(index=False):
#         print(row)
#         frm = int(getattr(row, "TIME_frame"))
#         tid = int(getattr(row, "TRACK_ID"))
#         x = float(getattr(row, "X_(com)"))
#         y = float(getattr(row, "Y_(com)"))
#         grouped[frm].append((tid, x, y))
#     return grouped

def _load_traj(traj_csv: str) -> dict[int, list[tuple[int, float, float]]]:
    """Load trajectory CSV and group by frame using column slicing (robust to weird names)."""
    import pandas as pd
    from collections import defaultdict

    df = pd.read_csv(traj_csv)
    # Column names we expect to exist in the saved CSV from AssignID.ReID
    need = ["TIME_frame", "TRACK_ID", "X_com", "Y_com"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing}. Found: {list(df.columns)}")

    grouped: dict[int, list[tuple[int, float, float]]] = defaultdict(list)

    # Use numpy tuples to avoid itertuples() name-mangling like _2, _3
    for frm, tid, x, y in df[need].to_numpy():
        grouped[int(frm)].append((int(tid), float(x), float(y)))

    return grouped


def _frame_index_from_name(path: str) -> int:
    """Extract frame index from a filename like 'frame_0123.tif'."""
    name = os.path.basename(path)
    stem, _ = os.path.splitext(name)
    # Expect 'frame_####'
    parts = stem.split("_")
    if len(parts) >= 2 and parts[0] == "frame":
        try:
            return int(parts[1])
        except Exception:
            pass
    # Fallback: raise error for unexpected pattern
    raise ValueError(f"Unexpected frame filename: {name}")
    
def visualize(traj_csv: str, frames_dir: str, out_dir: str) -> None:
    """Render overlays only for frames that appear in the trajectory CSV."""
    os.makedirs(out_dir, exist_ok=True)

    # Load trajectories grouped by frame
    by_frame = _load_traj(traj_csv)

    # Frame indices that actually have trajectory data
    frame_indices = sorted(by_frame.keys())

    for frm_idx in tqdm(frame_indices):
        fpath = os.path.join(frames_dir, f"frame_{frm_idx:04d}.tif")
        if not os.path.exists(fpath):
            print(f"[WARN] Missing frame file: {fpath}")
            continue

        img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[WARN] Could not read image: {fpath}")
            continue

        items = by_frame[frm_idx]
        canvas = _draw_one(img, items)

        out_name = f"frame_{frm_idx:04d}.png"
        cv2.imwrite(os.path.join(out_dir, out_name), canvas,
                    [int(cv2.IMWRITE_PNG_COMPRESSION), 3])

    print(f"[OK] Visualization saved to: {out_dir}")




if __name__ == "__main__":
    
    target_track_dir = '../projects/JM_221202_VD01/'
    
    timestamp = '20250920_165519'
    
    traj_csv    = os.path.join(target_track_dir, f'results/associ_demoAssoci/trajectory_JM_221202_VD01_demoAssoci_{timestamp}.csv')
    frames_dir  = os.path.join(target_track_dir, 'input_raw_images')
    out_dir     = os.path.join(target_track_dir, f'results/associ_demoAssoci/visualize_{timestamp}/snapshots')
    
    visualize(traj_csv, frames_dir, out_dir)

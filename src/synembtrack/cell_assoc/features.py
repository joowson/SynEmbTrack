
# features.py
from typing import List, Tuple
import numpy as np
import cv2

def _weighted_com(coords: np.ndarray, power: int) -> Tuple[float, float]:
    cy, cx = coords.mean(axis=0)
    dy, dx = coords[:,0] - cy, coords[:,1] - cx
    r = np.hypot(dy, dx)
    w = 1.0 / np.maximum(1.0, r)
    if power == 2:
        w = w * w
    ws = w.sum() + 1e-12
    wy = float((coords[:,0] * w).sum() / ws)
    wx = float((coords[:,1] * w).sum() / ws)
    return wy, wx

def _fit_angle_to_x(cnt: np.ndarray) -> float:
    vx, vy, _, _ = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    v = np.array([float(vx), float(vy)], dtype=float)
    v /= (np.linalg.norm(v) + 1e-12)
    ang = np.arccos(np.clip(v[0], -1.0, 1.0))
    if v[1] < 0:
        ang = np.pi - ang
    return float(ang / np.pi)

def extract_instances(int_mask: np.ndarray,
                        use_mask_info: bool = True,
                        use_box_info: bool = False, use_misc_info: bool = False):
    H, W = int_mask.shape[:2]
    top = int(int_mask.max())
    if top == 0:
        return [], [], [], [], np.zeros((H, W), np.uint8)
    # mask_union = np.zeros((H, W), np.uint8)
    mask_union = np.where(int_mask>0, 1, 0).astype(np.uint8)

    coms, lbled_pts = [], []
    mask_info = [] if use_mask_info else None
    boxs      = [] if use_box_info else None
    miscels   = [] if use_misc_info else None

    for lab in range(1, top + 1):

        ### position, pt_set, and area
        ys, xs = np.where(int_mask == lab)
        area_mask = int(ys.size)
        if area_mask == 0: continue

        coords = np.stack([ys, xs], axis=-1)
        com_y, com_x = float(ys.mean()), float(xs.mean())

        # angle_x_fit = _fit_angle_to_x(cnt_merged) ### TODO:

        if use_misc_info:
            w1y, w1x = _weighted_com(coords, 1)
            w2y, w2x = _weighted_com(coords, 2)

        single = (int_mask == lab).astype(np.uint8)
        #mask_union = np.where(single == 1, 1, mask_union).astype(np.uint8)

        ### box_info calculation
        if use_box_info:
            contours, _ = cv2.findContours(single, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: continue
            cnt_merged = np.vstack(contours)
            rect = cv2.minAreaRect(cnt_merged)
            w_rect, h_rect = rect[1]
            if h_rect > w_rect:
                width, height = float(w_rect), float(h_rect)
                bbox_angle = rect[2] / 180.0
            else:
                width, height = float(h_rect), float(w_rect)
                bbox_angle = (rect[2] + 90.0) / 180.0
            angle_x_fit = _fit_angle_to_x(cnt_merged) ### TODO:

        coms.append([float(com_x), float(com_y)])
        lbled_pts.append([tuple(p) for p in coords])
        if use_mask_info: mask_info.append([area_mask])
        if use_box_info:
            boxs.append([float(bbox_angle), float(width), float(height)])
        if use_misc_info:
            miscels.append([float(w1y), float(w1x), float(w2y), float(w2x)])

    return coms, lbled_pts, mask_info, boxs, miscels, mask_union

from __future__ import annotations

from typing import List
import cv2
import numpy as np

from .config import PatchConfig


def _pick_largest_mask(mask: np.ndarray) -> np.ndarray | None:
    cnts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    H, W = mask.shape[:2]
    cx, cy = W / 2.0, H / 2.0
    best, best_dist = None, None
    for c in cnts:
        dist = cv2.pointPolygonTest(c, (cx, cy), True) * (-1)
        if best is None or dist <= best_dist:
            best, best_dist = c, dist
    if best is None:
        return None
    out = np.zeros((H, W), dtype=np.uint8)
    for y in range(H):
        for x in range(W):
            if cv2.pointPolygonTest(best, (x, y), False) >= 0:
                out[y, x] = 255
    return out


class Processor:
    def __init__(self, cfg: PatchConfig) -> None:
        self.cfg = cfg

    @staticmethod
    def _adaptive(img: np.ndarray, threshold: int, kernel_size: int) -> np.ndarray:
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        thr = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            kernel_size,
            threshold,
        )
        return thr

    def threshold_frame(self, img_u8: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        if cfg.is_obj_brighter: frame_2_thresh = img_u8
        else:    frame_2_thresh = 255 - img_u8
        thr = self._adaptive(frame_2_thresh, threshold=-cfg.src_thresh, kernel_size=cfg.kernel_size_for_thresholding)
        return thr

    def find_candidates(self, thr: np.ndarray, img_shape: tuple[int, int]) -> List[tuple[int, int, float]]:
        orgsize = 80
        contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        H, W = img_shape
        out: List[tuple[int, int, float]] = []
        for cnt in contours:
            (cx, cy), (w, h), ang = cv2.minAreaRect(cnt)
            if w * h < self.cfg.area_th:
                continue
            if not (int(orgsize / 2) < cx < W - int(orgsize / 2)):
                continue
            if not (int(orgsize / 2) < cy < H - int(orgsize / 2)):
                continue
            out.append((int(cx), int(cy), float(ang)))
        return out

    @staticmethod
    def _alpha(img: np.ndarray, img_med: float, brighter: bool) -> np.ndarray:
        if brighter:
            alpha = 255 * (img.astype(float) - img_med) / (255 - img_med + 1e-6)
        else:
            alpha = 255 * (img_med - img.astype(float)) / (img_med + 1e-6)
        alpha = np.clip(alpha, 0, 255).astype(np.uint8)
        return alpha

    def compose_alpha(self, crop_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.cfg
        I_med, I_std = float(np.median(crop_u8)), float(np.std(crop_u8))
        decision_level = I_med + I_std * ((-1) ** (cfg.is_obj_brighter + 1))
        brighter_mask = (crop_u8 >= decision_level).astype(np.uint8)
        darker_mask = 1 - brighter_mask

        seed = _pick_largest_mask((brighter_mask if cfg.is_obj_brighter else darker_mask) * 255)
        if seed is None:
            return np.zeros_like(crop_u8), np.zeros_like(crop_u8), np.zeros_like(crop_u8)

        kernel33 = np.ones((3, 3), np.uint8)
        kernel_cross33 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        body_sector = cv2.dilate(seed, kernel33, iterations=1)
        bdry_sector = cv2.dilate(body_sector, kernel_cross33, iterations=3)

        alpha_b = self._alpha(crop_u8, I_med, brighter=True)
        alpha_d = self._alpha(crop_u8, I_med, brighter=False)

        if cfg.is_obj_brighter:
            alpha_body = np.where(body_sector != 0, alpha_b, 0)
            alpha_bdry = np.where(bdry_sector != 0, alpha_d, 0)
        else:
            alpha_body = np.where(body_sector != 0, alpha_d, 0)
            alpha_bdry = np.where(bdry_sector != 0, alpha_b, 0)

        return alpha_body.astype(np.uint8), alpha_bdry.astype(np.uint8), seed.astype(np.uint8)

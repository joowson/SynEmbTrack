from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import PatchConfig
from .processor import Processor
from .io_utils import PatchWriter

ManualSpec = Dict[str, List[Tuple[int, int, Optional[float]]]]


class PatchCollector:
    def __init__(self, cfg: PatchConfig, processor: Processor, writer: PatchWriter) -> None:
        self.cfg = cfg
        self.P = processor
        self.W = writer

    @staticmethod
    def _denoise(u8: np.ndarray, h: int = 5) -> np.ndarray:
        return cv2.fastNlMeansDenoising(u8, None, h, 7, 21)

    @staticmethod
    def _dedup(pts: List[Tuple[int, int, float, str]], tol_px: int) -> List[Tuple[int, int, float, str]]:
        out: List[Tuple[int, int, float, str]] = []
        tol2 = tol_px * tol_px
        for x, y, a, p in pts:
            ok = True
            for X, Y, _, P in out:
                if P == p and (X - x) ** 2 + (Y - y) ** 2 <= tol2:
                    ok = False
                    break
            if ok:
                out.append((x, y, a, p))
        return out

    def generate(
        self,
        image_paths: Sequence[str],
        manual: Optional[ManualSpec] = None,
        mode: str = "auto",
        dedup_px: int = 8,
    ) -> int:
        cfg = self.cfg
        if not image_paths and not (manual and mode in ("manual", "hybrid")):
            raise ValueError("No inputs: image_paths empty and no manual candidates provided.")

        sample_path = image_paths[0] if image_paths else next(iter(manual.keys()))
        sample = cv2.imread(sample_path, 0)
        if sample is None:
            raise FileNotFoundError(f"Cannot read sample image: {sample_path}")
        H, W = sample.shape[:2]

        patch_limit = cfg.patch_limit
        candi_limit = patch_limit * 2

        candi_set: List[Tuple[int, int, float, str]] = []

        if manual and mode in ("manual", "hybrid"):
            for path, coords in manual.items():
                for (x, y, a) in coords:
                    ang = float(a) if a is not None else 0.0
                    candi_set.append((int(x), int(y), ang, path))

        if mode in ("auto", "hybrid"):
            for path in image_paths:
                src = cv2.imread(path, 0)
                if src is None:
                    continue
                med = float(np.median(src)) or 1.0
                src_scaled = (src / med * cfg.rescaling_med).astype(np.uint8)
                gray = (src_scaled / max(1, src_scaled.max()) * 255).astype(np.uint8)
                thr = self.P.threshold_frame(gray)
                auto_cands = self.P.find_candidates(thr, (H, W))
                for (x, y, ang) in auto_cands:
                    candi_set.append((x, y, float(ang), path))
                if len(candi_set) > candi_limit:
                    break

        if mode == "hybrid" and dedup_px > 0:
            candi_set = self._dedup(candi_set, tol_px=dedup_px)

        orgsize = 80
        R = cfg.final_crop_radi
        patch_shape = (2 * R, 2 * R, 4)

        produced = 0
        for (cx, cy, _ang, img_path) in candi_set:
            if produced >= patch_limit:
                break
            src = cv2.imread(img_path, 0)
            if src is None:
                continue

            y0, y1 = cy - orgsize // 2, cy + orgsize // 2
            x0, x1 = cx - orgsize // 2, cx + orgsize // 2
            if y0 < 0 or x0 < 0 or y1 > src.shape[0] or x1 > src.shape[1]:
                continue
            crop = src[y0:y1, x0:x1].astype(np.uint8)

            crop = self._denoise(crop, h=5)
            alpha_body, alpha_bdry, seed_mask = self.P.compose_alpha(crop)
            if seed_mask.sum() == 0:
                continue

            if cfg.is_obj_brighter:
                patch_body = np.full(patch_shape, 255, dtype=np.uint8)
                patch_bdry = np.zeros(patch_shape, dtype=np.uint8)
            else:
                patch_body = np.zeros(patch_shape, dtype=np.uint8)
                patch_bdry = np.full(patch_shape, 255, dtype=np.uint8)

            ab = alpha_body[orgsize // 2 - R : orgsize // 2 + R, orgsize // 2 - R : orgsize // 2 + R]
            ad = alpha_bdry[orgsize // 2 - R : orgsize // 2 + R, orgsize // 2 - R : orgsize // 2 + R]
            if ab.size == 0 or ad.size == 0:
                continue

            patch_body[:, :, 3] = ab
            patch_bdry[:, :, 3] = ad


            ### Visualization for the integrity check of results

            
            mask_body = (patch_body[:, :, 3] > 0).astype(np.uint8)
            R = cfg.final_crop_radi

            extension = R
            padded = np.pad(src, pad_width=extension, mode="edge")
            cx_pad, cy_pad = cx + extension, cy + extension
            img2show = padded[cy_pad - R : cy_pad + R, cx_pad - R : cx_pad + R] 
            
            # body/boundary 색상 합성
            colored = np.zeros((2*R, 2*R, 3), dtype=np.uint8)
            colored[:, :, 2] = mask_body * 255                       # red: body
            colored[:, :, 0] = (patch_bdry[:, :, 3] > 0).astype(np.uint8) * 255  # blue: boundary
            
            alpha_body_vis = cv2.cvtColor((patch_body[:, :, 3] * 5).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            alpha_bdry_vis = cv2.cvtColor((patch_bdry[:, :, 3] * 5).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            
            top = np.concatenate([
                cv2.cvtColor(img2show, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(mask_body * 255, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(np.where(mask_body != 0, (img2show + 70).clip(0,255).astype(np.uint8), img2show), cv2.COLOR_GRAY2BGR),
            ], axis=1)
            
            
            bottom = np.concatenate([alpha_body_vis, alpha_bdry_vis, colored], axis=1)
            
            merge_vis = cv2.resize(np.concatenate([top, bottom], axis=0), None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)


            self.W.save_patch(patch_body, patch_bdry, merge_vis, idx=produced)
            
            produced += 1


        ## logging
        
        self.W.write_set_manifest(
            image_paths=list(image_paths),
            manual=manual,
            mode=mode,
            dedup_px=dedup_px,
            produced=produced,
        )
        

        return produced

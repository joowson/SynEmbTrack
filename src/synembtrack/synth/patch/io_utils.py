from __future__ import annotations

import cv2
import numpy as np
import json

from .config import PatchConfig
from datetime import datetime


class PatchWriter:
    def __init__(self, cfg: PatchConfig) -> None:
        self.cfg = cfg
        self.cfg.out_patch_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.out_integrity_dir.mkdir(parents=True, exist_ok=True)


    def save_patch(self, patch_body: np.ndarray, patch_bdry: np.ndarray, merge_vis: np.ndarray, idx: int) -> None:
        cv2.imwrite(str(self.cfg.out_patch_dir / f"patch_body_{idx:04d}.png"), patch_body)
        cv2.imwrite(str(self.cfg.out_patch_dir / f"patch_bdry_{idx:04d}.png"), patch_bdry)
        cv2.imwrite(str(self.cfg.out_integrity_dir / f"patch_integrity_{idx:04d}.png"), merge_vis)
        
        

    # Log path
    def write_set_manifest(
          self,
          image_paths: list[str],
          manual: dict[str, list[tuple[int,int, float|None]]] | None,
          mode: str,
          dedup_px: int,
          produced: int,
      ) -> None:
          manifest = {
              "timestamp": datetime.now().isoformat(timespec="seconds"),
              "patch_code": self.cfg.patch_code,
              "out_dir": str(self.cfg.out_dir),
              "alpha_dir": str(self.cfg.out_patch_dir),
              "integrity_dir": str(self.cfg.out_integrity_dir),
              "mode": mode,
              "dedup_px": dedup_px,
              "input": {
                  "num_images": len(image_paths),
                  "images": image_paths,  # 필요시 길면 생략해도 됨
                  "manual_num_images": len(manual) if manual else 0,
                  "manual_total_points": sum(len(v) for v in manual.values()) if manual else 0,
              },
              "config": {
                  "is_obj_brighter": self.cfg.is_obj_brighter,
                  "area_th": self.cfg.area_th,
                  "src_thresh": self.cfg.src_thresh,
                  "kernel_size_for_thresholding": self.cfg.kernel_size_for_thresholding,
                  "final_crop_radi": self.cfg.final_crop_radi,
                  "rescaling_med": self.cfg.rescaling_med,
                  "patch_limit": self.cfg.patch_limit,
              },
              "result": {"num_patches_generated": produced},
          }
          (self.cfg.out_dir).mkdir(parents=True, exist_ok=True)
          with (self.cfg.out_dir / "manifest.json").open("w", encoding="utf-8") as f:
              json.dump(manifest, f, ensure_ascii=False, indent=2)
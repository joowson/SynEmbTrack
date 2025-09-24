from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PatchConfig:
    is_obj_brighter: int = 0
    area_th: int = 15
    src_thresh: int = 4
    kernel_size_for_thresholding: int = 15

    final_crop_radi: int = 30
    rescaling_med: int = 128

    patch_limit: int = 300

    patch_code: str = "patchSet001"
    out_root: Path = Path("./results/extracted_patches/")

    @property
    def out_dir(self) -> Path:
        return self.out_root / f"generated_{self.patch_code}"
    @property
    def out_patch_dir(self) -> Path:
        # 실제 RGBA 패치(alpha 포함)
        return self.out_dir / "alpha_layer"      # ← 공백 없는 폴더명 권장
    @property
    def out_integrity_dir(self) -> Path:
        # body/boundary 표기된 시각화 결과
        return self.out_dir / "integrity_check"  # ← 공백 없는 폴더명 권장
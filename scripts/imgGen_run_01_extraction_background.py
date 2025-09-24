# generate_background.py
# -*- coding: utf-8 -*-

from __future__ import annotations
import os, glob
from typing import List, Tuple, Optional

from synembtrack._paths import get_raw_data_dir, get_results_dir
from synembtrack.synth.bg.extractor import extract_background_for_paths

# =========================
# MANUAL CONFIG (edit here)
# =========================


raw_data_code = "demo_2Dsuspension_25C"   # ← 원하는 실험 코드/폴더 이름
is_obj_brighter = 0

out_bg_code   = "bgSet_0001"



CONFIG = {


    "inputs": [ str(get_raw_data_dir() / raw_data_code / "images/*.tif") ],
    "out_dir": os.path.join(get_results_dir(), raw_data_code, "imgGen", f'generated_{out_bg_code}'),


    "is_obj_brighter": is_obj_brighter,

    # --------
    "method": 2,
    # BG generating methods:
    # 1) simple image
    # 2) remove_cells (default)
    # 3) z-stack median (not implemented)



    # --- method 1 params ---
    "base_value": 128.0,
    "noise_std": 3.0,
    "blur_ksize": 5,
    "blur_sigma": 0.0,

    # --- method 3 params ---
    "threshold_const": 4,
    "dilate_iter": 4,

    # --- 병행 옵션: 일부 위치를 ① 방식으로 대체 ---
    "crop_centers": [  # [(cx, cy), ...]
        # (120, 240),
        # (512, 256),
    ],
    "crop_size": 60,
    "patch_use_image_median": True,  # mean=median(img), std=median(img) 사용
}


def _expand_inputs(patterns: List[str]) -> List[str]:
    paths: List[str] = []
    for pat in patterns:
        expanded = glob.glob(pat)
        if not expanded and os.path.exists(pat):
            expanded = [pat]
        paths.extend(sorted(expanded))
    if not paths:
        raise FileNotFoundError("No inputs matched in CONFIG['inputs']")
    return paths

def main():

    ### config setting
    cfg = CONFIG
    inputs = _expand_inputs(cfg["inputs"])


    ### generate
    extract_background_for_paths(
        input_paths=inputs,
        method=cfg["method"],

        out_dir=cfg["out_dir"],

        base_value=cfg["base_value"],
        noise_std=cfg["noise_std"],
        blur_ksize=cfg["blur_ksize"],
        blur_sigma=cfg["blur_sigma"],


        threshold_const=cfg["threshold_const"]*(-1)**cfg["is_obj_brighter"],
        dilate_iter=cfg["dilate_iter"],

        crop_centers=cfg["crop_centers"] or None,
        crop_size=cfg["crop_size"],
        patch_use_image_median=cfg["patch_use_image_median"],
    )
    print(f"Done. Saved to: {cfg['out_dir']}")


if __name__ == "__main__":
    main()

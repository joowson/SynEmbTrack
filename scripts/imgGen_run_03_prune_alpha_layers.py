#!/usr/bin/env python3
"""
Alpha-layer cleanup script
- Uses the file names in the integrity_check folder (patch_integrity_XXXX.png)
  as the list of surviving IDs.
- In the alpha_layer folder, only keeps files with those IDs.
- Deletes all other patch_body_XXXX.png and patch_bdry_XXXX.png files.
"""

import os
from pathlib import Path

from synembtrack._paths import get_results_dir
# -------------------------------
# Configuration: set PATCH_CODE
# -------------------------------


raw_data_code = "demo_2Dsuspension_25C" 
PATCH_CODE = "patchSet_0002"

BASE_DIR = get_results_dir() / Path(f"{raw_data_code}/") / 'imgGen' /f"generated_{PATCH_CODE}"
INTEGRITY_DIR = BASE_DIR / "integrity_check"
ALPHA_DIR = BASE_DIR / "alpha_layer"



def main():
    
    # print(ALPHA_DIR)
    # 1) Collect surviving IDs from integrity_check
    keep_ids = set()
    for f in INTEGRITY_DIR.glob("patch_integrity_*.png"):
        num = f.stem.split("_")[-1]  # Extract XXXX part
        keep_ids.add(num)

    print(f"[INFO] {len(keep_ids)} IDs survived integrity check.")

    # 2) Iterate over alpha_layer and delete unwanted files
    removed = 0
    for f in ALPHA_DIR.glob("*.png"):
        num = f.stem.split("_")[-1]
        if num not in keep_ids:
            f.unlink()
            removed += 1
            print("  deleted:", f.name)

    print(f"[INFO] Done. {removed} files ({removed/2} patches) deleted, {len(keep_ids)} IDs kept.")

if __name__ == "__main__":
    main()

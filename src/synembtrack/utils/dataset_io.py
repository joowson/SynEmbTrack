# src/synembtrack/utils/dataset_io.py

from __future__ import annotations
from pathlib import Path
import shutil
from typing import Union, Tuple

# If you already have these helpers in your package, import them instead.
# from synembtrack._path import get_results_dir

def _pick_generated_dir(imggen_root: Path, import_img_code: str) -> Path:
    """Return the most recent 'generated_{import_img_code}*' directory under imgGen.
    Raises if not found."""
    if not imggen_root.is_dir():
        raise FileNotFoundError(f"imgGen root not found: {imggen_root}")

    prefix = f"generated_{import_img_code}"
    cands = [p for p in imggen_root.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not cands:
        raise FileNotFoundError(
            f"No directory starts with '{prefix}' under: {imggen_root}"
        )

    # Pick the newest by modified time (most recent run)
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def copy_dataset_to_training(
    raw_data_code: str,
    result_archive_dir: Union[Path, str],
    
    import_train_code: str,
    import_cv_code   : str,
    
    trainingSet_code : str,
    
    skip_if_exists: bool = False,   
) -> Tuple[Path, Path]:
    
    """Move(or copy) synthesized 'images/' and 'masks/' into the training folder structure.

    Parameters
    ----------
    raw_data_code : str
        E.g., 'demo_2Dsuspension_25C'
    import_img_code : str
        E.g., 'synthSet_VSreal'
    results_dir : Path or str
        Base results directory. Must be provided explicitly.
    move_instead_of_copy : bool
        If True, move files. If False, copy files.
    skip_if_exists : bool, default=False
        If True, when destination already exists, skip copying and just return.


    Returns
    -------
    (src_dir, dst_dir) : Tuple[Path, Path]
        The chosen source dataset directory and the final destination directory.
    """
    
        
    
    src_tr_dir = _pick_generated_dir(result_archive_dir / f'{raw_data_code}'/'imgGen', import_train_code)
    src_cv_dir = _pick_generated_dir(result_archive_dir / f'{raw_data_code}'/'imgGen', import_cv_code)
    
    
    
    # # Expect images/ and masks/ inside the chosen generated folder
    # src_images = src_tr_dir / "images"
    # src_masks  = src_tr_dir / "masks"
    # if not src_images.is_dir() or not src_masks.is_dir():
    #     raise RuntimeError(
    #         f"Expected 'images/' and 'masks/' inside {src_tr_dir}, "
    #         f"found images={src_images.is_dir()} masks={src_masks.is_dir()}"
    #     )
        
        

    # Destination: results/{raw_data_code}/training/{import_img_code}/training/
    dst_dir = result_archive_dir / raw_data_code / "training" / f"dataset_{trainingSet_code}" 
    dst_training = dst_dir / 'download' / 'train'
    dst_val = dst_dir / 'download' / 'test'
    
    # # Safety check to avoid mixing with existing data
    # if dst_images.exists() or dst_masks.exists():
    #     if skip_if_exists:
    #         print(f"[Dataset exists] Skipped copying. Destination: {dst_dir}")
    #         return dst_dir.parent, dst_dir.name
    #     else:
    #         raise RuntimeError(
    #             f"Destination already contains data: {dst_dir}\n"
    #             f"Remove or choose a different import_img_code to avoid overwriting."
    #         )



    # Create destination parents
    dst_training.mkdir(parents=True, exist_ok=True)
    dst_val.mkdir(parents=True, exist_ok=True)
    
    
    # Copy to keep the original generated set intact
    shutil.copytree(src_tr_dir / "images", dst_training / "images")
    shutil.copytree(src_tr_dir / "masks", dst_training / "masks")
    
    shutil.copytree(src_cv_dir / "images", dst_val / "images")
    shutil.copytree(src_cv_dir / "masks", dst_val / "masks")
    


    print(f"[Dataset copied]")
    print(f"Source-tr:      {src_tr_dir}")
    print(f"Source-cv:      {src_cv_dir}")
    print(f"Destination-tr: {dst_training}")
    print(f"Destination-cv: {dst_val}")
    


    return dst_dir.parent, dst_dir.name

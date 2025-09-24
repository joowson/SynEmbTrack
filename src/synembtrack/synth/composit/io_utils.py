import os
from typing import Sequence, Tuple
from synembtrack.synth.composit.configs.base import Config

def mkdirs(paths: Sequence[str]) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)

def resolve_input_paths(cfg: Config) -> Tuple[str, str]:
    if not cfg.patch_code_to_use or not cfg.bg_code_to_use:
        raise ValueError("patch_code_to_use / bg_code_to_use must be set in the preset.")
    patch_dir_final = os.path.join(cfg.input_patch_root, f"generated_{cfg.patch_code_to_use}",'alpha_layer')
    bg_dir_final = os.path.join(cfg.input_bg_root, f"generated_{cfg.bg_code_to_use}", "*.png")
    return patch_dir_final, bg_dir_final
# presets_loader.py (minimal, Python 3.11+)
from typing import Any, Dict, Mapping
from types import SimpleNamespace
import os
import tomllib  # stdlib TOML parser (Py 3.11+)



# Defaults (preset can override)
DEFAULT_SYNTH: Dict[str, Any] = {
    # Dataset/paths - fixed
    "input_patch_root": "results/extracted_patches",
    "input_bg_root": "results/extracted_backgrounds",

    # Output
    "out_root": "results/image_generation",

    # -------------- default values -------------------------------
    "pjt_name": "please_write_pjt_name",
    "pxum": 5.0,                     # pixels per micron
    "intended_density": 200.0,       # cells per (100um)^2

    "seed": 100,
    "is_cv": 0,

    # Simulation
    "box_size": (200, 200),
    "fps": 52.0,
    "velocity_pxps": 50.0,
    "D_T": 1.0,
    "D_R": 1.0,
    "omega": 0.0,                    # angular velocity for chiral rotational diffusion.

    "n_step": 10,
    "skip_interval": 1,
    "draw_index": False,

    # --- for visualization
    "draw_scalebar": False,

    # --- CODES (will be filled in preset) ---
    "raw_data_code": None,
    "patch_code_to_use": None,
    "bg_code_to_use": None,
}

from dataclasses import dataclass, replace
from typing import Optional, Tuple

@dataclass(frozen=True)
class SynthDefaults:
    input_patch_root: str = "results/extracted_patches"
    input_bg_root: str = "results/extracted_backgrounds"
    out_root: str = "results/image_generation"
    #
    pjt_name: str = "please_write_pjt_name"
    pxum: float = 5.0
    intended_density: float = 200.0  # cells per (100um)^2
    seed: int = 100
    is_cv: int = 0
    
    # Simulation
    box_size: Tuple[int,int] = (200,200)
    fps: float = 52.0
    velocity_pxps: float = 50.0
    D_T: float = 1.0
    D_R: float = 1.0
    omega: float = 0.0  # angular velocity for chiral rotational diffusion.
    
    n_step: int = 10
    skip_interval: int = 1
    draw_index: bool = False
    
    # --- for visualization
    draw_scalebar: bool = False
    
    # --- CODES (will be filled in preset) ---
    raw_data_code: Optional[str] = None
    patch_code_to_use: Optional[str] = None
    bg_code_to_use: Optional[str] = None

DEFAULT_SYNTH = SynthDefaults()  # 불변 객체

def make_synth_config(**overrides):
    return replace(DEFAULT_SYNTH, **overrides)  # 새 인스턴스 반환




def _read_toml(path: str) -> Dict[str, Any]:
    """Read a TOML file into a dict."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as f:  # tomllib expects binary mode
        return tomllib.load(f) or {}

# def load_data_preset(file_path: str, name: str) -> Dict[str, Any]:
#     """Return DATA[name] dict from file_path (expects [DATA.<name>])."""
#     root = _read_toml(file_path)
#     d = root["DATA"][name]
#     # minimal coercion
#     return {
#         "px_per_um": float(d["px_per_um"]),
#         "fps": float(d["fps"]),
#         "memo": d.get("memo"),
#     }

def _get_section(root: Mapping[str, Any], *path: str) -> Mapping[str, Any]:
    cur: Any = root
    for p in path:
        if not isinstance(cur, Mapping) or p not in cur:
            raise KeyError(f"Missing section [{'.'.join(path)}] in TOML")
        cur = cur[p]
    if not isinstance(cur, Mapping):
        raise TypeError(f"Section [{'.'.join(path)}] must be a table")
    return cur



def load_synth_config(file_path: str, name: str) -> Dict[str, Any]:
    """Return SYNTH[name] dict from file_path (expects [SYNTH.<name>])."""
    root = _read_toml(file_path)
    cfg = dict(_get_section(root, "SYNTH", name))  # 또는 "COMPOSIT"

    # --- box_size: list/tuple/str → tuple[int,int] 로 정규화 ---
    bs = cfg.get("box_size")
    if isinstance(bs, (list, tuple)) and len(bs) == 2:
        cfg["box_size"] = (int(bs[0]), int(bs[1]))
    elif isinstance(bs, str):
        parts = bs.strip().strip("()[]").split(",")
        if len(parts) == 2:
            cfg["box_size"] = (int(parts[0].strip()), int(parts[1].strip()))
    # 없거나 형식이 이상하면 기본값에 맡김

    return cfg


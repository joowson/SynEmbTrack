# presets_loader.py (minimal, Python 3.11+)
from typing import Any, Dict
from types import SimpleNamespace
import os
import tomllib  # stdlib TOML parser (Py 3.11+)

# Defaults (preset can override)
DEFAULT_ASSOC: Dict[str, Any] = {
    "neighbor_px": 20.0,
    "iou_th": 0.10,
    "speed_um_s": 100.0,
    "patience_s": 0.3,
}

def _read_toml(path: str) -> Dict[str, Any]:
    """Read a TOML file into a dict."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as f:  # tomllib expects binary mode
        return tomllib.load(f) or {}

def load_data_preset(file_path: str, name: str) -> Dict[str, Any]:
    """Return DATA[name] dict from file_path (expects [DATA.<name>])."""
    root = _read_toml(file_path)
    
    print(root)
    d = root["DATA"][name]
    # minimal coercion
    return {
        "px_per_um": float(d["px_per_um"]),
        "fps": float(d["fps"]),
        "memo": d.get("memo"),
    }

def load_assoc_preset(file_path: str, name: str) -> Dict[str, Any]:
    """Return ASSOC[name] dict from file_path (expects [ASSOC.<name>])."""
    root = _read_toml(file_path)
    return dict(root["ASSOC"][name])  # overrides only

def make_assoc(assoc_cfg: Dict[str, Any], data_cfg: Dict[str, Any]) -> SimpleNamespace:
    """Merge defaults with assoc_cfg and expose helpers as callables."""
    a: Dict[str, Any] = {**DEFAULT_ASSOC, **(assoc_cfg or {})}

    def speed_px_per_frame() -> float:
        return (float(a["speed_um_s"]) / float(data_cfg["fps"])) * float(data_cfg["px_per_um"])

    def patience_frames() -> int:
        return int(float(a["patience_s"]) * float(data_cfg["fps"]))

    return SimpleNamespace(**a, speed_px_per_frame=speed_px_per_frame, patience_frames=patience_frames)

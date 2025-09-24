from pathlib import Path

# Determine the project root (2 levels up from this file: project_root/src/synembtrack/_paths.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


##TODO: CLI setting. (with returning default path if NONE)

RAW_DATA_DIR = PROJECT_ROOT / "data_raw_images"
CONFIG_DIR   = PROJECT_ROOT / "configs"
RESULTS_DIR  = PROJECT_ROOT / "projects"

def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return PROJECT_ROOT

def get_raw_data_dir() -> Path:
    """Return the absolute path to the raw_data directory."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_DATA_DIR

def get_config_dir() -> Path:
    """Return the absolute path to the configs directory."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_DIR

def get_results_dir() -> Path:
    """Return the absolute path to the results directory."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR

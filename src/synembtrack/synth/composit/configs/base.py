from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class Config:
    
    
    
    # Dataset/paths - fixed
    input_patch_root: Optional[str] = "results/extracted_patches"  
    input_bg_root: Optional[str]   = "results/extracted_backgrounds"  

    # Output
    out_root: str = "results/image_generation"
    
    
    
    
    # -------------- default values -------------------------------
    
    pjt_name: str = "please_write_pjt_name"
    pxum: float = 5.0  # pixels per micron (CHANGE THIS to your dataset's value)
    intended_density: float = 200.0  # cells per (100um)^2

    seed: int = 100 
    is_cv: int = 0
    
    
    
    # Simulation
    box_size: Tuple[int, int] = (200, 200)
    
    fps: float = 52.0
    velocity_pxps: float = 50.0
    D_T: float = 1.0
    D_R: float = 1.0
    omega: float = 0.0 ### angular velocity for chiral rotational diffusion.
    
    
    n_step: int = 10
    skip_interval: int = 1
    draw_index: bool = False

    # --- for visualization
    draw_scalebar: bool = False
    
    
    # --- CODES (will be filled in preset) ---
    patch_code_to_use: Optional[str] = None
    bg_code_to_use: Optional[str] = None

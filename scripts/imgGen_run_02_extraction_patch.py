from __future__ import annotations

from pathlib import Path

from synembtrack.synth.patch.config import PatchConfig
from synembtrack.synth.patch.processor import Processor
from synembtrack.synth.patch.io_utils import PatchWriter
from synembtrack.synth.patch.pipeline import PatchCollector, ManualSpec

from synembtrack._paths import get_raw_data_dir, get_results_dir

import os

# ---------------- USER CONFIG (edit here) ----------------

raw_data_code = "demo_2Dsuspension_25C" 

out_patch_code = "patchSet_0002"


image_path = str(get_raw_data_dir() / raw_data_code / "images" )

IMAGE_LIST = [
    os.path.join(image_path, 'frame_0080.tif'),
    os.path.join(image_path, 'frame_0090.tif')
    #...
]


### set collecting method
candidate_collectin_MODE = "auto"      # "auto" | "manual" | "hybrid"

src_thresh=4
kernel_size_for_thresholding=15

out_path = os.path.join(get_results_dir(), raw_data_code, 'imgGen')

#%% For manual mode
# Manual candidates: {img_path: [(x, y, angle_or_None), ...]}
MANUAL: ManualSpec = {
    # "/abs/path/to/frame_0001.tif": [(123, 87, None), (220, 145, 15.0)],
}

# Used only in hybrid mode
DEDUP_PX = 8       


#%%
CFG = PatchConfig(
    is_obj_brighter=0,
    src_thresh=src_thresh,
    kernel_size_for_thresholding=kernel_size_for_thresholding,
    
    final_crop_radi=30,
    area_th=15,
    rescaling_med=128,
    
    patch_limit=300,
    patch_code=out_patch_code,
    out_root=Path(out_path)
)
#%% ---------------------------------------------------------

def main() -> None:
    processor = Processor(CFG)
    writer = PatchWriter(CFG)
    collector = PatchCollector(CFG, processor=processor, writer=writer)
    produced = collector.generate(IMAGE_LIST, mode=candidate_collectin_MODE, manual=MANUAL, dedup_px=DEDUP_PX)
    print(f"Generated {produced} patches → {CFG.out_patch_dir}")


#%%
if __name__ == "__main__":
    main()

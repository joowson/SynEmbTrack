
# pipeline.py (library)
from pathlib import Path
import time, csv
import numpy as np
from tqdm import tqdm

from .io import list_mask_frames, read_int_label_tif
from .features import extract_instances
from .association import associate_cells

from synembtrack.cell_assoc.presets_loader import (
    load_data_preset, load_assoc_preset, make_assoc
)

from synembtrack._paths import get_project_root, get_results_dir, get_raw_data_dir, get_config_dir

# NEW: dynamic header by toggles

def seg_masks_dir(data_key: str, seg_key: str) -> Path:    
    return Path(get_results_dir() / f"{data_key}/segmentation/inference_{seg_key}_{data_key}/predictions")

def associ_out_dir(data_key: str, assoc_key: str, timestamp: str) -> Path:
    return Path(get_results_dir() / f"{data_key}/tracking_results/associ_{assoc_key}")


def main(
    assoc_key: str,
    use_mask_info: bool = True,
    use_box_info:  bool = False,
    use_misc_info: bool = False,
) -> Path:
    """Entry point: just pass file paths and the preset names to use."""

    ### LOAD CONFIGURATION
    data_preset_file  = get_config_dir() / 'configs_data.toml'
    assoc_preset_file = get_config_dir() / 'configs_assoc.toml'

    assoc_cfg = load_assoc_preset(assoc_preset_file, assoc_key)
    data_key = assoc_cfg.get("data_preset_name")
    if not data_key: raise KeyError(f"'data_preset' must be set in ASSOC['{assoc_key}'].")
    seg_key  = assoc_cfg.get('seg_preset_name')

    DATA  = load_data_preset(data_preset_file,  data_key)
    
    ASSOC = make_assoc(assoc_cfg, DATA)
    
    masks = list_mask_frames(seg_masks_dir(data_key, seg_key))
    
    
    ### result file setting
    TRAJ_COLS = ['TIME_frame','TRACK_ID','X_com','Y_com']
    if use_mask_info: TRAJ_COLS += ['area_mask'] #,'angle_x_fit']
    if use_box_info:  TRAJ_COLS += ['bbox_angle','bbox_w','bbox_h']
    if use_misc_info: TRAJ_COLS += ['wcom1_y','wcom1_x','wcom2_y','wcom2_x']
    TRAJ_COLS += ['associ_code']

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = associ_out_dir(data_key, assoc_key, ts)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"trajectory_{data_key}_{assoc_key}_{ts}.csv"
    with csv_path.open("w", newline="") as f:
        csv.writer(f).writerow(TRAJ_COLS)


    ### process the 1st frame 
    im0 = read_int_label_tif(masks[0])
    f0 = int(masks[0].stem.split("_")[1])
    
    coms, lbled_pts, mask_info, boxs, miscels, _ = extract_instances(im0,
                                                          use_mask_info=use_mask_info,
                                                          use_box_info=use_box_info,
                                                          use_misc_info=use_misc_info,)

    pre_associ, pre_associ_pxs, next_id = [], [], 1
    for i, (c, pts) in enumerate(zip(coms, lbled_pts)):
        row = [f0, next_id] + c  # c = [x, y]
        if use_mask_info and mask_info is not None: row += list(mask_info[i]) 
        if use_box_info and boxs is not None:       row += list(boxs[i])
        if use_misc_info and miscels is not None:   row += list(miscels[i])
        row += [0]                                 # associ_code
        pre_associ.append(row)
        # keep last element as pts for IoU usage
        pre_associ_pxs.append([f0, next_id] + c + [pts])
        next_id += 1
    pre_associ = np.array(pre_associ, dtype=object)

    with csv_path.open("a", newline="") as f: csv.writer(f).writerows(pre_associ)

    ### initialize wait_tab
    wait_tab = np.empty((0, len(TRAJ_COLS)), dtype=object)
    wait_pxs = []
    patience_frames = ASSOC.patience_frames()


    ### Loop over entire frames
    for p in tqdm(masks[1:]):
    # for p in masks[1:]:
        frm = int(p.stem.split("_")[1])
        im  = read_int_label_tif(p)
        
        coms, lbled_pts, mask_info, boxs, miscels, _ = extract_instances(im,
                                                              use_mask_info=use_mask_info,
                                                              use_box_info=use_box_info,
                                                              use_misc_info=use_misc_info,)
        if not coms: continue

        add, add_pxs, lost, wait_tab, wait_pxs, next_id = associate_cells(
            frame=frm,
            pos=coms, lbled_pts=lbled_pts, 
            pre_associ=pre_associ, pre_associ_pxs=pre_associ_pxs,
            wait_tab=wait_tab, wait_pxs=wait_pxs,
            next_id=next_id,
            neighbor_px=ASSOC.neighbor_px,
            iou_th=ASSOC.iou_th,
            speed_px_per_frame=ASSOC.speed_px_per_frame(),
            
            # new toggles + optional inputs
            use_mask=use_mask_info,
            use_box=use_box_info,
            use_misc=use_misc_info,
            mask_info=mask_info,   # can be None
            boxs=boxs,             # can be None
            miscels=miscels,       # can be None
        )

        # print(frm)
        if len(wait_tab):
            drop = [i for i, w in enumerate(wait_tab) if frm - int(w[0]) >= patience_frames]
            if drop:
                wait_tab = np.delete(wait_tab, drop, axis=0)
                wait_pxs = [w for i, w in enumerate(wait_pxs) if i not in drop]
                # wait_pxs = list(np.delete(wait_pxs, drop, axis=0))

        with csv_path.open("a", newline="") as f:
            csv.writer(f).writerows(add)

        pre_associ, pre_associ_pxs = add.copy(), add_pxs.copy()

    return csv_path

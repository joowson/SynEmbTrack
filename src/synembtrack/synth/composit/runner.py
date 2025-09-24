import os, random
from glob import glob
import numpy as np

from PIL import Image, ImageOps
import cv2

from dataclasses import replace

from synembtrack.synth.composit.presets_loader import load_synth_config, make_synth_config

from synembtrack.synth.composit.io_utils import mkdirs, resolve_input_paths
from synembtrack.synth.composit.models import RotationalDiffusion
from synembtrack.synth.composit.masks import (
    stackBinaryMasks,
    stackBinaryMasks_and_assignIntVal,
    stackBinaryMasks_and_assignColor,
)

from synembtrack._paths import get_results_dir, get_project_root #, get_raw_data_dir, get_results_dir

def run_once(synth_key: str):
    
    
    config_path = os.path.join(get_project_root(), 'configs', 'configs_synth.toml')
    
    synth_raw  = load_synth_config(config_path, synth_key)
    cfg = make_synth_config(**synth_raw)
    
    
    
    raw_data_code = cfg.raw_data_code
    

    cfg = replace(cfg, pjt_name=synth_key)  
    cfg = replace(cfg, input_patch_root = os.path.join(get_results_dir(), raw_data_code, 'imgGen'))  
    cfg = replace(cfg, input_bg_root    = os.path.join(get_results_dir(), raw_data_code, "imgGen"))  
    cfg = replace(cfg, out_root         = os.path.join(get_results_dir(), raw_data_code, "imgGen"))
    
    # print(cfg)

    
    random.seed(cfg.seed); np.random.seed(cfg.seed)



    patch_dir_final, bg_dir_final = resolve_input_paths(cfg)
    bg_files = glob(bg_dir_final, recursive=True)
    if not bg_files:
        raise FileNotFoundError(f"No background files found: {bg_dir_final}")

    # Derived counts
    box_w, box_h = cfg.box_size
    dt = 1.0 / cfg.fps
    n_particle = int(cfg.intended_density * (box_w * box_h) * (1 / 100 / cfg.pxum) ** 2)

    # Output dirs under project result/
    savepath = os.path.join(cfg.out_root, f"generated_{cfg.pjt_name}_cv{cfg.is_cv}_density{int(cfg.intended_density)}")
    mkdirs([
        os.path.join(savepath, 'images'),
        os.path.join(savepath, 'masks'),
        os.path.join(savepath, 'masks_stacked'),
        os.path.join(savepath, 'masks_stacked_integer'),
        os.path.join(savepath, 'masks_stacked_colored'),
        os.path.join(savepath, 'visual_study'),
    ])
    print(f"Save to: {savepath}")

    # Simulator
    system = RotationalDiffusion(dt, cfg.D_T, cfg.D_R, patch_dir_final, cfg.pxum, cfg.box_size, cfg.omega)
    system.draw_index = cfg.draw_index

    x_min, x_max = -system.margin, system.size_w + system.margin
    y_min, y_max = -system.margin, system.size_h + system.margin

    for _ in range(n_particle):
        system.create_particle(
            np.random.uniform(x_min, x_max),
            np.random.uniform(y_min, y_max),
            cfg.velocity_pxps,
        )


    # logs
    f_track = open(os.path.join(savepath, 'tracklet.txt'), 'w')
    f_track_GT = open(os.path.join(savepath, 'tracklet.csv'), 'w')
    f_track_GT.write("TIME_frame,TRACK_ID,X_(com),Y_(com),angle,area\n")

    # main loop
    for timestep in range(cfg.n_step):
        system.step()
        if timestep % cfg.skip_interval != 0:
            continue

        # logs
        for p in system.list_particle:
            if p.index is not None:
                f_track.write(f"{timestep:5d} {p.index:5d} {int(p.x):5d} {int(p.y):5d} {-1:5d}\n")
                f_track_GT.write(f"{timestep:5d}, {p.index:5d}, {p.x:.6f}, {p.y:.6f}, {p.phi:.6f}, {p.area:5d}\n")

        # render
        img, mask_list, _ = system.draw_screen(timestep, np.random.choice(bg_files))
        gray_img = ImageOps.grayscale(img)
        gray_img.save(os.path.join(savepath, 'images', f'frame_{timestep:04d}.tif'), format='tiff')

        if mask_list:
            # multipage tiff
            mask_list[0].save(
                os.path.join(savepath, 'masks', f'mask_{timestep:04d}.tif'),
                compression='tiff_deflate', save_all=True, append_images=mask_list[1:]
            )
            # semantic single-page 0/255
            binary255 = (stackBinaryMasks(mask_list) * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(savepath, 'masks_stacked', f'img_{timestep:04d}.tif'), binary255)
            # per-instance integer levels
            int_mask = stackBinaryMasks_and_assignIntVal(mask_list)
            cv2.imwrite(os.path.join(savepath, 'masks_stacked_integer', f'img_{timestep:04d}.tif'), int_mask)
            # colored (x2, nearest)
            color_mask = stackBinaryMasks_and_assignColor(mask_list)
            color_uint8 = (color_mask * 255).clip(0, 255).astype(np.uint8)
            from PIL import Image as _Image
            pil_col = _Image.fromarray(color_uint8).resize((gray_img.width * 2, gray_img.height * 2), resample=_Image.NEAREST)
            pil_col.save(os.path.join(savepath, 'masks_stacked_colored', f'img_{timestep:04d}.tif'))
            
            
            # visual panel
            # zeros = np.zeros((system.size_h, system.size_w), dtype=np.uint8)
            # for m in mask_list:
            #     zeros = np.where(np.array(m) == 1, zeros + 250, zeros)
            # panel = np.concatenate([np.array(gray_img), zeros], axis=1)
            
            gray_np = np.array(gray_img)           
            gray_3c = np.repeat(gray_np[..., None], 3, axis=2)    # (H, W, 3)
            panel = np.concatenate([gray_3c, color_uint8], axis=1)   # (H, 2W, 3)
            cv2.imwrite(os.path.join(savepath, 'visual_study', f'{timestep:03d}.png'), panel)


    f_track.close(); f_track_GT.close()
    print('finished')

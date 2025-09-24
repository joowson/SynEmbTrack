**synembtrack** provides the source code, scripts and demo required to reproduce the pipeline used in our paper: synthetic data generation → embedding-based instance segmentation → frame-to-frame association and trajectory extraction. The repository favors small, script-driven workflows over complex CLIs and includes a tiny demo for a quick smoke test aimed at scholarly reproducibility.

This code accompanies our manuscript “Tracking of motile bacteria with synthetic image aided instance segmentation and quantitative analysis of run-and-tumble motion” (Son et al., under review).

---

## Repository structure
```
synembtrack/
├─ configs/                     # Presets & config samples (TOML/YAML)
├─ data_raw_images/             # (tiny) demo raw images for quick try-out
├─ projects/                    # Example project folders (outputs live here)
├─ scripts/                     # Step-by-step runnable scripts (generating synthetic dataset / training / segmentation and tracking)
├─ tools/                       # Small utilities (e.g., visualization)
├─ src/
│  └─ synembtrack/              # Python package (installable via pip -e .)
│     ├─ synth/                 # Synthetic image generator
│     ├─ EmbedSeg/              # Embedding-based segmentation components
│     ├─ cell_assoc/            # Association (tracking) utilities
│     ├─ utils/                 # I/O, metrics, helpers
│     └─ cmaps/                 # Colormaps, palettes
├─ pyproject.toml               # Packaging (PEP 621)
└─ README.md
```

## Installation

We recommend creating a clean virtual environment (e.g., with **conda**) before installation:

```bash
conda create -n synembtrack python=3.10 -y
conda activate synembtrack
```

Then install the package:

```bash
git clone https://github.com/joowson/synembtrack.git
cd synembtrack

# Editable mode (recommended for development)
pip install -e .
# Or, standard installation
pip install .
```

This makes the package synembtrack importable in Python.


## Quickstart

### Smoke test (5 min)
```
python -c "import synembtrack, importlib; print('synembtrack ok')"
```


### Minimal workflow outline
To reproduce the full pipeline, run the scripts in the following order:

```bash
# 1. Synthetic data preparation
python scripts/imgGen_run_01_extraction_background.py
python scripts/imgGen_run_02_extraction_patch.py
python scripts/imgGen_run_03_prune_alpha_layers.py
python scripts/imgGen_run_04_gen_image.py

# 2. Segmentation training
python scripts/segTrain_run_01_data.py
python scripts/segTrain_run_02_train.py

# 3. Tracking
python scripts/track_run_03_predict.py
python scripts/track_run_04_run_associ.py
````

### Step overview

1. **Background extraction** – extract background from raw images.
2. **Patch extraction** – cut out cell/patch regions.
3. **Prune alpha layers** – manually inspect extracted patches in the `integrity_check/` folder and delete non-bacterial objects. The corresponding patch files in `alpha_layer/` are removed accordingly.
4. **Synthetic image generation** – build the synthetic training dataset.
5. **Prepare training data** – organize images and masks for segmentation.
6. **Train segmentation model** – train the embedding-based instance segmentation network.
7. **Predict** – apply the trained model to generate instance masks.
8. **Association** – link objects across frames and produce trajectories.

The scripts are designed to be run sequentially.
Each step produces intermediate files required for the next step.

---

### Using your own data

The repository includes sample bacterial images for quick testing.
By default, the pipeline runs on these samples.

To test with your own data:

* **Folder structure**
  Place your images under a new project directory, for example:

  ```
  data_raw_images/my_data_name/images/
  data_raw_images/my_data_name/masks/       # (if manual masks are available)
  ```
  and insert data, synth, associ information 

* **File format**
  Images should be in standard formats (`frame_XXXX.tif`).
  Masks, if provided, are recommended to be labeled as integer labels (0 = background, 1..K = instances).

* **Configuration**  
Edit the corresponding config file (under `configs/`) to match your setup.:
  ```
  configs/configs_data.toml (raw images)  
  configs/configs_synth.toml (synthetic image generation)  
  configs/configs_assoc.toml (tracking / association)  
  ```


* **Execution**
  Update the script arguments so that each stage reads from your new directory.

The outputs (synthetic images, trained models, predictions, and trajectories)
will be saved under the corresponding `projects/my_experiment/` subfolders.




---
## Appendix

### Tested environment
  
  The pipeline was validated in the following Python environment.  
  These versions are not strict requirements, but indicate the setup in which the code was tested and the paper results were reproduced.
  
  - Python 3.10
  - numpy 1.26.4
  - scipy 1.12.0
  - pandas 2.2.2
  - pillow 10.x
  - scikit-image 0.23.x
  - opencv-python 4.10.0
  - matplotlib 3.9.2
  - seaborn 0.13.x
  - tqdm 4.66.x
  - tifffile 2024.x
  - albumentations 2.0.8
  - numba 0.59.x
  - numexpr 2.10.x
  - colorspacious 1.1.2
  - torch 2.8.0
  - torchvision 0.23.0
  


### Known limitations
  
  The demo is tuned for 2D swimming bacteria; dense biofilms or extreme overlaps need further tuning.
  Association is currently heuristic (IoU).
  Training code and pretrained weights are minimal; bring your own model if needed.


### Contributing
  
  Pull requests are welcome!
  Please keep code style simple and add short English comments/docstrings.
  For features with many parameters, prefer config files over long CLI flags.

### License

  Code: MIT (unless stated otherwise inside subfolders)
  
  Third-party notice: Parts of the embedding-based segmentation approach are inspired by / derived from EmbedSeg. If you include or adapt EmbedSeg code/assets governed by CC BY-NC 4.0, that material remains non-commercial and must retain attribution. See THIRD_PARTY_LICENSES.md for details. If you plan commercial use, review third-party licenses carefully and remove/replace non-commercial components.

### Citation

  If you use this software in your research, please cite:
  
  @software{synembtrack,
    author  = {Son, Joowang and contributors},
    title   = {synembtrack: embedding-based instance segmentation and tracking for motile bacteria},
    year    = {2025},
    url     = {https://github.com/joowson/synembtrack}
  }


### Acknowledgements

We acknowledge the developers of EmbedSeg and related open-source tools in the microscopy tracking ecosystem. Community feedback and issues are very welcome.

* Questions?

Open an Issue or start a Discussion on GitHub.
Happy tracking! 🦠📈

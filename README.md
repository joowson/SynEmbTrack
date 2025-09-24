synembtrack

Synthetic data → Embedding-based instance segmentation → Cell association / trajectory extraction
A lightweight, research-oriented toolbox for tracking motile bacteria in 2D microscopy videos.

Features

Synthetic dataset builder for moderately dense, swimming bacteria

Embedding-based instance segmentation pipeline (EmbedSeg-style)

Frame-to-frame association with IoU & distance heuristics

Trajectory export to CSV/Parquet and basic visualization utilities

Reproducible project layout (src-layout packaging, scripts, configs)

Note: This repo targets research & reproducibility first. APIs are intentionally simple and avoid complex CLI with many parameters.

Repository structure
synembtrack/
├─ configs/                     # Presets & config samples (TOML/YAML)
├─ data_raw_images/             # (tiny) demo raw images for quick try-out
├─ projects/                    # Example project folders (outputs live here)
├─ scripts/                     # Step-by-step runnable scripts (training/eval)
├─ tools/                       # Small utilities (e.g., trajectory visualization)
├─ src/
│  └─ synembtrack/              # Python package (installable via pip -e .)
│     ├─ synth/                 # Synthetic image generator
│     ├─ EmbedSeg/              # Embedding-based segmentation components
│     ├─ cell_assoc/            # Association (tracking) utilities
│     ├─ utils/                 # I/O, metrics, helpers
│     └─ cmaps/                 # Colormaps, palettes
├─ pyproject.toml               # Packaging (PEP 621)
└─ README.md

Installation
1) Create environment (recommended)
# conda (example)
conda create -n synembtrack python=3.10 -y
conda activate synembtrack

2) Editable install
# from the repository root
pip install -e .


This installs synembtrack in development mode so you can edit source under src/ and use it immediately.

Quick start (5 min)
Smoke test
python -c "import synembtrack, importlib; print('synembtrack ok')"

Minimal workflow outline

Prepare data
Put a few microscopy frames (TIF/PNG) under a project folder, e.g.:

projects/demo_2Dsuspension_25C/
  ├─ images/
  └─ (optional) masks/     # if you already have instance masks


Run segmentation & save instance masks
Use the example in scripts/ to segment images into per-frame instance masks.

# example: segmentation pipeline (adjust to your script names)
python scripts/seg_run_demo.py \
  --project projects/demo_2Dsuspension_25C


Associate & export trajectories
Link instances across frames into tracks and export CSV.

python scripts/assoc_run_demo.py \
  --project projects/demo_2Dsuspension_25C \
  --out projects/demo_2Dsuspension_25C/trajs.csv


Visualize tracks (optional)
Overlay tracks on raw images for sanity check.

python tools/visualize_tracks.py \
  --images projects/demo_2Dsuspension_25C/images \
  --traj   projects/demo_2Dsuspension_25C/trajs.csv \
  --out    projects/demo_2Dsuspension_25C/vis


The exact script names/options may differ in your repo; start from the examples in scripts/ and tools/. We intentionally keep interfaces simple (few required args).

Configuration

Global/data presets live in configs/.

Project-specific overrides can be placed under projects/<name>/config.(toml|yaml).

Typical parameters include pixel size, FPS, association radius, IoU threshold, etc.

If a parameter table CSV is used (for simulation/sweeps), ensure integer-type keys like NperiSmpl are handled as integers.

Outputs

Masks: multi-page TIF or per-frame label TIF (0 background, 1..K instances)

Trajectories: CSV/Parquet with columns like
TIME_frame, TRACK_ID, X_(com), Y_(com), bbox_w, bbox_h, area_mask, ...

Diagnostics: optional figures/PNG overlays under projects/<name>/vis/

Known limitations

The demo is tuned for 2D swimming bacteria; dense biofilms or extreme overlaps need further tuning.

Association is currently heuristic (IoU + distance). Long gaps or fast motion may require model-based linking.

Training code and pretrained weights are minimal; bring your own model if needed.

Roadmap

 Add tiny unit tests for I/O & association logic

 Provide a single quickstart script that runs end-to-end on demo images

 Publish docs (MkDocs) with troubleshooting and FAQs

 Optional: release pretrained weights for the demo

Contributing

Pull requests are welcome!
Please keep code style simple and add short English comments/docstrings.
For features with many parameters, prefer config files over long CLI flags.

License

Code: MIT (unless stated otherwise inside subfolders)

Third-party notice: Parts of the embedding-based segmentation approach are inspired by / derived from EmbedSeg. If you include or adapt EmbedSeg code/assets governed by CC BY-NC 4.0, that material remains non-commercial and must retain attribution. See THIRD_PARTY_LICENSES.md (to be added) for details.

If you plan commercial use, review third-party licenses carefully and remove/replace non-commercial components.

Citation

If you use this software in your research, please cite:

@software{synembtrack,
  author  = {Son, Joowang and contributors},
  title   = {synembtrack: embedding-based instance segmentation and tracking for motile bacteria},
  year    = {2025},
  url     = {https://github.com/joowson/synembtrack}
}


A CITATION.cff will be provided in releases for citation managers and Zenodo DOI integration.

Acknowledgements

We acknowledge the developers of EmbedSeg and related open-source tools in the microscopy tracking ecosystem. Community feedback and issues are very welcome.

Questions?

Open an Issue or start a Discussion on GitHub.
Happy tracking! 🦠📈

# LaneSuite

A single, menu-driven interface to run eight different lane-detection models — SCNN, ERFNet, three ENet-SAD variants, YOLOP, Ultra-Fast-Lane-Detection-v2, and a classical (non-deep-learning) OpenCV pipeline — without needing to juggle their conflicting dependencies yourself.

Each model was originally built for a different, mutually incompatible environment (TensorFlow 1.x on Python 3.7, modern PyTorch, Lua Torch7, or plain OpenCV). Rather than force them into one shared environment, LaneSuite keeps each model in its own isolated environment and launches the right one automatically when you pick it from the menu.

## What's included

| # | Model | Framework | Trained on |
|---|-------|-----------|------------|
| 1 | SCNN | TensorFlow 1.15 (VGG-16 backbone) | CULane |
| 2 | ERFNet | PyTorch | CULane |
| 3 | ENet-SAD | Lua Torch7 | CULane |
| 4 | ENet-SAD | Lua Torch7 | TuSimple |
| 5 | ENet-SAD | Lua Torch7 | BDD100K |
| 6 | YOLOP | PyTorch | Lane + drivable area + object detection |
| 7 | Ultra-Fast-Lane-Detection-v2 | PyTorch | CULane — runs on CPU or GPU |
| 8 | Classical CV (OpenCV, no training data) | OpenCV / scikit-learn | Color threshold + Hough + DBSCAN — tuned for Indian roads |

Models 7 and 8 each have more than one infer script registered (see `model_registry.py`). After picking the model number in the menu, you'll get a second prompt asking which script variant to run:

- **Model 7:** `infer_folder.py` (default) or `infer_folder_letterbox.py` (pads the image to CULane's aspect ratio before inference — use this for photos with a lot of extra sky or an aspect ratio very different from CULane's dashcam-style framing; the default script hard-crops to a fixed 1600×320 window regardless of input shape, so a mismatched aspect ratio can crop lane markings right out of frame).
- **Model 8:** `infer_folder.py` (currently the only variant — more can be added the same way).

## Project structure

```
lane-suite/
├── run.py                   # entry point — shows the menu, launches the chosen model
├── model_registry.py         # maps each model number to its interpreter, script(s), and env
├── jpg_to_ppm.py              # shared preprocessing for models 3, 4, 5 (Torch7 has no jpg codec)
├── common_input/              # drop your input images here
└── models/
    ├── 1_scnn_tensorflow/
    │   ├── repo/               # model code
    │   ├── weights/             # NOT included in this repo — see "Model weights" below
    │   ├── venv/                # you create this — see Setup
    │   └── requirements.txt
    ├── 2_erfnet_culane/
    │   ├── repo/
    │   ├── venv/
    │   └── requirements.txt
    ├── 3_enet_culane/
    │   ├── repo/
    │   └── weights/
    ├── 4_enet_tusimple/
    │   ├── repo/
    │   └── weights/
    ├── 5_enet_bdd100k/
    │   ├── repo/
    │   └── weights/
    ├── 6_yolop/
    │   ├── repo/
    │   ├── weights/
    │   ├── venv/
    │   └── requirements.txt
    ├── 7_ufld_v2/
    │   ├── repo/                # cloned from cfzd/Ultra-Fast-Lane-Detection-v2
    │   │                        #   infer_folder.py            — standard inference
    │   │                        #   infer_folder_letterbox.py  — aspect-ratio-preserving variant
    │   ├── weights/               # culane_res34.pth goes here
    │   ├── dali_stub/             # workaround for an unused training-only import — see setup below
    │   ├── venv/
    │   └── requirements.txt
    └── 8_cv_lane/
        ├── repo/                # infer_folder.py — classical OpenCV pipeline, no checkpoint
        ├── output/               # results + output/debug/ (HSV mask, Canny, Hough overlay with --debug)
        ├── venv/
        └── requirements.txt
```

## Prerequisites

- **Git** — [git-scm.com](https://git-scm.com/downloads)
- **Python 3.11** (for models 2, 6, 7, and 8)
- **Python 3.7** (for model 1 — TensorFlow 1.15 does not support newer Python versions)
  - On Windows, install both versions and use the `py` launcher (`py -3.7`, `py -3.11`) to pick the right one, since a single `python` command can only point at one version at a time.
- **Torch7 + LuaJIT** (for models 3, 4, 5) — see below, this is the one non-Python dependency.

### Setting up Torch7 (models 3, 4, 5 only)

Models 3–5 run on [Torch7](https://github.com/torch/torch7), an older deep learning framework, via LuaJIT rather than Python. Install it using the official distribution:

- Install guide: [torch/distro](https://github.com/torch/distro)

**Important — CPU vs GPU:**
This project's own `infer.lua` scripts already auto-detect a GPU at runtime (via `cutorch`) and will use it automatically if one is available — no code changes needed. However, the **standard Windows Torch7 install** (the one linked above) is CPU-only by default; it does not include `cutorch`/`cudnn` support out of the box. If you follow the standard install, models 3–5 will run on CPU, which is slower but fully functional.

If you want GPU acceleration for these three models, you'll need a Torch7 build that includes CUDA support (`cutorch`, `cunn`, `cudnn`) — this typically means building Torch7 from source with CUDA enabled, which is more involved than the standard install. This is a limitation of Torch7 itself on Windows, not of this project's code.

Models 1, 2, 6, and 7 (TensorFlow/PyTorch) do **not** have this limitation — they use GPU automatically if a CUDA-enabled install of TensorFlow/PyTorch is present, and fall back to CPU otherwise, with no extra setup beyond a normal `pip install`. Model 8 is classical CV (OpenCV) and always runs on CPU — there's no GPU path for it, by design.

## Setup

### 1. Clone this repository

```bash
git clone https://github.com/YOUR_USERNAME/lane-suite.git
cd lane-suite
```

### 2. Download the model weights

Model weights for models 1–6 are hosted separately (too large for a git repository) on Hugging Face:

**[ALTF4-pro/lane-suite-weights](https://huggingface.co/ALTF4-pro/lane-suite-weights)**

Download the weights for whichever models you want to run, and place them in the matching folder:

| Model | Place weight file(s) in |
|---|---|
| 1 – SCNN | `models/1_scnn_tensorflow/weights/` |
| 2 – ERFNet | `models/2_erfnet_culane/repo/ERFNet-CULane-PyTorch/trained/` |
| 3 – ENet CULane | `models/3_enet_culane/weights/` |
| 4 – ENet TuSimple | `models/4_enet_tusimple/weights/` |
| 5 – ENet BDD100K | `models/5_enet_bdd100k/weights/` |
| 6 – YOLOP | `models/6_yolop/weights/` |

**Model 7 (UFLDv2)** uses the original authors' own pretrained checkpoint instead, linked directly from their repo:

- [CULane, ResNet34, F1 76.0 (Google Drive)](https://drive.google.com/file/d/1AjnvAD3qmqt_dGPveZJsLZ1bOyWv62Yj/view?usp=sharing)

Download it and place it at `models/7_ufld_v2/weights/culane_res34.pth`.

**Model 8 (Classical CV)** needs no weights or checkpoint at all — it's a plain OpenCV pipeline (color threshold + Canny + Hough + DBSCAN clustering), not a trained network.

### 3. Set up each model's environment

You only need to set up the environments for the models you actually plan to run.

**Model 1 (SCNN):**
```bash
cd models/1_scnn_tensorflow
py -3.7 -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
cd ../..
```

**Model 2 (ERFNet):**
```bash
cd models/2_erfnet_culane
py -3.11 -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
cd ../..
```

**Models 3, 4, 5 (ENet-SAD):** no venv needed — these run through the system-wide Torch7/LuaJIT install described above.

**Model 6 (YOLOP):**
```bash
cd models/6_yolop
py -3.11 -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
cd ../..
```

**Model 7 (Ultra-Fast-Lane-Detection-v2) — CPU or GPU:**

This model auto-detects a CUDA-capable GPU at runtime and uses it if present, otherwise falls back to CPU (slower, but fully functional — no build errors either way).

```bash
cd models/7_ufld_v2
py -3.11 -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
cd ../..
```

**Extra one-time step for model 7 only:** the original repo's code imports NVIDIA DALI (a fast-dataloading library) at the top of a file that's shared between training and inference — even though inference never actually uses it. Rather than requiring you to install the full DALI library just to satisfy an unused import, this repo ships a tiny stub in `models/7_ufld_v2/dali_stub/` that satisfies the import without doing anything. Copy it into your venv's site-packages once, after installing requirements:

```bash
cp -r models/7_ufld_v2/dali_stub/nvidia models/7_ufld_v2/venv/Lib/site-packages/nvidia
```

(macOS/Linux: the destination is `models/7_ufld_v2/venv/lib/pythonX.X/site-packages/nvidia` instead.)

**Model 8 (Classical CV) — CPU only, no checkpoint needed:**

```bash
cd models/8_cv_lane
py -3.11 -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
cd ../..
```

(macOS/Linux: replace `venv\Scripts\python.exe` with `venv/bin/python` throughout, for every model above.)

## Usage

Drop your test images (`.jpg`) into `common_input/`, then run:

```bash
python run.py
```

This shows a menu of all eight models — enter the number of the one you want to run. For models 7 and 8, you'll then get a second prompt listing that model's available infer-script variants (see "What's included" above) — enter the number of the script you want.

To skip the model menu and run a specific model directly:

```bash
python run.py --model 1
```

(If that model has multiple script variants, you'll still be prompted to choose one.)

Output is written to that model's own `output/` folder inside its `models/<n>_.../repo/` or `models/<n>_.../` directory. Model 8 also supports a `--debug` flag (run its script directly, not through `run.py`, if you want this) that additionally saves the HSV color mask, Canny edges, and a Hough-line overlay to `output/debug/` for troubleshooting.

## Known limitations

- **Models 3, 4, 5 are CPU-only by default on Windows**, as explained above — this is a Torch7/Windows limitation, not a limitation of this project's code, which already supports GPU automatically if a CUDA-enabled Torch7 build is used.
- Model 1 requires Python 3.7 specifically, due to TensorFlow 1.15's version constraints.
- Model 7 runs on CPU or GPU, but CPU inference is noticeably slower — a GPU (local or via Colab) is recommended for anything beyond a handful of test images.
- **Model 8 is a classical color-threshold pipeline, not a trained model** — it has no notion of "lane" beyond HSV color + straight-edge geometry, so it's sensitive to lighting, faded/worn paint, and other white or yellow objects (e.g. painted walls, curbs) inside its region-of-interest. It works well on clear, well-lit, unobstructed road markings; it is not a substitute for the trained models (1–7) on harder scenes.

## Credits

This project is an orchestration layer around the following original work — all credit for the underlying models, training code, and pretrained weights goes to their original authors:

- Models 1–5 (SCNN, ERFNet, ENet-SAD): [cardwing/Codes-for-Lane-Detection](https://github.com/cardwing/Codes-for-Lane-Detection) — *"Learning Lightweight Lane Detection CNNs by Self Attention Distillation" (ICCV 2019)*
- Model 6 (YOLOP): [hustvl/YOLOP](https://github.com/hustvl/YOLOP) — MIT licensed
- Model 7 (Ultra-Fast-Lane-Detection-v2): [cfzd/Ultra-Fast-Lane-Detection-v2](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2) — MIT licensed
- Model 8 (Classical CV pipeline): pipeline design (gamma correction → bilateral filter → HSV color threshold → Canny → ROI mask → Hough transform → DBSCAN clustering) adapted from [Giscle/Moto-Dream-Lane-Detection-](https://github.com/Giscle/Moto-Dream-Lane-Detection-.git) (team "Moto Dream"); re-tuned and ported to a still-image folder pipeline for this project.

## License

LaneSuite's own code (the orchestration layer — `run.py`, `model_registry.py`, and the portability fixes on top of the original repos) is released under the MIT License — see [LICENSE](LICENSE).

The underlying model code and weights for models 1–5 remain governed by the terms of their original repository (no formal license file there; please cite the associated papers if you use them, as requested by the original authors — see [LICENSE](LICENSE) for full citation details). Model 6 (YOLOP) is separately MIT licensed by its original authors.

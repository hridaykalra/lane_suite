# LaneSuite

A single, menu-driven interface to run nine different lane-detection models — SCNN, ERFNet, three ENet-SAD variants, YOLOP, Ultra-Fast-Lane-Detection-v2, a classical (non-deep-learning) OpenCV pipeline, and a Torch7 implementation of SCNN — without needing to juggle their conflicting dependencies yourself.

Each model was originally built for a different, mutually incompatible environment (TensorFlow 1.x on Python 3.7, modern PyTorch, Lua Torch7, or plain OpenCV). Rather than force them into one shared environment, LaneSuite keeps each model in its own isolated environment where necessary and launches the right one automatically when you pick it from the menu.

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
| 9 | SCNN | Lua Torch7 | CULane |

Models 7 and 8 each have more than one infer script registered (see `model_registry.py`). After picking the model number in the menu, you'll get a second prompt asking which script variant to run:

- **Model 7:** `infer_folder.py` (default) or `infer_folder_letterbox.py` (pads the image to CULane's aspect ratio before inference).
- **Model 8:** `infer_folder.py` (currently the only variant — more can be added the same way).
- **Model 9:** `infer_folder.lua` — folder-based SCNN inference using the included CPU checkpoint.

## Project structure

```text
lane-suite/
├── run.py
├── model_registry.py
├── jpg_to_ppm.py
├── common_input/
└── models/
    ├── 1_scnn_tensorflow/
    │   ├── repo/
    │   ├── weights/
    │   ├── venv/
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
    │   ├── repo/
    │   ├── weights/
    │   ├── dali_stub/
    │   ├── venv/
    │   └── requirements.txt
    ├── 8_cv_lane/
    │   ├── repo/
    │   ├── output/
    │   ├── venv/
    │   └── requirements.txt
    └── 9_scnn_culane/
        ├── repo/
        │   ├── infer_folder.lua
        │   ├── postprocess_ppm_to_jpg.py
        │   └── output/
        └── weights/
            ├── vgg_SCNN_DULR_w9.t7
            └── vgg_SCNN_DULR_w9_cpu.t7
```

## Prerequisites

- **Git**
- **Python 3.11** (for models 2, 6, 7, and 8)
- **Python 3.7** (for model 1 — TensorFlow 1.15 does not support newer Python versions)
  - On Windows, install both versions and use the `py` launcher (`py -3.7`, `py -3.11`) to pick the right one.
- **Torch7 + LuaJIT** (for models 3, 4, 5, and 9)

### Setting up Torch7 (models 3, 4, 5, 9)

Models 3–5 and Model 9 run on **Torch7**, an older deep learning framework, via LuaJIT rather than Python.

Install Torch7 using the official distribution:

https://github.com/torch/distro

The Torch7 inference scripts can use a CUDA-enabled installation when available. The standard Windows setup is CPU-oriented.

Model 9 includes a CPU-compatible SCNN checkpoint:

```text
models/9_scnn_culane/weights/vgg_SCNN_DULR_w9_cpu.t7
```

Therefore **Model 9 does not require CUDA or a GPU for the standard folder-inference workflow**.

## Setup

### 1. Clone this repository

```bash
git clone https://github.com/YOUR_USERNAME/lane-suite.git
cd lane-suite
```

### 2. Download the model weights

Model weights for models 1–6 are hosted separately (too large for a git repository) on Hugging Face:

**ALTF4-pro/lane-suite-weights**

Download the weights for whichever models you want to run and place them in the matching folder:

| Model | Place weight file(s) in |
|---|---|
| 1 – SCNN | `models/1_scnn_tensorflow/weights/` |
| 2 – ERFNet | `models/2_erfnet_culane/repo/ERFNet-CULane-PyTorch/trained/` |
| 3 – ENet CULane | `models/3_enet_culane/weights/` |
| 4 – ENet TuSimple | `models/4_enet_tusimple/weights/` |
| 5 – ENet BDD100K | `models/5_enet_bdd100k/weights/` |
| 6 – YOLOP | `models/6_yolop/weights/` |

**Model 7 (UFLDv2)** uses the original authors' pretrained checkpoint:

- CULane, ResNet34, F1 76.0 — Google Drive

Place it at:

```text
models/7_ufld_v2/weights/culane_res34.pth
```

**Model 8 (Classical CV)** needs no weights or checkpoint.

**Model 9 (SCNN)** uses the original SCNN pretrained model.

**[Download SCNN CULane pretrained weights](https://drive.google.com/open?id=1Wv3r3dCYNBwJdKl_WPEfrEOt-XGaROKu)**

For the LaneSuite CPU inference workflow, the CPU-compatible checkpoint is:

```text
models/9_scnn_culane/weights/vgg_SCNN_DULR_w9_cpu.t7
```

The non-CPU checkpoint is:

```text
models/9_scnn_culane/weights/vgg_SCNN_DULR_w9.t7
```

No separate CULane dataset download is required for LaneSuite's folder-inference workflow.

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

**Models 3, 4, 5, and 9 (Torch7):**

No Python venv is needed — these use the system-wide Torch7/LuaJIT installation.

**Model 6 (YOLOP):**

```bash
cd models/6_yolop
py -3.11 -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
cd ../..
```

**Model 7 (Ultra-Fast-Lane-Detection-v2):**

```bash
cd models/7_ufld_v2
py -3.11 -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
cd ../..
```

**Model 8 (Classical CV):**

```bash
cd models/8_cv_lane
py -3.11 -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
cd ../..
```

(macOS/Linux: replace `venv\Scripts\python.exe` with `venv/bin/python` throughout.)

## Usage

Drop your test images (`.jpg`) into `common_input/`, then run:

```bash
python run.py
```

This shows a menu of all nine models. Enter the number of the model you want to run.

To skip the model menu and run a specific model directly:

```bash
python run.py --model 1
```

If that model has multiple script variants, you'll still be prompted to choose one.

Model 9 uses the same shared `common_input/` workflow as the other folder-based models. Its Torch7 inference script runs SCNN inference using the included CPU checkpoint and writes its results to the Model 9 output directory.

## Model 9 — SCNN folder inference

Model 9 is based on the original **Spatial CNN (SCNN)** implementation for traffic lane detection.

SCNN was introduced in *Spatial As Deep: Spatial CNN for Traffic Scene Understanding* (AAAI 2018).

It uses the VGG16-based SCNN model trained on CULane and the included CPU-compatible checkpoint:

```text
models/9_scnn_culane/weights/vgg_SCNN_DULR_w9_cpu.t7
```

The folder inference script is:

```text
models/9_scnn_culane/repo/infer_folder.lua
```

The workflow is:

```text
common_input/
      │
      ▼
SCNN / Torch7 inference
      │
      ├── lane coordinate .txt files
      │
      └── overlay images
              │
              ▼
        JPG output
```

The script is designed for folder-based inference rather than requiring the original CULane test-set directory structure.

For the standard Windows setup, Model 9 runs using the CPU checkpoint and does not require CUDA.

## Known limitations

- **Models 3, 4, 5, and 9 use Torch7/LuaJIT**, an older framework that can be more difficult to install and maintain than the Python-based environments.
- Model 1 requires Python 3.7 because of TensorFlow 1.15.
- Model 7 can run on CPU or GPU, but CPU inference is slower.
- **Model 8 is a classical color-threshold pipeline, not a trained model**, so it is sensitive to lighting, faded/worn paint, and other white or yellow objects inside its region of interest.
- **Model 9 uses an older Torch7/LuaJIT stack** and is primarily intended as a CPU-compatible folder-inference implementation in LaneSuite.
- The original SCNN repository's full CULane evaluation pipeline requires additional tools such as MATLAB and OpenCV; these are not required for LaneSuite's simplified Model 9 folder-inference workflow.

## Credits

LaneSuite's own code (the orchestration layer — `run.py`, `model_registry.py`, and the portability fixes on top of the original repos) is released under the MIT License — see [LICENSE](LICENSE).

The underlying model code and weights remain governed by the terms of their original repositories. All credit for the underlying models, training code, and pretrained weights goes to their original authors.

- Models 1–5: [cardwing/Codes-for-Lane-Detection](https://github.com/cardwing/Codes-for-Lane-Detection)
- Model 6 (YOLOP): [hustvl/YOLOP](https://github.com/hustvl/YOLOP) — MIT licensed
- Model 7 (Ultra-Fast-Lane-Detection-v2): [cfzd/Ultra-Fast-Lane-Detection-v2](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2) — MIT licensed
- Model 8 (Classical CV pipeline): adapted from [Giscle/Moto-Dream-Lane-Detection-](https://github.com/Giscle/Moto-Dream-Lane-Detection-.git), then re-tuned and ported to a still-image folder pipeline for this project.
- Model 9 (SCNN Torch7): [XingangPan/SCNN](https://github.com/XingangPan/SCNN) — *Spatial As Deep: Spatial CNN for Traffic Scene Understanding* (AAAI 2018)

## License

LaneSuite's own code (the orchestration layer — `run.py`, `model_registry.py`, and the portability fixes on top of the original repos) is released under the MIT License — see [LICENSE](LICENSE).

The underlying model code and weights for models 1–5 and 9 remain governed by the terms of their original repositories. Model 6 (YOLOP) and Model 7 (Ultra-Fast-Lane-Detection-v2) are separately MIT licensed by their original authors.

Please cite the associated papers and original repositories when using the underlying models.

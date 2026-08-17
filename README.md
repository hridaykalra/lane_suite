# LaneSuite

A single, menu-driven interface to run nine different lane-detection models — SCNN, ERFNet, three ENet-SAD variants, YOLOP, Ultra-Fast-Lane-Detection-v2, a classical (non-deep-learning) OpenCV pipeline, and a Torch7 implementation of SCNN — without needing to juggle their conflicting dependencies yourself.

Each model was originally built for a different, mutually incompatible environment (TensorFlow 1.x on Python 3.7, modern PyTorch, Lua Torch7, or plain OpenCV). Rather than force them into one shared environment, LaneSuite keeps each model in its own isolated environment where necessary and launches the right one automatically when you pick it from the menu.

**Want to run just one model?** Clone this repo once, then jump straight to that model's section under [Models](#models) — each one lists everything it needs on its own, so you don't have to read about the other eight.

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

Models 7 and 8 each have more than one infer script registered (see `model_registry.py`); after picking a model number in the menu you may get a second prompt asking which script variant to run.

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

## 0. Clone the repository

Every model below assumes you've already done this once:

```bash
git clone https://github.com/YOUR_USERNAME/lane-suite.git
cd lane-suite
```

All paths in the sections below (e.g. `models/1_scnn_tensorflow/`) are relative to this `lane-suite/` folder.

## General usage (all models)

Drop your test images (`.jpg`) into `common_input/`, then run:

```bash
python run.py
```

This shows a menu of all nine models — enter the number of the model you want to run. To skip the menu and go straight to one model:

```bash
python run.py --model 1
```

If that model has multiple script variants, you'll still be prompted to choose one.

> macOS/Linux users: replace `venv\Scripts\python.exe` with `venv/bin/python` and `py -3.x` with `python3.x` throughout the model sections below.

---

## Models

Each section is self-contained: prerequisites, weight download, environment setup, and how to run — so you can set up only the model(s) you want.

### Model 1 — SCNN (TensorFlow)

- **Framework:** TensorFlow 1.15, VGG-16 backbone
- **Trained on:** CULane
- **Source:** [cardwing/Codes-for-Lane-Detection](https://github.com/cardwing/Codes-for-Lane-Detection)

**Prerequisites**
- Python 3.7 (required — TensorFlow 1.15 does not support newer Python)
  - Windows: install alongside any other Python versions and select it with `py -3.7`

**1. Download weights**

Hosted on Hugging Face: **[ALTF4-pro/lane-suite-weights](https://huggingface.co/ALTF4-pro/lane-suite-weights)**

Download the Model 1 weight file(s) and place them in:

```text
models/1_scnn_tensorflow/weights/
```

**2. Set up the environment**

```bash
cd models/1_scnn_tensorflow
py -3.7 -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
cd ../..
```

**3. Run**

Drop `.jpg` images into `common_input/`, then from the repo root:

```bash
python run.py --model 1
```

**Notes**
- This is the only model requiring Python 3.7 — keep its venv isolated from the others.

---

### Model 2 — ERFNet

- **Framework:** PyTorch
- **Trained on:** CULane
- **Source:** [cardwing/Codes-for-Lane-Detection](https://github.com/cardwing/Codes-for-Lane-Detection)

**Prerequisites**
- Python 3.11

**1. Download weights**

Hosted on Hugging Face: **[ALTF4-pro/lane-suite-weights](https://huggingface.co/ALTF4-pro/lane-suite-weights)**

Download the Model 2 weight file(s) and place them in:

```text
models/2_erfnet_culane/repo/ERFNet-CULane-PyTorch/trained/
```

**2. Set up the environment**

```bash
cd models/2_erfnet_culane
py -3.11 -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
cd ../..
```

**3. Run**

Drop `.jpg` images into `common_input/`, then from the repo root:

```bash
python run.py --model 2
```

---

### Model 3 — ENet-SAD (CULane)

- **Framework:** Lua Torch7
- **Trained on:** CULane
- **Source:** [cardwing/Codes-for-Lane-Detection](https://github.com/cardwing/Codes-for-Lane-Detection)

**Prerequisites**
- Torch7 + LuaJIT (system-wide install) — see [Setting up Torch7](#setting-up-torch7-models-3-4-5-9) below
- No Python venv needed

**1. Download weights**

Hosted on Hugging Face: **[ALTF4-pro/lane-suite-weights](https://huggingface.co/ALTF4-pro/lane-suite-weights)**

Download the Model 3 weight file(s) and place them in:

```text
models/3_enet_culane/weights/
```

**2. Run**

Drop `.jpg` images into `common_input/`, then from the repo root:

```bash
python run.py --model 3
```

---

### Model 4 — ENet-SAD (TuSimple)

- **Framework:** Lua Torch7
- **Trained on:** TuSimple
- **Source:** [cardwing/Codes-for-Lane-Detection](https://github.com/cardwing/Codes-for-Lane-Detection)

**Prerequisites**
- Torch7 + LuaJIT (system-wide install) — see [Setting up Torch7](#setting-up-torch7-models-3-4-5-9) below
- No Python venv needed

**1. Download weights**

Hosted on Hugging Face: **[ALTF4-pro/lane-suite-weights](https://huggingface.co/ALTF4-pro/lane-suite-weights)**

Download the Model 4 weight file(s) and place them in:

```text
models/4_enet_tusimple/weights/
```

**2. Run**

Drop `.jpg` images into `common_input/`, then from the repo root:

```bash
python run.py --model 4
```

---

### Model 5 — ENet-SAD (BDD100K)

- **Framework:** Lua Torch7
- **Trained on:** BDD100K
- **Source:** [cardwing/Codes-for-Lane-Detection](https://github.com/cardwing/Codes-for-Lane-Detection)

**Prerequisites**
- Torch7 + LuaJIT (system-wide install) — see [Setting up Torch7](#setting-up-torch7-models-3-4-5-9) below
- No Python venv needed

**1. Download weights**

Hosted on Hugging Face: **[ALTF4-pro/lane-suite-weights](https://huggingface.co/ALTF4-pro/lane-suite-weights)**

Download the Model 5 weight file(s) and place them in:

```text
models/5_enet_bdd100k/weights/
```

**2. Run**

Drop `.jpg` images into `common_input/`, then from the repo root:

```bash
python run.py --model 5
```

---

### Model 6 — YOLOP

- **Framework:** PyTorch
- **Trained on:** Lane detection + drivable area + object detection (multi-task)
- **Source:** [hustvl/YOLOP](https://github.com/hustvl/YOLOP) — MIT licensed

**Prerequisites**
- Python 3.11

**1. Download weights**

Hosted on Hugging Face: **[ALTF4-pro/lane-suite-weights](https://huggingface.co/ALTF4-pro/lane-suite-weights)**

Download the Model 6 weight file(s) and place them in:

```text
models/6_yolop/weights/
```

**2. Set up the environment**

```bash
cd models/6_yolop
py -3.11 -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
cd ../..
```

**3. Run**

Drop `.jpg` images into `common_input/`, then from the repo root:

```bash
python run.py --model 6
```

---

### Model 7 — Ultra-Fast-Lane-Detection-v2 (UFLDv2)

- **Framework:** PyTorch (CPU or GPU)
- **Trained on:** CULane, ResNet34 backbone, F1 76.0
- **Source:** [cfzd/Ultra-Fast-Lane-Detection-v2](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2) — MIT licensed

**Prerequisites**
- Python 3.11

**1. Download weights**

This model uses the **original authors' pretrained checkpoint** (not the LaneSuite Hugging Face repo):

**[CULane ResNet34 checkpoint (Google Drive)](https://drive.google.com/open?id=1Wv3r3dCYNBwJdKl_WPEfrEOt-XGaROKu)**

Place it at:

```text
models/7_ufld_v2/weights/culane_res34.pth
```

**2. Set up the environment**

```bash
cd models/7_ufld_v2
py -3.11 -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
cd ../..
```

**3. Run**

Drop `.jpg` images into `common_input/`, then from the repo root:

```bash
python run.py --model 7
```

You'll be prompted to choose a script variant:
- `infer_folder.py` (default)
- `infer_folder_letterbox.py` — pads the image to CULane's aspect ratio before inference

**Notes**
- Can run on CPU or GPU; CPU inference is slower.

---

### Model 8 — Classical CV (OpenCV, no deep learning)

- **Framework:** OpenCV / scikit-learn
- **Approach:** Color threshold + Hough transform + DBSCAN clustering — tuned for Indian roads
- **No training data, no weights required**
- **Source:** adapted from [Giscle/Moto-Dream-Lane-Detection-](https://github.com/Giscle/Moto-Dream-Lane-Detection-.git), re-tuned and ported to a still-image folder pipeline for this project

**Prerequisites**
- Python 3.11

**1. Weights**

None needed — this is a classical pipeline, not a trained model.

**2. Set up the environment**

```bash
cd models/8_cv_lane
py -3.11 -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
cd ../..
```

**3. Run**

Drop `.jpg` images into `common_input/`, then from the repo root:

```bash
python run.py --model 8
```

You'll be prompted to choose a script variant:
- `infer_folder.py` (currently the only variant — more can be added the same way)

**Notes**
- Being a color-threshold pipeline rather than a trained model, it's sensitive to lighting, faded/worn paint, and other white or yellow objects inside its region of interest.

---

### Model 9 — SCNN (Torch7)

- **Framework:** Lua Torch7
- **Trained on:** CULane
- **Source:** [XingangPan/SCNN](https://github.com/XingangPan/SCNN) — *Spatial As Deep: Spatial CNN for Traffic Scene Understanding* (AAAI 2018)

Model 9 is based on the original Spatial CNN (SCNN) implementation, using the VGG16-based SCNN model trained on CULane.

**Prerequisites**
- Torch7 + LuaJIT (system-wide install) — see [Setting up Torch7](#setting-up-torch7-models-3-4-5-9) below
- No Python venv needed
- **No GPU/CUDA required** — Model 9 ships with a CPU-compatible checkpoint and is designed for the standard CPU folder-inference workflow

**1. Download weights**

Uses the original SCNN pretrained model:

**[Download SCNN CULane pretrained weights (Google Drive)](https://drive.google.com/open?id=1Wv3r3dCYNBwJdKl_WPEfrEOt-XGaROKu)**

Two checkpoints are used:

```text
models/9_scnn_culane/weights/vgg_SCNN_DULR_w9_cpu.t7   # used by LaneSuite's CPU inference workflow
models/9_scnn_culane/weights/vgg_SCNN_DULR_w9.t7       # non-CPU (GPU) checkpoint
```

**2. Run**

Drop `.jpg` images into `common_input/`, then from the repo root:

```bash
python run.py --model 9
```

You'll be prompted to choose a script variant:
- `infer_folder.lua` — folder-based SCNN inference using the included CPU checkpoint

This runs the inference script directly:

```text
models/9_scnn_culane/repo/infer_folder.lua
```

**Workflow**

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

**Notes**
- The script is designed for folder-based inference rather than requiring the original CULane test-set directory structure.
- The original SCNN repository's full CULane evaluation pipeline requires additional tools such as MATLAB and OpenCV; these are **not** required for LaneSuite's simplified Model 9 folder-inference workflow.
- No separate CULane dataset download is required.

---

## Setting up Torch7 (models 3, 4, 5, 9)

Models 3, 4, 5, and 9 run on **Torch7**, an older deep learning framework, via LuaJIT rather than Python. No Python venv is needed for these — they use a system-wide Torch7/LuaJIT installation.

Install Torch7 using the official distribution:

**https://github.com/torch/distro**

The Torch7 inference scripts can use a CUDA-enabled installation when available, but the standard Windows setup is CPU-oriented. Model 9 specifically includes a CPU-compatible checkpoint (`vgg_SCNN_DULR_w9_cpu.t7`), so **Model 9 does not require CUDA or a GPU** for the standard folder-inference workflow.

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

Please cite the associated papers and original repositories when using the underlying models.

## License

LaneSuite's own code (the orchestration layer — `run.py`, `model_registry.py`, and the portability fixes on top of the original repos) is released under the MIT License — see [LICENSE](LICENSE).

The underlying model code and weights for models 1–5 and 9 remain governed by the terms of their original repositories. Model 6 (YOLOP) and Model 7 (Ultra-Fast-Lane-Detection-v2) are separately MIT licensed by their original authors.

"""
model_registry.py

Maps a menu number -> everything needed to run that model as an
isolated subprocess: which Python interpreter to use (its own venv,
NOT the one running this script), which adapter script to call, and
where its output folder is.

Why subprocess instead of importing a plugin class (like our earlier
TF/PyTorch demo): these repos require mutually incompatible dependency
versions (TensorFlow 1.3 on Python 3.5 vs. modern PyTorch). They
cannot share one interpreter. Each model folder gets its own venv;
this file just knows how to call out to it.

Most models have ONE infer script: use the "script" key (a single Path).

Models 7 and 8 support MULTIPLE infer scripts (e.g. different tuning
variants) via the "scripts" key instead: a dict of
    {number: {"label": <shown in menu>, "path": <Path to .py file>}}
run.py detects "scripts" vs "script" and shows a second numbered prompt
only when "scripts" is present -- so this only affects 7 and 8, and only
after you've already picked that model number. To add a new variant
script, drop the new .py file in that model's repo/ folder and add an
entry here with the next number.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
COMMON_INPUT = BASE_DIR / "common_input"

# venv layout differs by OS: Windows uses venv/Scripts/python.exe,
# Mac/Linux uses venv/bin/python. Detect once, use everywhere below.
if sys.platform.startswith("win"):
    VENV_PYTHON = Path("venv") / "Scripts" / "python.exe"
else:
    VENV_PYTHON = Path("venv") / "bin" / "python"


def _model_dir(name: str) -> Path:
    return BASE_DIR / "models" / name


# "script" = the model's own infer script, called with NO arguments.
# That script must have its own input/output paths hardcoded internally
# (pointing at COMMON_INPUT and its own models/<n>/output folder) --
# see each model folder's infer script for exactly where those live.
MODELS = {
    "1": {
    "label": "SCNN (TensorFlow, VGG-16, CULane)",
    "folder": _model_dir("1_scnn_tensorflow"),
    "python": _model_dir("1_scnn_tensorflow") / VENV_PYTHON,
    "script": _model_dir("1_scnn_tensorflow") / "repo" / "scnn_infer_fixed.py",
    },
     "2": {
            "label": "ERFNet (PyTorch, CULane)",
            "folder": _model_dir("2_erfnet_culane"),
            "python": _model_dir("2_erfnet_culane") / VENV_PYTHON,
            "script": _model_dir("2_erfnet_culane") / "repo" / "ERFNet-CULane-PyTorch" / "infer_folder.py",
        },
    "3": {
        "label": "ENet-SAD (Lua Torch7, CULane)",
        "folder": _model_dir("3_enet_culane"),
        "python": Path(r"C:\torch\bin\luajit.exe"),
        "script": _model_dir("3_enet_culane") / "repo" / "infer.lua",
        "env": {
            "LUA_CPATH": "C:/torch/bin/?.dll;;",
            "LUA_DEV": "C:/torch",
            "LUA_PATH": "C:/torch/lua/?;C:/torch/lua/?.lua;C:/torch/lua/?/init.lua;;",
            "PATH_PREPEND": r"C:\torch\bin",
        },
    },
    "4": {
        "label": "ENet-SAD (Lua Torch7, TuSimple)",
        "folder": _model_dir("4_enet_tusimple"),
        "python": Path(r"C:\torch\bin\luajit.exe"),
        "script": _model_dir("4_enet_tusimple") / "repo" / "infer.lua",
        "env": {
            "LUA_CPATH": "C:/torch/bin/?.dll;;",
            "LUA_DEV": "C:/torch",
            "LUA_PATH": "C:/torch/lua/?;C:/torch/lua/?.lua;C:/torch/lua/?/init.lua;;",
            "PATH_PREPEND": r"C:\torch\bin",
        },
    },

    "5": {
        "label": "ENet-SAD (Lua Torch7, BDD100K)",
        "folder": _model_dir("5_enet_bdd100k"),
        "python": Path(r"C:\torch\bin\luajit.exe"),
        "script": _model_dir("5_enet_bdd100k") / "repo" / "infer.lua",
        "env": {
            "LUA_CPATH": "C:/torch/bin/?.dll;;",
            "LUA_DEV": "C:/torch",
            "LUA_PATH": "C:/torch/lua/?;C:/torch/lua/?.lua;C:/torch/lua/?/init.lua;;",
            "PATH_PREPEND": r"C:\torch\bin",
        },
    },

    "6": {
        "label": "YOLOP (PyTorch, lane + drivable area + object detection)",
        "folder": _model_dir("6_yolop"),
        "python": _model_dir("6_yolop") / VENV_PYTHON,
        "script": _model_dir("6_yolop") / "repo" / "infer_yolop.py",
    },

    # --- Models 7 and 8: multi-script support via "scripts" instead of "script".
    # Add more numbered entries here as you add variant .py files; run.py will
    # automatically show them in the sub-menu, no changes needed there.

    "7": {
        "label": "Ultra-Fast-Lane-Detection-v2 (PyTorch, CULane) — GPU required",
        "folder": _model_dir("7_ufld_v2"),
        "python": _model_dir("7_ufld_v2") / VENV_PYTHON,
        "scripts": {
            "1": {
                "label": "lane markings",
                "path": _model_dir("7_ufld_v2") / "repo" / "infer_folder.py",
            },

            "2": {
                "label": "lane markings after padding",
                "path": _model_dir("7_ufld_v2") / "repo" / "infer_folder_letterbox.py",
            },
        },
    },

    "8": {
        "label": "Classical CV Lane Detection (OpenCV, Indian roads) — CPU only",
        "folder": _model_dir("8_cv_lane"),
        "python": _model_dir("8_cv_lane") / "venv" / "Scripts" / "python.exe",
        "scripts": {
            "1": {
                "label": "vector lane marking",
                "path": _model_dir("8_cv_lane") / "repo" / "infer_folder.py",
            },

            "2": {
                "label": "black and white lane markings",
                "path": _model_dir("8_cv_lane") / "repo" / "infer_folder1.py",
            },
        },
    },
}

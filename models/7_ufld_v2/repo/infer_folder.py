"""
infer_folder.py  (model 7: Ultra-Fast-Lane-Detection-v2, CULane, PyTorch)

Standalone inference script — runs the CULane ResNet34 checkpoint on a
folder of your own images, instead of the original repo's demo.py
(which only works on the CULane test dataset's own list files and
outputs a video).

GPU required: this model is intended for GPU-equipped machines only
(per project decision) — it will raise a clear error on a CPU-only
machine rather than silently running very slowly or crashing deep
inside a .cuda() call.

Preprocessing / anchor values below are copied exactly from
configs/culane_res34.py and utils/common.py, so they match how this
checkpoint was trained. Do not change these unless you're using a
different checkpoint/config.

Run with:  python infer_folder.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import cv2
from PIL import Image
import torchvision.transforms as transforms

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR  # this script lives inside repo/, alongside model/utils/etc.
PROJECT_ROOT = SCRIPT_DIR / ".." / ".." / ".."

INPUT_DIR = PROJECT_ROOT / "common_input"
OUTPUT_DIR = SCRIPT_DIR / "output"
WEIGHTS_PATH = SCRIPT_DIR / ".." / "weights" / "culane_res34.pth"

# So `from model...` / `from utils...` imports (this repo's own modules)
# resolve correctly regardless of where infer_folder.py is invoked from.
sys.path.insert(0, str(REPO_DIR))

from model.model_culane import get_model

# ==========================================================
# Config values copied from configs/culane_res34.py
# (must match the checkpoint's training config exactly)
# ==========================================================
class Cfg:
    dataset = 'CULane'
    backbone = '34'
    griding_num = 200
    use_aux = False
    num_lanes = 4
    num_row = 72
    num_col = 81
    train_width = 1600
    train_height = 320
    num_cell_row = 200
    num_cell_col = 100
    fc_norm = True
    crop_ratio = 0.6

cfg = Cfg()
cfg.row_anchor = np.linspace(0.42, 1, cfg.num_row)
cfg.col_anchor = np.linspace(0, 1, cfg.num_col)

IMG_W, IMG_H = 1640, 590   # CULane's native image size — coordinates are scaled to this
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def pred2coords(pred, row_anchor, col_anchor, local_width=1,
                 original_image_width=1640, original_image_height=590):
    """Unchanged from the original repo's demo.py — converts raw
    grid-classification output into actual (x, y) lane point coordinates."""
    batch_size, num_grid_row, num_cls_row, num_lane_row = pred['loc_row'].shape
    batch_size, num_grid_col, num_cls_col, num_lane_col = pred['loc_col'].shape

    max_indices_row = pred['loc_row'].argmax(1).cpu()
    valid_row = pred['exist_row'].argmax(1).cpu()

    max_indices_col = pred['loc_col'].argmax(1).cpu()
    valid_col = pred['exist_col'].argmax(1).cpu()

    pred['loc_row'] = pred['loc_row'].cpu()
    pred['loc_col'] = pred['loc_col'].cpu()

    coords = []
    row_lane_idx = [1, 2]
    col_lane_idx = [0, 3]

    for i in row_lane_idx:
        tmp = []
        if valid_row[0, :, i].sum() > num_cls_row / 2:
            for k in range(valid_row.shape[1]):
                if valid_row[0, k, i]:
                    all_ind = torch.tensor(list(range(
                        max(0, max_indices_row[0, k, i] - local_width),
                        min(num_grid_row - 1, max_indices_row[0, k, i] + local_width) + 1)))
                    out_tmp = (pred['loc_row'][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_row - 1) * original_image_width
                    tmp.append((int(out_tmp), int(row_anchor[k] * original_image_height)))
            coords.append(tmp)

    for i in col_lane_idx:
        tmp = []
        if valid_col[0, :, i].sum() > num_cls_col / 4:
            for k in range(valid_col.shape[1]):
                if valid_col[0, k, i]:
                    all_ind = torch.tensor(list(range(
                        max(0, max_indices_col[0, k, i] - local_width),
                        min(num_grid_col - 1, max_indices_col[0, k, i] + local_width) + 1)))
                    out_tmp = (pred['loc_col'][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_col - 1) * original_image_height
                    tmp.append((int(col_anchor[k] * original_image_width), int(out_tmp)))
            coords.append(tmp)

    return coords


def banner():
    print("=" * 60)
    print("        Ultra-Fast-Lane-Detection-v2 -- CULane (PyTorch)")
    print("=" * 60)


def load_model():
    if DEVICE.type == 'cpu':
        print("WARNING: No CUDA-capable GPU detected. Running on CPU instead -- "
            "this will be significantly slower than GPU inference.\n")

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {WEIGHTS_PATH}\n"
            "Download culane_res34.pth from the link in the README and place "
            "it in this model's weights/ folder."
        )

    print("Loading checkpoint (this can take a little while)...")
    net = get_model(cfg)

    state_dict = torch.load(str(WEIGHTS_PATH), map_location='cpu', weights_only=False)['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.to(DEVICE)
    net.eval()

    print("Model loaded successfully.\n")
    return net


def main():
    banner()
    torch.backends.cudnn.benchmark = True

    net = load_model()

    img_transforms = transforms.Compose([
        transforms.Resize((int(cfg.train_height / cfg.crop_ratio), cfg.train_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for p in INPUT_DIR.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    print(f"Input Folder : {INPUT_DIR}")
    print(f"Output Folder: {OUTPUT_DIR}")
    print(f"Images Found : {len(images)}\n")

    if not images:
        print("No supported images were found in that folder.")
        return

    processed, failed = 0, 0

    for img_path in images:
        print(f"Processing : {img_path.name}")
        try:
            pil_img = Image.open(img_path).convert("RGB")
            input_tensor = img_transforms(pil_img)
            input_tensor = input_tensor[:, -cfg.train_height:, :]
            input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                pred = net(input_tensor)

            vis = cv2.imread(str(img_path))
            orig_h, orig_w = vis.shape[:2]   # use the image's OWN native size, not CULane's fixed size

            coords = pred2coords(
                pred, cfg.row_anchor, cfg.col_anchor,
                original_image_width=orig_w, original_image_height=orig_h
            )
            for lane in coords:
                for coord in lane:
                    cv2.circle(vis, coord, 10, (0, 255, 0), -1)

            out_path = OUTPUT_DIR / f"{img_path.stem}_overlay.jpg"
            cv2.imwrite(str(out_path), vis)
            print(f"  Saved -> {out_path}\n")
            processed += 1
        except Exception as e:
            print(f"  ERROR while processing {img_path.name}: {e}\n")
            failed += 1

    print("=" * 60)
    print(f"Done. Processed: {processed}  Failed: {failed}  Total: {len(images)}")
    print("=" * 60)


if __name__ == "__main__":
    main()

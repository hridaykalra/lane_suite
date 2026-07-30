"""
infer_folder_letterbox.py  (model 7: Ultra-Fast-Lane-Detection-v2, CULane, PyTorch)

Variant of infer_folder.py using LETTERBOX PADDING for the output
visualization, instead of either (a) force-stretching to CULane's
fixed 1640x590 size, or (b) leaving the image at native resolution.

Why: CULane's model expects/predicts coordinates in a 2.78:1 (1640x590)
aspect ratio space. If your source images have a very different aspect
ratio, keeping them at native resolution (as in the plain fix) can look
slightly off since the model's internal "sense" of proportions doesn't
match your photo's proportions. Letterboxing pads the image (with black
bars) up to CULane's aspect ratio FIRST, so the coordinate space and the
image proportions match exactly -- no stretching, no distortion, just
padding.

Run with:  python infer_folder_letterbox.py
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

IMG_W, IMG_H = 1640, 590   # CULane's native image size / aspect ratio target
TARGET_ASPECT = IMG_W / IMG_H  # ~2.7797

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def letterbox_to_aspect(img, target_aspect):
    """
    Pad `img` (a cv2/numpy BGR image) with black bars so its aspect
    ratio matches `target_aspect`, WITHOUT stretching or cropping any
    original content. Returns (padded_img, pad_left, pad_top, scale)
    where scale is always 1.0 here since we only pad, never resize --
    kept for clarity/symmetry with the coordinate-mapping step below.

    Padding is added either left/right (if the image is relatively too
    tall/narrow) or top/bottom (if relatively too wide/short) so the
    final canvas has exactly `target_aspect`.
    """
    h, w = img.shape[:2]
    current_aspect = w / h

    if current_aspect < target_aspect:
        # Image is too narrow for the target aspect -> pad left/right
        new_w = int(round(h * target_aspect))
        pad_total = new_w - w
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        padded = cv2.copyMakeBorder(img, 0, 0, pad_left, pad_right,
                                     cv2.BORDER_CONSTANT, value=(0, 0, 0))
        pad_top = 0
    else:
        # Image is too wide/short for the target aspect -> pad top/bottom
        new_h = int(round(w / target_aspect))
        pad_total = new_h - h
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, 0, 0,
                                     cv2.BORDER_CONSTANT, value=(0, 0, 0))
        pad_left = 0

    return padded, pad_left, pad_top


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
    print("   Ultra-Fast-Lane-Detection-v2 -- CULane (PyTorch)")
    print("   [letterbox variant]")
    print("=" * 60)


def load_model():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA-capable GPU detected. Model 7 (UFLDv2) requires a GPU "
            "machine to run -- this is a project requirement, not a code bug. "
            "See the README's hardware notes for details."
        )

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
    net.cuda()
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
            # ---- Model input: unchanged, still the direct stretch resize
            # the checkpoint was trained on. Do not touch this part. ----
            pil_img = Image.open(img_path).convert("RGB")
            input_tensor = img_transforms(pil_img)
            input_tensor = input_tensor.unsqueeze(0).cuda()

            with torch.no_grad():
                pred = net(input_tensor)

            # ---- Visualization: letterbox pad to CULane's aspect ratio
            # instead of stretching (old bug) or leaving native-res
            # (plain fix) — keeps proportions correct with no distortion.
            raw = cv2.imread(str(img_path))
            vis, pad_left, pad_top = letterbox_to_aspect(raw, TARGET_ASPECT)
            padded_h, padded_w = vis.shape[:2]

            coords = pred2coords(
                pred, cfg.row_anchor, cfg.col_anchor,
                original_image_width=padded_w, original_image_height=padded_h
            )
            for lane in coords:
                for coord in lane:
                    cv2.circle(vis, coord, 5, (0, 255, 0), -1)

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

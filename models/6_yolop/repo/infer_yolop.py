"""
infer.py  (model 6: YOLOP)

Fixed-path version -- no CLI args, no re-downloading weights every run.
Edit the three paths below to match your setup, then run.py will call
this with zero arguments.
"""

from pathlib import Path

import numpy as np
import torch
import cv2

# --- relative paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / ".." / ".." / ".." / "common_input"
OUTPUT_DIR = SCRIPT_DIR / "output"
LANE_DIR = OUTPUT_DIR / "lane"
DRIVABLE_DIR = OUTPUT_DIR / "drivable"
WEIGHTS_PATH = SCRIPT_DIR / ".." / "weights" / "End-to-end.pth"
# ------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 640

# thresholds (tune these if masks look too sparse or too noisy)
LANE_PROB_THRESH = 0.45
DRIVABLE_PROB_THRESH = 0.5


def load_model():
    # pretrained=False -> skips YOLOP's own auto-download of weights.
    # We load our local .pth file into the architecture ourselves instead.
    model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=False, trust_repo=True)
    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    state_dict = checkpoint.get("state_dict", checkpoint)  # some checkpoints wrap it, some don't
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    return model


def letterbox(img, new_shape=640, color=(114, 114, 114)):
    """Resize + pad image to a square while preserving aspect ratio (YOLO-style)."""
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
    dw /= 2
    dh /= 2

    if (w, h) != new_unpad:
        img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    else:
        img_resized = img

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                     cv2.BORDER_CONSTANT, value=color)
    return img_padded, r, (dw, dh)


def preprocess(image_path: Path, img_size: int = IMG_SIZE):
    img0 = cv2.imread(str(image_path))
    img_padded, ratio, (dw, dh) = letterbox(img0, img_size)

    img = img_padded[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(img).unsqueeze(0).to(DEVICE)

    return tensor, img0, ratio, (dw, dh)


def unpad_mask(mask, ratio, pad, orig_shape):
    """Remove letterbox padding from a mask and resize back to original image size."""
    dw, dh = pad
    h_pad, w_pad = mask.shape[:2]

    top = int(round(dh - 0.1))
    left = int(round(dw - 0.1))
    bottom = h_pad - int(round(dh + 0.1))
    right = w_pad - int(round(dw + 0.1))

    cropped = mask[top:bottom, left:right]
    restored = cv2.resize(cropped, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)
    return restored


def run_inference(model, image_path: Path):
    tensor, original, ratio, pad = preprocess(image_path)
    with torch.no_grad():
        det_out, da_seg_out, ll_seg_out = model(tensor)

    # soft probability instead of hard argmax -> keeps thin/low-confidence lane pixels
    ll_prob = torch.softmax(ll_seg_out, dim=1)[0, 1].cpu().numpy()
    da_prob = torch.softmax(da_seg_out, dim=1)[0, 1].cpu().numpy()

    lane_mask = (ll_prob > LANE_PROB_THRESH).astype("uint8") * 255
    drivable_mask = (da_prob > DRIVABLE_PROB_THRESH).astype("uint8") * 255

    orig_shape = original.shape[:2]
    lane_mask = unpad_mask(lane_mask, ratio, pad, orig_shape)
    drivable_mask = unpad_mask(drivable_mask, ratio, pad, orig_shape)

    return lane_mask, drivable_mask, original


def make_overlay(original_img, lane_mask, drivable_mask, alpha: float = 0.5):
    """Paint drivable area (green) and lane lines (red) onto a copy of the original image."""
    overlay = original_img.copy()

    da_bool = drivable_mask > 0
    overlay[da_bool] = (
        overlay[da_bool] * (1 - alpha) + np.array([0, 255, 0]) * alpha
    ).astype("uint8")

    ll_bool = lane_mask > 0
    overlay[ll_bool] = (
        overlay[ll_bool] * (1 - alpha) + np.array([0, 0, 255]) * alpha
    ).astype("uint8")

    return overlay


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LANE_DIR.mkdir(parents=True, exist_ok=True)
    DRIVABLE_DIR.mkdir(parents=True, exist_ok=True)

    images = [p for p in sorted(INPUT_DIR.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    if not images:
        print(f"No images found in {INPUT_DIR}")
        return

    print("Loading YOLOP model...")
    model = load_model()

    for img_path in images:
        lane_mask, drivable_mask, original = run_inference(model, img_path)

        overlay_out = make_overlay(original, lane_mask, drivable_mask)
        cv2.imwrite(str(LANE_DIR / f"{img_path.stem}_lane.png"), lane_mask)
        cv2.imwrite(str(DRIVABLE_DIR / f"{img_path.stem}_drivable.png"), drivable_mask)
        cv2.imwrite(str(OUTPUT_DIR / f"{img_path.stem}_overlay.png"), overlay_out)

        print(f"  {img_path.name} -> {img_path.stem}_lane.png, {img_path.stem}_drivable.png, {img_path.stem}_overlay.png")

    print(f"Done. {len(images)} image(s) processed -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
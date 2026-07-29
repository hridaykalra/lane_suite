import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress TF's own C++ level logs

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # suppress TF's Python-level warnings

import os
import sys
import cv2
import time
import traceback
import numpy as np
from PIL import Image, ImageOps

from pathlib import Path

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# ==========================================================
# Configuration
# ==========================================================

# Path to the checkpoint PREFIX (no .data-00000-of-00001 / .index / .meta suffix)
CHECKPOINT = r"C:\Users\hrida\Documents\Internship\lane_suite\lane_suite\models\1_scnn_tensorflow\weights\culane_lanenet_vgg_2018-12-01-14-38-37.ckpt-10000"

# --- fixed input/output paths (no more folder picker) ---
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / ".." / ".." / ".." / "common_input"
OUTPUT_DIR = SCRIPT_DIR / "output"

# Subfolders inside OUTPUT_DIR, split out so the root output folder only
# holds the actual deliverable (the final *_lanes.jpg) instead of being a
# flat mix of debug crops, heatmaps, and raw probability arrays.
DEBUG_DIR = OUTPUT_DIR / "debug"
HEATMAP_DIR = OUTPUT_DIR / "heatmaps"
# ----------------------------------------------------------

# SCNN-Tensorflow / CULane standard input size (from global_config.py: CFG.TRAIN.IMG_HEIGHT/WIDTH)
MODEL_WIDTH = 800
MODEL_HEIGHT = 288

# RGB mean values from the actual lanenet_data_processor_test.py (images are
# decoded via tf.image.decode_jpeg, which is RGB - NOT BGR/opencv order)
VGG_MEAN = [123.68, 116.779, 103.939]

USE_GPU = False  # set True if you have a GPU-enabled tensorflow install

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
)

CONFIDENCE_THRESHOLD = 0.3   # tweak if lanes look too thin/thick
EXIST_THRESHOLD = 0.5        # tweak if lanes are missing/appearing spuriously

LANE_LINE_COLOR = (0, 0, 0)      # black, BGR
LANE_LINE_THICKNESS = 3

DRIVABLE_FILL_COLOR = (0, 200, 0)  # green, BGR
DRIVABLE_FILL_ALPHA = 0.35         # 0 = invisible, 1 = fully opaque

# Multi-anchor crop fallback: a single fixed ANCHOR_CENTER_FRAC assumes
# every photo is shot from the same height/angle, which real handheld
# phone photos don't guarantee -- debug crops have shown some photos
# landing on trees/sky instead of road at the default anchor. Instead of
# committing to one guess, try each of these vertical crop-center
# fractions in turn and keep the FIRST one that yields at least one
# usable lane curve. Ordered with the previously-working default (0.40)
# first, since it should still be the best guess for most images -- this
# keeps behavior unchanged for images that already worked, and only
# falls through to the alternates for images that didn't.
ANCHOR_CANDIDATES = (0.40, 0.30, 0.50, 0.25, 0.55)


def extract_lane_curve(mask, min_points=8, outlier_px=40, smooth_window=9):
    """
    Given a boolean mask (H, W), trace a single smooth curve through it.

    Robust against noisy/stray blobs:
      1. Take the per-row MEDIAN x (not mean) - resistant to a single stray pixel cluster.
      2. Reject rows whose x deviates too far from a local rolling median (outlier removal).
      3. Fit a smooth polynomial x = f(y) through the remaining inlier points and
         resample it row-by-row, so the final line has no jagged zigzags.

    Returns an (N, 2) int32 array of (x, y) points sorted top-to-bottom, or
    None if there isn't enough reliable signal to draw a line.
    """
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None

    row_to_xs = {}
    for y, x in zip(ys, xs):
        row_to_xs.setdefault(int(y), []).append(int(x))

    raw_ys = np.array(sorted(row_to_xs.keys()))
    raw_xs = np.array([int(np.median(row_to_xs[y])) for y in raw_ys], dtype=float)

    if len(raw_ys) < min_points:
        return None

    # Rolling median to establish a "local trend" for outlier rejection
    half = smooth_window // 2
    local_trend = np.copy(raw_xs)
    for i in range(len(raw_xs)):
        lo = max(0, i - half)
        hi = min(len(raw_xs), i + half + 1)
        local_trend[i] = np.median(raw_xs[lo:hi])

    residual = np.abs(raw_xs - local_trend)
    inliers = residual < outlier_px

    if inliers.sum() < min_points:
        inliers = np.ones_like(inliers, dtype=bool)  # fallback: keep everything

    ys_in = raw_ys[inliers]
    xs_in = raw_xs[inliers]

    if len(ys_in) < 2:
        return None

    # Fit a smooth curve (quadratic handles gentle road curvature; falls back to
    # linear if too few points for a stable quadratic fit)
    degree = 2 if len(ys_in) >= 6 else 1
    coeffs = np.polyfit(ys_in, xs_in, degree)
    poly = np.poly1d(coeffs)

    y_smooth = np.arange(ys_in.min(), ys_in.max() + 1)
    x_smooth = poly(y_smooth)

    curve = np.stack([x_smooth, y_smooth], axis=1).astype(np.int32)
    return curve


# ==========================================================
# Helper Functions
# ==========================================================

def banner():
    print("=" * 60, flush=True)
    print("        SCNN-Tensorflow Lane Detection", flush=True)
    print("=" * 60, flush=True)
    print("TensorFlow :", tf.__version__, flush=True)
    print()


# ==========================================================
# Model
# ==========================================================

def load_model():
    """
    Builds the LaneNet graph directly (bypassing the repo's file-path-based
    tf.map_fn input pipeline) so we can feed already-preprocessed numpy arrays,
    and restores the pretrained CULane checkpoint into it.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from lanenet_model import lanenet_merge_model
    from config import global_config

    CFG = global_config.cfg

    print("Creating LaneNet (SCNN) model...", flush=True)

    input_tensor = tf.placeholder(
        dtype=tf.float32,
        shape=[None, MODEL_HEIGHT, MODEL_WIDTH, 3],
        name='input_tensor'
    )
    phase_tensor = tf.constant('test', tf.string)

    net = lanenet_merge_model.LaneNet()
    binary_seg_ret, existence_output = net.test_inference(input_tensor, phase_tensor, 'lanenet_loss')

    initial_var = tf.global_variables()
    final_var = initial_var[:-1]
    saver = tf.train.Saver(final_var)

    if not os.path.exists(CHECKPOINT + ".index") and not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(
            f"Checkpoint not found at:\n{os.path.abspath(CHECKPOINT)}\n"
            f"(CHECKPOINT should be the path PREFIX, without .index/.meta/.data-... suffix)"
        )

    if USE_GPU:
        sess_config = tf.ConfigProto(device_count={'GPU': 1})
    else:
        sess_config = tf.ConfigProto(device_count={'GPU': 0})
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    from tensorflow.core.protobuf import rewriter_config_pb2
    sess_config.graph_options.rewrite_options.remapping = rewriter_config_pb2.RewriterConfig.OFF
    sess_config.graph_options.rewrite_options.disable_meta_optimizer = True

    print("Loading checkpoint (this can take a little while on CPU)...", flush=True)
    sess = tf.Session(config=sess_config)
    sess.run(tf.global_variables_initializer())
    saver.restore(sess=sess, save_path=CHECKPOINT)

    print("Model loaded successfully.\n", flush=True)
    return sess, input_tensor, binary_seg_ret, existence_output


# ==========================================================
# Image Preprocessing
# ==========================================================

def preprocess(image, anchor_center_frac):
    """
    Same crop/resize/normalize logic as before, but anchor_center_frac is
    now a parameter instead of a fixed module-level constant, so main()
    can try several candidate anchors per image (see ANCHOR_CANDIDATES).
    """
    original = image.copy()
    original_h, original_w = original.shape[:2]

    target_ratio = MODEL_WIDTH / MODEL_HEIGHT  # 800/288 ~= 2.778
    current_ratio = original_w / original_h

    if current_ratio > target_ratio:
        crop_w = int(round(original_h * target_ratio))
        crop_x_offset = (original_w - crop_w) // 2
        crop_y_offset = 0
        cropped = image[:, crop_x_offset: crop_x_offset + crop_w]
        crop_h = original_h
    else:
        crop_h = int(round(original_w / target_ratio))
        crop_x_offset = 0
        anchor_px = int(round(original_h * anchor_center_frac))
        crop_y_offset = anchor_px - crop_h // 2
        crop_y_offset = max(0, min(crop_y_offset, original_h - crop_h))  # clamp in-bounds
        cropped = image[crop_y_offset: crop_y_offset + crop_h, :]
        crop_w = original_w

    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(cropped_rgb, (MODEL_WIDTH, MODEL_HEIGHT), interpolation=cv2.INTER_CUBIC)

    resized = resized.astype(np.float32)
    resized[:, :, 0] -= VGG_MEAN[0]   # Red
    resized[:, :, 1] -= VGG_MEAN[1]   # Green
    resized[:, :, 2] -= VGG_MEAN[2]   # Blue

    batch = np.expand_dims(resized, axis=0)  # NHWC, shape (1, H, W, 3)

    pad_info = {
        "crop_x_offset": crop_x_offset, "crop_y_offset": crop_y_offset,
        "crop_w": crop_w, "crop_h": crop_h,
    }

    return batch, original, original_h, original_w, pad_info


def load_image_any_format(image_path):
    try:
        pil_img = Image.open(image_path)
        pil_img = ImageOps.exif_transpose(pil_img)
        pil_img = pil_img.convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        return cv2.imread(str(image_path))


def run_inference_at_anchor(sess, input_tensor, binary_seg_ret, existence_output,
                             image, anchor_center_frac):
    """
    Runs one full forward pass at a given anchor and returns everything
    main() needs: the per-lane curves found, the raw prediction/exist
    arrays, and pad_info (needed to build the debug crop image). Factored
    out of main() so the multi-anchor retry loop can call this once per
    candidate anchor without duplicating the pre/post-processing logic.
    """
    input_batch, original, original_h, original_w, pad_info = preprocess(image, anchor_center_frac)

    binary_out, exist_out = sess.run(
        [binary_seg_ret, existence_output],
        feed_dict={input_tensor: input_batch}
    )

    prediction = binary_out[0]   # (H, W, 5)
    exist = exist_out[0]         # (4,)

    lane_curves = {}
    lane_prob_maps = {}  # lane_idx -> full-original-size float32 prob map, for saving/heatmap

    for lane_idx in range(4):
        if exist[lane_idx] < EXIST_THRESHOLD:
            continue

        class_prob = prediction[:, :, lane_idx + 1]

        class_prob_crop = cv2.resize(
            class_prob, (pad_info["crop_w"], pad_info["crop_h"]),
            interpolation=cv2.INTER_LINEAR
        )
        class_prob_resized = np.zeros((original_h, original_w), dtype=class_prob_crop.dtype)
        class_prob_resized[
            pad_info["crop_y_offset"]: pad_info["crop_y_offset"] + pad_info["crop_h"],
            pad_info["crop_x_offset"]: pad_info["crop_x_offset"] + pad_info["crop_w"]
        ] = class_prob_crop
        lane_mask = class_prob_resized > CONFIDENCE_THRESHOLD

        lane_prob_maps[lane_idx] = class_prob_resized

        curve = extract_lane_curve(lane_mask)
        if curve is not None:
            lane_curves[lane_idx] = curve

    return {
        "lane_curves": lane_curves,
        "lane_prob_maps": lane_prob_maps,
        "exist": exist,
        "input_batch": input_batch,
        "original": original,
        "original_h": original_h,
        "original_w": original_w,
        "pad_info": pad_info,
    }


# ==========================================================
# Main
# ==========================================================

def main():
    banner()

    sess, input_tensor, binary_seg_ret, existence_output = load_model()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted([
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ])

    print()
    print(f"Input Folder : {INPUT_DIR}")
    print(f"Output Folder: {OUTPUT_DIR}")
    print(f"  Debug crops -> {DEBUG_DIR}")
    print(f"  Heatmaps/prob maps -> {HEATMAP_DIR}")
    print(f"Images Found : {len(images)}")
    print()

    if len(images) == 0:
        print("No supported images were found in that folder.")
        sess.close()
        return

    processed = 0
    failed = 0

    for image_name in images:
        start = time.time()
        image_path = os.path.join(INPUT_DIR, image_name)

        print(f"Processing : {image_name}", flush=True)

        image = load_image_any_format(image_path)
        if image is None:
            print("  Could not read image (corrupt file or unsupported format). Skipping.\n")
            failed += 1
            continue

        print(f"  Original size: {image.shape[1]}x{image.shape[0]} (w x h)", flush=True)

        try:
            # --- multi-anchor retry loop ---
            # Try each candidate anchor in order; keep the first one that
            # finds at least one usable lane curve. If none find a curve,
            # fall back to keeping the FIRST anchor's result (ANCHOR_CANDIDATES[0])
            # so downstream code (existence scores, debug images) still has
            # something consistent to report, same as before this change.
            result = None
            used_anchor = None
            for anchor in ANCHOR_CANDIDATES:
                candidate_result = run_inference_at_anchor(
                    sess, input_tensor, binary_seg_ret, existence_output, image, anchor
                )
                if candidate_result["lane_curves"]:
                    result = candidate_result
                    used_anchor = anchor
                    break
                if result is None:
                    # keep the first attempt as the fallback in case NONE
                    # of the candidates find a curve
                    result = candidate_result
                    used_anchor = anchor

            lane_curves = result["lane_curves"]
            exist = result["exist"]
            input_batch = result["input_batch"]
            original = result["original"]
            original_h = result["original_h"]
            original_w = result["original_w"]
            pad_info = result["pad_info"]
            lane_prob_maps = result["lane_prob_maps"]

            print(f"  Anchor used: {used_anchor} (tried {len(ANCHOR_CANDIDATES)} candidates)", flush=True)
            print(f"  Lane existence scores: {[round(float(e), 3) for e in exist]}", flush=True)

            # DEBUG: save exactly what the network saw at the WINNING anchor
            # (post-crop, pre-normalize), in its own subfolder.
            debug_crop_bgr = cv2.cvtColor(
                ((input_batch[0] + VGG_MEAN).clip(0, 255)).astype(np.uint8), cv2.COLOR_RGB2BGR
            )

            # Save per-lane heatmaps + raw probability arrays into heatmaps/
            for lane_idx, class_prob_resized in lane_prob_maps.items():
                heatmap = (np.clip(class_prob_resized, 0, 1) * 255).astype(np.uint8)
                cv2.imwrite(
                    str(HEATMAP_DIR / f"DEBUG_{Path(image_name).stem}_heatmap_lane{lane_idx}.jpg"),
                    heatmap
                )
                np.save(
                    str(HEATMAP_DIR / f"PROB_{Path(image_name).stem}_lane{lane_idx}.npy"),
                    class_prob_resized.astype(np.float32)
                )

            print(f"  Lanes with usable curve: {list(lane_curves.keys())} (out of 0-3)", flush=True)

            # ---------------------------------------
            # Figure out the "drivable lane"
            # ---------------------------------------
            img_center_x = original_w / 2.0
            left_candidates = []
            right_candidates = []

            for idx, curve in lane_curves.items():
                bottom_x = int(curve[-1][0])
                if bottom_x < img_center_x:
                    left_candidates.append((idx, bottom_x))
                else:
                    right_candidates.append((idx, bottom_x))

            left_idx = max(left_candidates, key=lambda t: t[1])[0] if left_candidates else None
            right_idx = min(right_candidates, key=lambda t: t[1])[0] if right_candidates else None

            blended = original.copy()

            if left_idx is not None and right_idx is not None:
                left_map = {int(y): int(x) for x, y in lane_curves[left_idx]}
                right_map = {int(y): int(x) for x, y in lane_curves[right_idx]}
                common_ys = sorted(set(left_map.keys()) & set(right_map.keys()))

                if len(common_ys) >= 2:
                    poly_points = [(left_map[y], y) for y in common_ys]
                    poly_points += [(right_map[y], y) for y in reversed(common_ys)]
                    poly_points = np.array(poly_points, dtype=np.int32)

                    highlight_layer = original.copy()
                    cv2.fillPoly(highlight_layer, [poly_points], DRIVABLE_FILL_COLOR)
                    blended = cv2.addWeighted(
                        original, 1 - DRIVABLE_FILL_ALPHA,
                        highlight_layer, DRIVABLE_FILL_ALPHA, 0
                    )

            for idx, curve in lane_curves.items():
                cv2.polylines(
                    blended, [curve.reshape((-1, 1, 2))],
                    isClosed=False, color=LANE_LINE_COLOR,
                    thickness=LANE_LINE_THICKNESS, lineType=cv2.LINE_AA
                )

            # NEW: also draw the detected lane curves onto the debug crop
            # image itself, so you can see in ONE picture both (a) exactly
            # what region the model was shown and (b) whether it found a
            # lane in that region -- makes it immediately visible when a
            # crop landed on trees/sky vs. actual road.
            debug_with_lanes = debug_crop_bgr.copy()
            for idx, curve in lane_curves.items():
                # curve coordinates are in ORIGINAL image space; convert
                # to the debug crop's coordinate space (crop offset, then
                # scaled from crop size down to MODEL_WIDTH x MODEL_HEIGHT)
                cx, cy = pad_info["crop_x_offset"], pad_info["crop_y_offset"]
                cw, ch = pad_info["crop_w"], pad_info["crop_h"]
                sx = MODEL_WIDTH / cw
                sy = MODEL_HEIGHT / ch
                pts = []
                for x, y in curve:
                    px = (x - cx) * sx
                    py = (y - cy) * sy
                    if 0 <= px < MODEL_WIDTH and 0 <= py < MODEL_HEIGHT:
                        pts.append([int(px), int(py)])
                if len(pts) >= 2:
                    pts = np.array(pts, dtype=np.int32)
                    cv2.polylines(
                        debug_with_lanes, [pts.reshape((-1, 1, 2))],
                        isClosed=False, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA
                    )
            cv2.putText(
                debug_with_lanes, f"anchor={used_anchor}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA
            )
            cv2.imwrite(
                str(DEBUG_DIR / f"DEBUG_{Path(image_name).stem}_networkinput.jpg"),
                debug_with_lanes
            )

            # Save using the same extension as the input file
            input_ext = Path(image_name).suffix.lower()
            if input_ext in (".jpg", ".jpeg"):
                out_ext = ".jpg"
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
            else:
                out_ext = input_ext if input_ext else ".png"
                encode_params = []

            save_path = os.path.join(OUTPUT_DIR, f"{Path(image_name).stem}_lanes{out_ext}")
            cv2.imwrite(save_path, blended, encode_params)

            elapsed = time.time() - start
            print(f"  Saved -> {save_path}  ({elapsed:.2f}s)\n", flush=True)
            processed += 1

        except Exception:
            print(f"  ERROR while processing {image_name}:")
            traceback.print_exc()
            print()
            failed += 1

    print("=" * 60)
    print(f"Done. Processed: {processed}  Failed: {failed}  Total: {len(images)}")
    print("=" * 60)

    sess.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("FATAL ERROR:")
        traceback.print_exc()

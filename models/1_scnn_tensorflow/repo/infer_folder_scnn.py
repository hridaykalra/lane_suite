import os
import sys
import cv2
import time
import traceback
import numpy as np
import tkinter as tk

from tkinter import filedialog
from pathlib import Path

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# ==========================================================
# Configuration
# ==========================================================

# Path to the checkpoint PREFIX (no .data-00000-of-00001 / .index / .meta suffix),
# e.g. "trained/culane_lanenet_vgg_2018-12-01-14-38-37.ckpt-10000"
CHECKPOINT = r"C:\Users\hrida\Documents\Internship\Codes-for-Lane-Detection\SCNN-Tensorflow\lane-detection-model\model\culane_lanenet_vgg_2018-12-01-14-38-37.ckpt-10000"

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


def choose_folder(title):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.update()  # force the window manager to process the topmost request

    folder = filedialog.askdirectory(title=title, parent=root)

    root.destroy()
    return folder


# ==========================================================
# Model
# ==========================================================

def load_model():
    """
    Builds the LaneNet graph directly (bypassing the repo's file-path-based
    tf.map_fn input pipeline) so we can feed already-preprocessed numpy arrays,
    and restores the pretrained CULane checkpoint into it.
    """
    # Make sure the SCNN-Tensorflow model package is importable. This script is
    # expected to live in SCNN-Tensorflow/lane-detection-model/ alongside the
    # lanenet_model/, config/, encoder_decoder_model/ folders, same as the
    # repo's own tools/test_lanenet.py.
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

    # Matches the repo's own test_lanenet.py: drop the last global variable
    # (the non-trainable smoothing kernel) before restoring, since it isn't
    # part of the saved checkpoint.
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

    # Avoids spurious MKL/Grappler remapper errors on some CPU builds
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

def preprocess(image):
    original = image.copy()
    original_h, original_w = original.shape[:2]

    target_ratio = MODEL_WIDTH / MODEL_HEIGHT  # 800/288 ~= 2.778
    current_ratio = original_w / original_h

    if current_ratio > target_ratio:
        # image is relatively too wide -> pad height (top/bottom)
        new_h = int(round(original_w / target_ratio))
        pad = new_h - original_h
        top = pad // 2
        bottom = pad - top
        left = 0
        right = 0
    else:
        # image is relatively too tall (or square) -> pad width (left/right)
        new_w = int(round(original_h * target_ratio))
        pad = new_w - original_w
        left = pad // 2
        right = pad - left
        top = 0
        bottom = 0

    padded = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    padded_h, padded_w = padded.shape[:2]

    # cv2 reads BGR; the model was trained on RGB (tf.image.decode_jpeg), so convert
    padded_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

    # BICUBIC to match tf.image.resize_images(..., method=BICUBIC) in the original pipeline
    resized = cv2.resize(padded_rgb, (MODEL_WIDTH, MODEL_HEIGHT), interpolation=cv2.INTER_CUBIC)

    resized = resized.astype(np.float32)
    # RGB mean subtraction, matching lanenet_data_processor_test.py exactly
    resized[:, :, 0] -= VGG_MEAN[0]   # Red
    resized[:, :, 1] -= VGG_MEAN[1]   # Green
    resized[:, :, 2] -= VGG_MEAN[2]   # Blue

    batch = np.expand_dims(resized, axis=0)  # NHWC, shape (1, H, W, 3)

    pad_info = {
        "top": top, "bottom": bottom,
        "left": left, "right": right,
        "padded_h": padded_h, "padded_w": padded_w,
    }

    return batch, original, original_h, original_w, pad_info


# ==========================================================
# Main
# ==========================================================

def main():
    banner()

    sess, input_tensor, binary_seg_ret, existence_output = load_model()

    print("Select the INPUT folder (a picker window should appear -")
    print("check your taskbar/Alt+Tab if you don't see it immediately)...", flush=True)
    input_folder = choose_folder("Choose Input Folder")

    if not input_folder:
        print("\nNo input folder selected. Exiting.")
        sess.close()
        return

    print("Select the OUTPUT folder...", flush=True)
    output_folder = choose_folder("Choose Output Folder")

    if not output_folder:
        print("\nNo output folder selected. Exiting.")
        sess.close()
        return

    os.makedirs(output_folder, exist_ok=True)

    images = sorted([
        f for f in os.listdir(input_folder)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ])

    print()
    print(f"Input Folder : {input_folder}")
    print(f"Output Folder: {output_folder}")
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
        image_path = os.path.join(input_folder, image_name)

        print(f"Processing : {image_name}", flush=True)

        image = cv2.imread(image_path)
        if image is None:
            print("  Could not read image (corrupt file or unsupported format). Skipping.\n")
            failed += 1
            continue

        print(f"  Original size: {image.shape[1]}x{image.shape[0]} (w x h)", flush=True)

        try:
            input_batch, original, original_h, original_w, pad_info = preprocess(image)

            binary_out, exist_out = sess.run(
                [binary_seg_ret, existence_output],
                feed_dict={input_tensor: input_batch}
            )

            prediction = binary_out[0]   # (H, W, 5)
            exist = exist_out[0]         # (4,)

            print(f"  Lane existence scores: {[round(float(e), 3) for e in exist]}", flush=True)

            # Extract a thin curve (list of points) for every lane that's actually present
            lane_curves = {}
            for lane_idx in range(4):
                if exist[lane_idx] < EXIST_THRESHOLD:
                    continue

                class_prob = prediction[:, :, lane_idx + 1]

                # Undo the padding: resize prediction back up to the padded
                # canvas size, then crop out just the region that corresponds
                # to the original image (same fix as the ERFNet version).
                class_prob_padded = cv2.resize(
                    class_prob, (pad_info["padded_w"], pad_info["padded_h"]),
                    interpolation=cv2.INTER_LINEAR
                )
                class_prob_resized = class_prob_padded[
                    pad_info["top"]: pad_info["top"] + original_h,
                    pad_info["left"]: pad_info["left"] + original_w
                ]
                lane_mask = class_prob_resized > CONFIDENCE_THRESHOLD

                curve = extract_lane_curve(lane_mask)
                if curve is not None:
                    lane_curves[lane_idx] = curve

            print(f"  Lanes with usable curve: {list(lane_curves.keys())} (out of 0-3)", flush=True)

            # ---------------------------------------
            # Figure out the "drivable lane": the two lane lines immediately
            # left and right of image-center, using each curve's bottom-most point
            # ---------------------------------------
            img_center_x = original_w / 2.0
            left_candidates = []   # (lane_idx, bottom_x) for lanes left of center
            right_candidates = []  # (lane_idx, bottom_x) for lanes right of center

            for idx, curve in lane_curves.items():
                bottom_x = int(curve[-1][0])  # last point = bottom-most row (sorted ascending y)
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

            # Draw every detected lane as a slender black line on top, full opacity
            for idx, curve in lane_curves.items():
                cv2.polylines(
                    blended, [curve.reshape((-1, 1, 2))],
                    isClosed=False, color=LANE_LINE_COLOR,
                    thickness=LANE_LINE_THICKNESS, lineType=cv2.LINE_AA
                )

            # Save using the same extension as the input file (e.g. .jpg stays .jpg)
            input_ext = Path(image_name).suffix.lower()
            if input_ext in (".jpg", ".jpeg"):
                out_ext = ".jpg"
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
            else:
                out_ext = input_ext if input_ext else ".png"
                encode_params = []

            save_path = os.path.join(output_folder, f"{Path(image_name).stem}_lanes{out_ext}")
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
    input("\nPress Enter to exit...")

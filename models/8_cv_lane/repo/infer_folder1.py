"""
Model 8 - Classical CV lane detection (OpenCV pipeline, ported from Moto Dream's
Final notebook.ipynb). Runs on a folder of still images.

Pipeline: bilateral filter -> HSV color threshold (yellow+white) -> Canny edges
-> crop to ROI trapezoid -> Hough transform -> DBSCAN clustering -> draw final lines.

Fixes vs. original notebook (tuned on real daylight phone photos, not the
training video dayLane.webm):
  - Dropped gamma correction. It was tuned for a dark night video; on bright
    daylight photos it blows out the frame so road asphalt becomes numerically
    identical to white lane paint in HSV, and everything gets masked in.
  - White HSV threshold tightened: min [0,0,210], max [255,45,255]
    (was [0,0,195] / [255,80,255]) to actually separate paint from road.
  - BUG FIX: the original notebook computed a ROI-cropped Canny image but then
    ran Hough on the *uncropped* Canny anyway. Fixed so Hough only sees edges
    inside the road ROI (this was why it picked up a flowerbed edge).
  - DBSCAN eps tightened 0.5 -> 0.08 (0.5 was merging both lane lines into a
    single cluster on these image sizes).
  - Main output changed to black background + white HSV-matched pixels,
    cropped to the road ROI (matches img/color.jpg style in the source repo).
    This keeps the real curve/dash shape of lane paint instead of forcing it
    through Hough + DBSCAN into fake straight lines (which is also what was
    drawing a false straight line through a white building wall). The
    Hough/DBSCAN straight-line fit is kept only as an optional debug overlay,
    not the main output.

Usage:
    python infer_folder.py            # normal run
    python infer_folder.py --debug    # also dumps hsv-mask/canny/overlay per image
"""
import os
import sys
import math
import argparse
from collections import defaultdict

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

# --- paths (matches suite convention: repo/ -> models/8_cv_lane/ -> models/ -> project root) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..", "..", "..")
COMMON_INPUT = os.path.join(PROJECT_ROOT, "common_input")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "output")
DEBUG_DIR = os.path.join(OUTPUT_DIR, "debug")

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# --- tuned color thresholds ---
MIN_VAL_Y = np.array([15, 80, 190])
MAX_VAL_Y = np.array([30, 255, 255])
MIN_VAL_W = np.array([0, 0, 210])
MAX_VAL_W = np.array([255, 45, 255])

HOUGH_THRESHOLD = 14
DISCARD_HORIZONTAL = 0.4
DBSCAN_EPS = 0.08
DBSCAN_MIN_SAMPLES = 3


def hsv_filter(image, min_y, max_y, min_w, max_w):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_yellow = cv2.inRange(hsv, min_y, max_y)
    mask_white = cv2.inRange(hsv, min_w, max_w)
    mask = cv2.bitwise_or(mask_yellow, mask_white)
    img_filtered = cv2.bitwise_and(image, image, mask=mask)
    return img_filtered, mask


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    shape = img.shape
    mask_color = (255,) * shape[-1] if len(shape) == 3 else 255
    cv2.fillPoly(mask, vertices, mask_color)
    return cv2.bitwise_and(img, mask)


def get_roi_points(h, w):
    top_left = (w * (3.0 / 8), h * (2.7 / 5))
    top_right = (w * (5.0 / 8), h * (2.7 / 5))
    pts = [
        (0, h),
        (0, h * (3.4 / 5)),
        top_left,
        top_right,
        (w, h * (3.4 / 5)),
        (w, h),
    ]
    return np.array([pts], np.int32)


def hough_transform(gray_img, threshold, discard_horizontal=DISCARD_HORIZONTAL):
    lines = cv2.HoughLines(gray_img, 0.5, np.pi / 360, threshold)
    lines_ok = []
    if lines is not None:
        for i in range(len(lines)):
            rho, theta = lines[i][0]
            m = -math.cos(theta) / (math.sin(theta) + 1e-10)
            if abs(m) < discard_horizontal:
                continue
            lines_ok.append([rho, theta])
    return np.array(lines_ok)


def get_cluster_lines(lines, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES):
    """Returns list of (rho, theta) — one per detected lane line, after DBSCAN clustering."""
    if lines.shape[0] == 0:
        return []

    scaler = MinMaxScaler()
    scaler.fit(lines)
    lines_scaled = scaler.fit_transform(lines)

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(lines_scaled)
    labels = db.labels_
    lines_orig = scaler.inverse_transform(lines_scaled)

    grouped = defaultdict(list)
    for i, label in enumerate(labels):
        if label == -1:
            continue  # noise point, skip
        grouped[label].append([lines_orig[i, 0], lines_orig[i, 1]])

    result = []
    num_clusters = max(grouped.keys(), default=-1) + 1
    for c in range(num_clusters):
        if c not in grouped:
            continue
        rho, theta = np.mean(np.array(grouped[c]), axis=0)
        result.append((rho, theta))
    return result


def draw_lines(canvas, cluster_lines, color, thickness=6):
    for rho, theta in cluster_lines:
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
        cv2.line(canvas, pt1, pt2, color, thickness, cv2.LINE_AA)
    return canvas


def process_image(img, debug_name=None, debug=False):
    h, w = img.shape[:2]

    bilateral = cv2.bilateralFilter(img, 9, 80, 80)
    filtered, mask = hsv_filter(bilateral, MIN_VAL_Y, MAX_VAL_Y, MIN_VAL_W, MAX_VAL_W)
    canny = cv2.Canny(filtered, 100, 255)

    roi_pts = get_roi_points(h, w)
    cropped_canny = region_of_interest(canny, roi_pts)

    # Main output: black background, white where the HSV filter matched lane paint,
    # cropped to the road ROI. Keeps the real curve/dash shape of the paint instead
    # of forcing it through Hough + DBSCAN into fake straight lines (matches
    # img/color.jpg style from the source repo, not img/hough_lines.jpg).
    mask_roi = region_of_interest(mask, roi_pts)
    lanes_only = cv2.cvtColor(mask_roi, cv2.COLOR_GRAY2BGR)

    if debug and debug_name:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{debug_name}_hsv_mask.jpg"), mask)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{debug_name}_canny_roi.jpg"), cropped_canny)
        # Hough+DBSCAN straight-line fit kept only as an optional debug overlay,
        # not the main output, since it invents straight lines on curves/dashes.
        lines = hough_transform(cropped_canny, HOUGH_THRESHOLD)
        cluster_lines = get_cluster_lines(lines)
        overlay = draw_lines(img.copy(), cluster_lines, color=(0, 0, 255), thickness=6)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{debug_name}_overlay.jpg"), overlay)

    return lanes_only


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true",
                         help="also save hsv-mask/canny-in-roi per image to output/debug/")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isdir(COMMON_INPUT):
        print(f"ERROR: common_input folder not found at {os.path.abspath(COMMON_INPUT)}")
        sys.exit(1)

    files = sorted(f for f in os.listdir(COMMON_INPUT) if f.lower().endswith(VALID_EXTS))
    if not files:
        print(f"No images found in {os.path.abspath(COMMON_INPUT)}")
        sys.exit(1)

    print(f"Model 8 (classical CV): processing {len(files)} image(s) from {COMMON_INPUT}")
    for fname in files:
        in_path = os.path.join(COMMON_INPUT, fname)
        img = cv2.imread(in_path)
        if img is None:
            print(f"  [skip] could not read {fname}")
            continue

        name_noext = os.path.splitext(fname)[0]
        result = process_image(img, debug_name=name_noext, debug=args.debug)

        out_path = os.path.join(OUTPUT_DIR, f"{name_noext}_lanes.jpg")
        cv2.imwrite(out_path, result)
        print(f"  [ok] {fname} -> output/{name_noext}_lanes.jpg")

    print("Done.")
    if args.debug:
        print(f"Debug images saved to {os.path.abspath(DEBUG_DIR)}")


if __name__ == "__main__":
    main()

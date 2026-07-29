import os
import sys
import cv2
import time
import torch
import models
import traceback
import numpy as np

from pathlib import Path
from collections import OrderedDict

import torch.nn.functional as F

# ==========================================================
# Configuration
# ==========================================================

CHECKPOINT = "trained/ERFNet_trained.tar"

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / ".." / ".." / ".." / ".." / "common_input"
OUTPUT_DIR = SCRIPT_DIR / "output"

DEBUG_DIR = OUTPUT_DIR / "debug"
HEATMAP_DIR = OUTPUT_DIR / "heatmaps"
# ----------------------------------------------------------

MODEL_WIDTH = 976
MODEL_HEIGHT = 208

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
)

LANE_COLORS = [
    (0, 0, 255),      # Lane 1 - Red
    (0, 255, 0),      # Lane 2 - Green
    (255, 0, 0),      # Lane 3 - Blue
    (0, 255, 255)     # Lane 4 - Yellow
]

CONFIDENCE_THRESHOLD = 0.3
EXIST_THRESHOLD = 0.5

LANE_LINE_COLOR = (0, 0, 0)
LANE_LINE_THICKNESS = 3

DRIVABLE_FILL_COLOR = (0, 200, 0)
DRIVABLE_FILL_ALPHA = 0.35

# ==========================================================
# MULTI-PASS CROP (replaces single-anchor and multi-anchor-retry
# approaches used earlier)
# ==========================================================
MULTI_PASS_ANCHORS = (0.25, 0.45, 0.65, 0.85)


def extract_lane_curve(mask, min_points=8, outlier_px=40, smooth_window=9):
    """
    Given a boolean mask (H, W), trace a single smooth curve through it.
    (Unchanged from before -- used per-pass, before merging across passes.)
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

    half = smooth_window // 2
    local_trend = np.copy(raw_xs)
    for i in range(len(raw_xs)):
        lo = max(0, i - half)
        hi = min(len(raw_xs), i + half + 1)
        local_trend[i] = np.median(raw_xs[lo:hi])

    residual = np.abs(raw_xs - local_trend)
    inliers = residual < outlier_px

    if inliers.sum() < min_points:
        inliers = np.ones_like(inliers, dtype=bool)

    ys_in = raw_ys[inliers]
    xs_in = raw_xs[inliers]

    if len(ys_in) < 2:
        return None

    degree = 2 if len(ys_in) >= 6 else 1
    coeffs = np.polyfit(ys_in, xs_in, degree)
    poly = np.poly1d(coeffs)

    y_smooth = np.arange(ys_in.min(), ys_in.max() + 1)
    x_smooth = poly(y_smooth)

    curve = np.stack([x_smooth, y_smooth], axis=1).astype(np.int32)
    return curve


def merge_lane_curves(curves_list, degree=2):
    """
    Combine several (N,2) [x,y] curve arrays for the SAME lane -- one per
    crop pass -- into a single continuous curve.
    """
    if not curves_list:
        return None

    pass_fits = []
    for curve in curves_list:
        if len(curve) < 3:
            continue
        xs, ys = curve[:, 0].astype(float), curve[:, 1].astype(float)
        deg = degree if len(curve) >= degree + 4 else 1
        coeffs = np.polyfit(ys, xs, deg)
        pass_fits.append((float(ys.min()), float(ys.max()), coeffs))

    if not pass_fits:
        return None
    if len(pass_fits) == 1:
        lo, hi, coeffs = pass_fits[0]
        y_smooth = np.arange(int(lo), int(hi))
        x_smooth = np.polyval(coeffs, y_smooth)
        return np.stack([x_smooth, y_smooth], axis=1).astype(np.int32)

    pass_fits.sort(key=lambda r: r[0])
    y_min = min(r[0] for r in pass_fits)
    y_max = max(r[1] for r in pass_fits)
    y_full = np.arange(int(y_min), int(y_max))
    x_full = np.full(len(y_full), np.nan)

    for i, y in enumerate(y_full):
        covering = [(lo, hi, c) for lo, hi, c in pass_fits if lo <= y <= hi]
        if not covering:
            continue
        if len(covering) == 1:
            lo, hi, c = covering[0]
            x_full[i] = np.polyval(c, y)
        else:
            covering = sorted(covering, key=lambda r: abs(y - (r[0] + r[1]) / 2))[:2]
            covering = sorted(covering, key=lambda r: r[0])
            (lo1, hi1, c1), (lo2, hi2, c2) = covering[0], covering[1]
            overlap_lo, overlap_hi = lo2, hi1
            span = max(overlap_hi - overlap_lo, 1)
            t = np.clip((y - overlap_lo) / span, 0, 1)
            x1, x2 = np.polyval(c1, y), np.polyval(c2, y)
            x_full[i] = (1 - t) * x1 + t * x2

    valid = ~np.isnan(x_full)
    if valid.sum() < 2:
        return None
    return np.stack([x_full[valid], y_full[valid]], axis=1).astype(np.int32)


# ==========================================================
# CROSS-PASS LANE CORRESPONDENCE (new in v4)
# ==========================================================
# WHY: lane_idx (0-3) from the model is a POSITIONAL label ("2nd lane
# from left" etc.) assigned based on what's visible in THAT crop. Since
# each of the 4 passes sees a different vertical slice of road, there is
# no guarantee "lane_idx 2" in one pass refers to the same physical lane
# as "lane_idx 2" in another pass -- lanes can appear/disappear/reorder
# between crops. The old code merged purely by matching lane_idx number,
# which risked silently stitching together two DIFFERENT physical lanes
# into one corrupted curve. That's a worse failure than a curve just
# being short, because it can point the wrong direction entirely.
#
# FIX: before merging, verify that two passes' curves actually agree in
# x-position across their shared y-overlap. Only group curves that
# agree; anything that can't be confidently matched is kept as its OWN
# separate curve rather than force-merged. This is deliberately
# conservative -- worst case is an under-merged (shorter/segmented)
# lane, never a mis-merged one. That means this change can only ever
# equal or beat the old blind-lane_idx merge in terms of curve
# correctness; it should not make results worse. Set
# CORRESPONDENCE_MAX_X_DIFF generously so genuinely-matching lanes still
# merge -- this is a "reject obvious mismatches" filter, not a tight
# tolerance.

CORRESPONDENCE_MAX_X_DIFF = 60      # px; above this, two curves are NOT
                                     # considered the same physical lane
CORRESPONDENCE_MIN_OVERLAP_YS = 3   # need at least this many shared y's
                                     # to trust an agreement/disagreement
                                     # judgement at all


def _overlap_x_diff(curve_a, curve_b, min_overlap_ys=CORRESPONDENCE_MIN_OVERLAP_YS):
    """
    Median |x_a - x_b| over the y's where both curves have a point.
    Returns None if there isn't enough shared y-range to judge (NOT the
    same as "they disagree" -- absence of evidence isn't evidence of a
    mismatch, so callers must NOT treat None as a rejection).
    """
    map_a = {int(y): int(x) for x, y in curve_a}
    map_b = {int(y): int(x) for x, y in curve_b}
    common_ys = sorted(set(map_a.keys()) & set(map_b.keys()))
    if len(common_ys) < min_overlap_ys:
        return None
    diffs = [abs(map_a[y] - map_b[y]) for y in common_ys]
    return float(np.median(diffs))


def group_curves_across_passes(pass_results, max_x_diff=CORRESPONDENCE_MAX_X_DIFF):
    """
    Build groups of curves (from different passes) that are likely the
    SAME physical lane, using spatial agreement instead of trusting
    lane_idx. Returns a list of groups; each group is a list of dicts
    {"pass": pass_index, "lane_idx": original lane_idx, "curve": curve}.

    Rules (all conservative -- prefer under-merging over mis-merging):
      - Two curves from the SAME pass are NEVER grouped together (a
        single pass's lane 0 and lane 1 are, by construction, different
        physical lanes -- no correspondence check needed or wanted).
      - A curve only joins an existing group if it agrees (within
        max_x_diff) with EVERY current member it has enough y-overlap
        to compare against. One disagreement is enough to block the
        join, even if other members would have matched -- this avoids
        "transitive drift" where A~B and B~C get joined even though A
        and C are actually 60px+ apart.
      - A curve with NO measurable overlap against a group (no shared
        y's with any member) does not join it -- silence is not a
        match; it becomes its own group instead. This can lead to a
        real lane being split into more than one group (e.g. if two
        non-adjacent passes both see it but their common middle pass
        didn't detect it) -- that's an accepted, visible limitation
        (multiple shorter curves get drawn for that lane) rather than a
        silent wrong-merge.
    """
    entries = []
    for pass_i, r in enumerate(pass_results):
        for lane_idx, curve in r["lane_curves"].items():
            entries.append({"pass": pass_i, "lane_idx": lane_idx, "curve": curve})

    groups = []
    for entry in entries:
        matched_group = None
        for group in groups:
            compatible = True
            has_overlap_with_someone = False
            for member in group:
                if member["pass"] == entry["pass"]:
                    # never merge two curves from the same pass
                    compatible = False
                    break
                diff = _overlap_x_diff(entry["curve"], member["curve"])
                if diff is None:
                    continue  # no shared y-range with this member; uninformative
                has_overlap_with_someone = True
                if diff > max_x_diff:
                    compatible = False
                    break
            if compatible and has_overlap_with_someone:
                matched_group = group
                break
        if matched_group is not None:
            matched_group.append(entry)
        else:
            groups.append([entry])

    return groups


# ==========================================================
# Helper Functions
# ==========================================================

def banner():
    print("=" * 60, flush=True)
    print("           ERFNet Lane Detection", flush=True)
    print("=" * 60, flush=True)
    print("PyTorch :", torch.__version__, flush=True)
    print()


# ==========================================================
# Model
# ==========================================================

def load_model():
    print("Creating ERFNet model...", flush=True)
    model = models.ERFNet(5)

    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(f"Checkpoint not found:\n{os.path.abspath(CHECKPOINT)}")

    print("Loading checkpoint (this can take a little while on CPU)...", flush=True)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)

    print("Loading weights into model...", flush=True)
    state_dict = OrderedDict()
    for key, value in checkpoint["state_dict"].items():
        state_dict[key.replace("module.", "")] = value

    model.load_state_dict(state_dict)
    model.eval()

    print("Model loaded successfully.\n", flush=True)
    return model


# ==========================================================
# Image Preprocessing
# ==========================================================

def preprocess(image, anchor_center_frac):
    original = image.copy()
    original_h, original_w = original.shape[:2]

    target_ratio = MODEL_WIDTH / MODEL_HEIGHT
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
        crop_y_offset = max(0, min(crop_y_offset, original_h - crop_h))
        cropped = image[crop_y_offset: crop_y_offset + crop_h, :]
        crop_w = original_w

    resized = cv2.resize(cropped, (MODEL_WIDTH, MODEL_HEIGHT), interpolation=cv2.INTER_LINEAR)

    resized = resized.astype(np.float32)
    resized[:, :, 0] -= 103.939
    resized[:, :, 1] -= 116.779
    resized[:, :, 2] -= 123.680

    chw = resized.transpose(2, 0, 1)
    tensor = torch.from_numpy(chw).unsqueeze(0).float()

    pad_info = {
        "crop_x_offset": crop_x_offset, "crop_y_offset": crop_y_offset,
        "crop_w": crop_w, "crop_h": crop_h,
    }

    return tensor, original, original_h, original_w, pad_info


def run_inference_at_anchor(model, image, anchor_center_frac):
    input_tensor, original, original_h, original_w, pad_info = preprocess(image, anchor_center_frac)

    with torch.no_grad():
        output, lane_exist = model(input_tensor)

    output = F.softmax(output, dim=1)
    prediction = output.squeeze(0).cpu().numpy()   # (5, H, W)
    exist = lane_exist.squeeze(0).cpu().numpy()     # (4,)

    lane_curves = {}
    lane_prob_maps = {}

    for lane_idx in range(4):
        if exist[lane_idx] < EXIST_THRESHOLD:
            continue

        class_prob = prediction[lane_idx + 1]

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
        "input_tensor": input_tensor,
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

    model = load_model()

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
    print(f"Multi-pass anchors: {MULTI_PASS_ANCHORS} ({len(MULTI_PASS_ANCHORS)} passes/image)")
    print()

    if len(images) == 0:
        print("No supported images were found in that folder.")
        return

    processed = 0
    failed = 0

    for image_name in images:
        start = time.time()
        image_path = os.path.join(INPUT_DIR, image_name)

        print(f"Processing : {image_name}", flush=True)

        image = cv2.imread(image_path)
        if image is None:
            print("  Could not read image (corrupt file or unsupported format). Skipping.\n")
            failed += 1
            continue

        print(f"  Original size: {image.shape[1]}x{image.shape[0]} (w x h)", flush=True)

        try:
            pass_results = []
            for anchor in MULTI_PASS_ANCHORS:
                pass_results.append(run_inference_at_anchor(model, image, anchor))

            original = pass_results[0]["original"]
            original_h = pass_results[0]["original_h"]
            original_w = pass_results[0]["original_w"]

            all_exist = np.stack([r["exist"] for r in pass_results])
            exist_summary = all_exist.max(axis=0)
            print(f"  Lane existence scores (max across {len(MULTI_PASS_ANCHORS)} passes): "
                  f"{[round(float(e), 3) for e in exist_summary]}", flush=True)
            for pi, anchor in enumerate(MULTI_PASS_ANCHORS):
                per_pass_scores = [round(float(e), 3) for e in all_exist[pi]]
                print(f"    pass {pi} (anchor={anchor}): exist={per_pass_scores}", flush=True)

            # --- group curves across passes by spatial correspondence
            #     (NOT by raw lane_idx -- see group_curves_across_passes
            #     docstring for why lane_idx alone isn't trustworthy
            #     across differently-cropped passes), then merge each
            #     group's curves the same way as before ---
            groups = group_curves_across_passes(pass_results)

            # Log what got grouped with what, so a bad correspondence
            # decision is visible in the console instead of silent.
            print(f"  Cross-pass lane groups: {len(groups)}", flush=True)
            for gi, group in enumerate(groups):
                members = [(e["pass"], e["lane_idx"]) for e in group]
                print(f"    group {gi}: passes/lane_idx = {members}", flush=True)

            lane_curves = {}
            for gi, group in enumerate(groups):
                curves = [e["curve"] for e in group]
                merged = merge_lane_curves(curves)
                if merged is not None:
                    lane_curves[gi] = merged

            print(f"  Lanes with usable curve (merged): {list(lane_curves.keys())} (out of {len(groups)} groups)", flush=True)

            for lane_idx in range(4):
                candidates = [
                    (r["lane_prob_maps"][lane_idx].max(), r["lane_prob_maps"][lane_idx])
                    for r in pass_results if lane_idx in r["lane_prob_maps"]
                ]
                if candidates:
                    candidates.sort(key=lambda t: t[0], reverse=True)
                    best_prob_map = candidates[0][1]
                    heatmap = (np.clip(best_prob_map, 0, 1) * 255).astype(np.uint8)
                    cv2.imwrite(
                        str(HEATMAP_DIR / f"DEBUG_{Path(image_name).stem}_heatmap_lane{lane_idx}.jpg"),
                        heatmap
                    )
                    np.save(
                        str(HEATMAP_DIR / f"PROB_{Path(image_name).stem}_lane{lane_idx}.npy"),
                        best_prob_map.astype(np.float32)
                    )

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

            debug_tiles = []
            for anchor, r in zip(MULTI_PASS_ANCHORS, pass_results):
                input_tensor = r["input_tensor"]
                pad_info = r["pad_info"]
                debug_chw = input_tensor.squeeze(0).cpu().numpy()
                debug_bgr = debug_chw.transpose(1, 2, 0).copy()
                debug_bgr[:, :, 0] += 103.939
                debug_bgr[:, :, 1] += 116.779
                debug_bgr[:, :, 2] += 123.680
                debug_bgr = np.clip(debug_bgr, 0, 255).astype(np.uint8)

                for idx, curve in r["lane_curves"].items():
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
                            debug_bgr, [pts.reshape((-1, 1, 2))],
                            isClosed=False, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA
                        )
                cv2.putText(
                    debug_bgr, f"anchor={anchor}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA
                )
                debug_tiles.append(debug_bgr)

            debug_composite = np.vstack(debug_tiles)
            cv2.imwrite(
                str(DEBUG_DIR / f"DEBUG_{Path(image_name).stem}_networkinput.jpg"),
                debug_composite
            )

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


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("FATAL ERROR:")
        traceback.print_exc()

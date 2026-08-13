"""
infer_folder.py
Model 9: SCNN PyTorch converted from Torch7

ROBUST SCNN INFERENCE / VISUALIZATION

This version is intentionally designed for inference on arbitrary road
images rather than reproducing the original Torch7 visualization literally.

Important:
    - The SCNN existence head is NOT used as a hard gate.
    - Lane scoremaps are decoded directly.
    - Lane positions are traced from bottom -> top.
    - Local continuity is used to prevent random horizontal jumps.
    - Weighted centroids are used instead of raw argmax whenever possible.
    - Small gaps are interpolated.
    - Dense lane dots are drawn because the user prefers visible points.
    - Original image resolution is preserved for visualization.

Expected model output:
    output[0] = 1 x 5 x 288 x 800
                channel 0 = background
                channels 1..4 = lane 1..4

    output[1] = 1 x 4
                existence logits

The existence head of this converted checkpoint appears poorly calibrated
(~0 logits -> ~0.5 sigmoid), so it is intentionally diagnostic only.
"""

import os
import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image


# ============================================================
# Torch7 compatibility modules
# ============================================================

class ConcatTableModule(nn.Module):

    def __init__(self, children):
        super().__init__()
        self.mods = nn.ModuleList(children)

    def forward(self, x):
        return [m(x) for m in self.mods]


class ParallelTableModule(nn.Module):

    def __init__(self, children):
        super().__init__()
        self.mods = nn.ModuleList(children)

    def forward(self, xs):
        return [
            module(x)
            for module, x in zip(self.mods, xs)
        ]


class CAddTableModule(nn.Module):

    def forward(self, xs):

        out = xs[0]

        for x in xs[1:]:
            out = out + x

        return out


class JoinTableModule(nn.Module):

    def __init__(self, dimension):
        super().__init__()
        self.dim = int(dimension)

    def forward(self, xs):
        return torch.cat(
            xs,
            dim=self.dim
        )


class NarrowTableModule(nn.Module):

    def __init__(self, offset, length):
        super().__init__()

        self.offset = int(offset)
        self.length = int(length)

    def forward(self, xs):

        start = self.offset - 1

        return list(
            xs[
                start:
                start + self.length
            ]
        )


class SelectTableModule(nn.Module):

    def __init__(self, index):
        super().__init__()
        self.index = int(index)

    def forward(self, xs):

        if self.index > 0:
            idx = self.index - 1
        else:
            idx = self.index

        return xs[idx]


class FlattenTableModule(nn.Module):

    def forward(self, x):

        out = []

        def rec(v):

            if isinstance(v, (list, tuple)):

                for e in v:
                    rec(e)

            else:

                out.append(v)

        rec(x)

        return out


class IdentityModule(nn.Module):

    def forward(self, x):
        return x


class ContiguousModule(nn.Module):

    def forward(self, x):
        return x.contiguous()


# ============================================================
# Torch7 View
# ============================================================

class ViewModule(nn.Module):

    def __init__(
        self,
        num_elements,
        num_input_dims,
        size=None
    ):

        super().__init__()

        if num_elements is None:

            self.num_elements = tuple()

        elif isinstance(
            num_elements,
            (list, tuple)
        ):

            self.num_elements = tuple(
                int(v)
                for v in num_elements
            )

        else:

            try:

                self.num_elements = tuple(
                    int(v)
                    for v in num_elements
                )

            except TypeError:

                self.num_elements = (
                    int(num_elements),
                )

        self.num_input_dims = int(
            num_input_dims
        )

        if size is not None:

            try:

                self.view_size = tuple(
                    int(v)
                    for v in np.array(
                        size
                    ).reshape(-1)
                )

            except Exception:

                self.view_size = None

        else:

            self.view_size = None

    def forward(self, x):

        if self.view_size is not None:

            keep = tuple(
                x.shape[
                    : x.dim()
                    - self.num_input_dims
                ]
            )

            target = (
                keep
                + self.view_size
            )

            expected = 1

            for v in target:
                expected *= int(v)

            if expected != x.numel():

                raise RuntimeError(
                    "\n"
                    "Torch7 View shape mismatch\n"
                    "--------------------------------\n"
                    f"Input shape : {tuple(x.shape)}\n"
                    f"View size   : {self.view_size}\n"
                    f"Target      : {target}\n"
                    f"Input numel : {x.numel()}\n"
                    f"Target numel: {expected}\n"
                )

            return x.reshape(*target)

        if len(self.num_elements) == 0:

            raise RuntimeError(
                "Torch7 nn.View contains neither "
                "usable size nor numElements."
            )

        expected = 1

        for v in self.num_elements:
            expected *= int(v)

        if x.numel() != expected:

            raise RuntimeError(
                "\n"
                "Torch7 View fallback mismatch\n"
                "--------------------------------\n"
                f"Input shape : {tuple(x.shape)}\n"
                f"numElements: {self.num_elements}\n"
                f"Input numel : {x.numel()}\n"
                f"Expected    : {expected}\n"
            )

        keep = tuple(
            x.shape[
                : x.dim()
                - self.num_input_dims
            ]
        )

        return x.reshape(
            *keep,
            *self.num_elements
        )


# ============================================================
# SplitTable
# ============================================================

class SplitTableModule(nn.Module):

    def __init__(self, dimension):

        super().__init__()

        self.dim = int(
            dimension
        )

    def forward(self, x):

        return list(
            torch.unbind(
                x,
                dim=self.dim
            )
        )


# ============================================================
# Spatial SoftMax
# ============================================================

class SoftMaxChannel(nn.Module):

    def forward(self, x):

        return torch.softmax(
            x,
            dim=1
        )


# ============================================================
# Paths
# ============================================================

SCRIPT_DIR = Path(
    __file__
).resolve().parent

INPUT_DIR = (
    SCRIPT_DIR
    / ".."
    / ".."
    / ".."
    / "common_input"
)

OUTPUT_DIR = (
    SCRIPT_DIR
    / "output"
)

WEIGHTS_PATH = (
    SCRIPT_DIR
    / ".."
    / "weights"
    / "scnn_pytorch_fixed.pth"
)


# ============================================================
# Device
# ============================================================

DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)


# ============================================================
# Network configuration
# ============================================================

NET_W = 800
NET_H = 288

NUM_LANES = 4

NUM_ROW_ANCHORS = 18

CULANE_REF_H = 590


# ============================================================
# Decoder configuration
# ============================================================

# Minimum probability for a pixel to participate in the
# weighted lane centroid.
#
# We intentionally keep this lower than the old decoder.
PIXEL_THRESH = 0.18


# Minimum probability for accepting a row anchor.
ROW_THRESH = 0.20


# Maximum horizontal movement between consecutive anchors,
# measured in network pixels.
MAX_JUMP = 120


# Radius around the predicted x position that is searched.
SEARCH_RADIUS = 80


# How many nearby pixels participate in the weighted centroid.
CENTROID_RADIUS = 18


# Number of interpolated points between anchors.
INTERPOLATION_STEP = 4


# Dense visualization dot size.
DOT_RADIUS = 4


# Dense visualization line thickness.
LINE_WIDTH = 3


# ============================================================
# Normalization
# ============================================================

MEAN = np.array(
    [
        0.3598,
        0.3653,
        0.3662
    ],
    dtype=np.float32
)

STD = np.array(
    [
        0.2573,
        0.2663,
        0.2756
    ],
    dtype=np.float32
)


# ============================================================
# Colors
# ============================================================

LANE_COLORS = [
    (0, 255, 0),       # green
    (255, 0, 0),       # blue-ish in RGB, converted below
    (0, 0, 255),       # red
    (0, 255, 255)      # yellow
]


# ============================================================
# Image types
# ============================================================

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp"
)


# ============================================================
# Model loading
# ============================================================

def load_model():

    print(
        f"Loading model: {WEIGHTS_PATH} ..."
    )

    if not WEIGHTS_PATH.exists():

        raise FileNotFoundError(
            f"\nCheckpoint not found:\n"
            f"{WEIGHTS_PATH}"
        )

    model = torch.load(
        WEIGHTS_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    if isinstance(model, dict):

        raise TypeError(
            "Checkpoint contains a state_dict instead "
            "of a complete model."
        )

    model.to(
        DEVICE
    )

    model.eval()

    print(
        "Model loaded successfully."
    )

    print(
        f"Device: {DEVICE}"
    )

    print()

    return model


# ============================================================
# Preprocessing
# ============================================================

def preprocess(img):

    orig_w, orig_h = img.size

    # --------------------------------------------------------
    # Preserve aspect ratio.
    # --------------------------------------------------------

    scale = min(
        NET_W / orig_w,
        NET_H / orig_h
    )

    scaled_w = max(
        1,
        int(round(orig_w * scale))
    )

    scaled_h = max(
        1,
        int(round(orig_h * scale))
    )

    pad_left = (
        NET_W - scaled_w
    ) // 2

    pad_top = (
        NET_H - scaled_h
    ) // 2

    resized = img.resize(
        (
            scaled_w,
            scaled_h
        ),
        Image.BICUBIC
    )

    # --------------------------------------------------------
    # Mean-value padding.
    # --------------------------------------------------------

    mean_255 = tuple(
        int(
            round(
                float(v) * 255.0
            )
        )
        for v in MEAN
    )

    canvas = Image.new(
        "RGB",
        (
            NET_W,
            NET_H
        ),
        mean_255
    )

    canvas.paste(
        resized,
        (
            pad_left,
            pad_top
        )
    )

    # --------------------------------------------------------
    # Normalize.
    # --------------------------------------------------------

    arr = np.asarray(
        canvas
    ).astype(
        np.float32
    ) / 255.0

    arr = (
        arr - MEAN
    ) / STD

    tensor = (
        torch
        .from_numpy(arr)
        .permute(
            2,
            0,
            1
        )
        .unsqueeze(0)
        .float()
        .to(DEVICE)
    )

    return (
        tensor,
        scale,
        pad_left,
        pad_top
    )


# ============================================================
# Row anchors
# ============================================================

def get_row_anchors():

    anchors = []

    for i in range(
        NUM_ROW_ANCHORS
    ):

        row = (
            NET_H
            - (
                i
                * 20
                / CULANE_REF_H
                * NET_H
            )
        )

        row = int(
            round(row)
        )

        row = max(
            1,
            min(
                NET_H,
                row
            )
        )

        anchors.append(
            row - 1
        )

    return anchors


# ============================================================
# Find strongest local region
# ============================================================

def local_peak(
    row,
    center_x,
    radius
):

    h = row.shape[0]

    left = max(
        0,
        int(
            center_x - radius
        )
    )

    right = min(
        h,
        int(
            center_x + radius + 1
        )
    )

    if right <= left:

        return None

    local = row[
        left:right
    ]

    peak_local = int(
        np.argmax(local)
    )

    peak_x = (
        left
        + peak_local
    )

    peak_value = float(
        local[peak_local]
    )

    return (
        peak_x,
        peak_value,
        left,
        right
    )


# ============================================================
# Weighted centroid
# ============================================================

def weighted_centroid(
    row,
    center_x,
    radius
):

    result = local_peak(
        row,
        center_x,
        radius
    )

    if result is None:
        return None

    peak_x, peak_value, left, right = result

    # --------------------------------------------------------
    # We don't just take argmax.
    #
    # Give nearby high-probability pixels influence too.
    # This makes the lane much smoother.
    # --------------------------------------------------------

    c_left = max(
        left,
        peak_x - CENTROID_RADIUS
    )

    c_right = min(
        right,
        peak_x + CENTROID_RADIUS + 1
    )

    values = row[
        c_left:c_right
    ].astype(
        np.float64
    )

    # Remove very weak background-like responses.
    weights = np.maximum(
        values - PIXEL_THRESH,
        0.0
    )

    if (
        weights.sum() <= 1e-8
    ):

        return (
            float(peak_x),
            peak_value
        )

    xs = np.arange(
        c_left,
        c_right,
        dtype=np.float64
    )

    x = float(
        (
            xs * weights
        ).sum()
        / weights.sum()
    )

    return (
        x,
        peak_value
    )


# ============================================================
# Trace one lane
# ============================================================

def trace_lane(
    prob_map,
    anchors
):

    points = []

    previous_x = None

    # --------------------------------------------------------
    # Start from the bottom.
    #
    # This is important for road lanes because the lower part
    # usually provides the strongest spatial evidence.
    # --------------------------------------------------------

    for row_index in anchors:

        row = prob_map[
            row_index
        ]

        # ----------------------------------------------------
        # First point:
        # global search.
        # ----------------------------------------------------

        if previous_x is None:

            peak_x = int(
                np.argmax(row)
            )

            peak_value = float(
                row[peak_x]
            )

            result = weighted_centroid(
                row,
                peak_x,
                SEARCH_RADIUS
            )

        # ----------------------------------------------------
        # Later points:
        # local search around previous x.
        # ----------------------------------------------------

        else:

            result = weighted_centroid(
                row,
                previous_x,
                SEARCH_RADIUS
            )

            # ------------------------------------------------
            # If local search failed, try a global search.
            # ------------------------------------------------

            if result is None:

                peak_x = int(
                    np.argmax(row)
                )

                result = weighted_centroid(
                    row,
                    peak_x,
                    SEARCH_RADIUS
                )

        if result is None:

            continue

        x, peak_value = result

        # ----------------------------------------------------
        # Continuity check.
        # ----------------------------------------------------

        if previous_x is not None:

            jump = abs(
                x - previous_x
            )

            if jump > MAX_JUMP:

                # Try a wider search around the previous point.
                retry = weighted_centroid(
                    row,
                    previous_x,
                    MAX_JUMP
                )

                if retry is not None:

                    retry_x, retry_peak = retry

                    if abs(
                        retry_x - previous_x
                    ) <= MAX_JUMP:

                        x = retry_x
                        peak_value = retry_peak

                    else:

                        continue

                else:

                    continue

        # ----------------------------------------------------
        # Reject completely dead rows.
        # ----------------------------------------------------

        if (
            peak_value < ROW_THRESH
            and previous_x is not None
        ):

            continue

        points.append(
            (
                float(x),
                float(row_index)
            )
        )

        previous_x = x

    # --------------------------------------------------------
    # Need at least 3 anchors to consider this a lane.
    # --------------------------------------------------------

    if len(points) < 3:

        return []

    return points


# ============================================================
# Smooth lane
# ============================================================

def smooth_lane(
    points
):

    if len(points) < 3:

        return points

    arr = np.array(
        points,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Small moving-average smoothing.
    # --------------------------------------------------------

    smoothed = arr.copy()

    for i in range(
        1,
        len(arr) - 1
    ):

        smoothed[i, 0] = (
            0.25 * arr[i - 1, 0]
            + 0.50 * arr[i, 0]
            + 0.25 * arr[i + 1, 0]
        )

    return [
        (
            float(x),
            float(y)
        )
        for x, y in smoothed
    ]


# ============================================================
# Interpolate lane
# ============================================================

def interpolate_lane(
    points
):

    if len(points) < 2:

        return points

    points = sorted(
        points,
        key=lambda p: p[1],
        reverse=True
    )

    dense = []

    for i in range(
        len(points) - 1
    ):

        x1, y1 = points[i]
        x2, y2 = points[i + 1]

        dense.append(
            (
                x1,
                y1
            )
        )

        distance = abs(
            y2 - y1
        )

        steps = max(
            1,
            int(
                distance
                / INTERPOLATION_STEP
            )
        )

        for s in range(
            1,
            steps
        ):

            t = (
                s
                / steps
            )

            x = (
                x1
                + t
                * (x2 - x1)
            )

            y = (
                y1
                + t
                * (y2 - y1)
            )

            dense.append(
                (
                    x,
                    y
                )
            )

    dense.append(
        points[-1]
    )

    return dense


# ============================================================
# Network -> original coordinates
# ============================================================

def network_to_original(
    points,
    scale,
    pad_left,
    pad_top,
    orig_w,
    orig_h
):

    converted = []

    for x_net, y_net in points:

        x = (
            x_net
            - pad_left
        ) / scale

        y = (
            y_net
            - pad_top
        ) / scale

        x = max(
            0.0,
            min(
                float(orig_w - 1),
                x
            )
        )

        y = max(
            0.0,
            min(
                float(orig_h - 1),
                y
            )
        )

        converted.append(
            (
                int(round(x)),
                int(round(y))
            )
        )

    return converted


# ============================================================
# Draw lane
# ============================================================

def draw_lane(
    image,
    points,
    color
):

    if len(points) < 2:

        return

    # --------------------------------------------------------
    # Draw connecting line.
    # --------------------------------------------------------

    for i in range(
        len(points) - 1
    ):

        p1 = points[i]
        p2 = points[i + 1]

        cv2.line(
            image,
            p1,
            p2,
            color,
            LINE_WIDTH,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # Draw dense dots.
    # --------------------------------------------------------

    for x, y in points:

        cv2.circle(
            image,
            (
                int(x),
                int(y)
            ),
            DOT_RADIUS,
            color,
            -1,
            cv2.LINE_AA
        )


# ============================================================
# Calculate diagnostics
# ============================================================

def lane_diagnostics(
    prob_map
):

    peak = float(
        prob_map.max()
    )

    mean = float(
        prob_map.mean()
    )

    # --------------------------------------------------------
    # Count pixels with meaningful lane probability.
    # --------------------------------------------------------

    active = int(
        (
            prob_map
            > PIXEL_THRESH
        ).sum()
    )

    return (
        peak,
        mean,
        active
    )


# ============================================================
# Process image
# ============================================================

def process_image(
    model,
    image_path,
    output_dir
):

    name = os.path.basename(
        image_path
    )

    stem = os.path.splitext(
        name
    )[0]

    print(
        f"Processing : {name}"
    )

    # --------------------------------------------------------
    # Load image.
    # --------------------------------------------------------

    pil_img = Image.open(
        image_path
    ).convert(
        "RGB"
    )

    orig_w, orig_h = pil_img.size

    # --------------------------------------------------------
    # Preprocess.
    # --------------------------------------------------------

    (
        input_tensor,
        scale,
        pad_left,
        pad_top
    ) = preprocess(
        pil_img
    )

    # --------------------------------------------------------
    # Forward pass.
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            input_tensor
        )

    if not isinstance(
        output,
        (list, tuple)
    ):

        raise RuntimeError(
            "Model output is not a list/tuple."
        )

    if len(output) < 2:

        raise RuntimeError(
            "Model returned fewer than 2 outputs."
        )

    scoremap = output[0]
    exist_raw = output[1]

    print(
        f"  scoremap : "
        f"{tuple(scoremap.shape)}"
    )

    print(
        f"  exist    : "
        f"{tuple(exist_raw.shape)}"
    )

    # --------------------------------------------------------
    # Validate.
    # --------------------------------------------------------

    if tuple(scoremap.shape) != (
        1,
        5,
        NET_H,
        NET_W
    ):

        raise RuntimeError(
            "Unexpected scoremap shape: "
            f"{tuple(scoremap.shape)}"
        )

    # --------------------------------------------------------
    # Scoremap softmax.
    # --------------------------------------------------------

    probs = (
        torch
        .softmax(
            scoremap,
            dim=1
        )[0]
        .detach()
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Existence diagnostics.
    # --------------------------------------------------------

    raw_exist = (
        exist_raw[0]
        .detach()
        .cpu()
        .numpy()
    )

    exist = (
        torch
        .sigmoid(
            exist_raw[0]
        )
        .detach()
        .cpu()
        .numpy()
    )

    print(
        "  raw exist:"
        f" {[round(float(v), 4) for v in raw_exist]}"
    )

    print(
        "  existence:"
        f" {[round(float(v), 3) for v in exist]}"
    )

    # --------------------------------------------------------
    # Original image for OpenCV.
    # --------------------------------------------------------

    vis = cv2.imread(
        str(image_path)
    )

    if vis is None:

        raise RuntimeError(
            "OpenCV could not read image."
        )

    # --------------------------------------------------------
    # Row anchors.
    # --------------------------------------------------------

    anchors = get_row_anchors()

    print(
        "  anchors  : "
        f"{anchors}"
    )

    # --------------------------------------------------------
    # Lines file.
    # --------------------------------------------------------

    lines_path = (
        output_dir
        / f"{stem}_lines.txt"
    )

    lanes_drawn = 0

    with open(
        lines_path,
        "w"
    ) as fp:

        for lane_idx in range(
            NUM_LANES
        ):

            prob_map = probs[
                lane_idx + 1
            ]

            peak, mean, active = (
                lane_diagnostics(
                    prob_map
                )
            )

            print(
                f"  lane {lane_idx + 1}: "
                f"peak={peak:.3f} "
                f"mean={mean:.4f} "
                f"active={active}"
            )

            # ------------------------------------------------
            # Trace.
            # ------------------------------------------------

            network_points = trace_lane(
                prob_map,
                anchors
            )

            print(
                f"    anchor points: "
                f"{len(network_points)}"
            )

            if len(network_points) < 3:

                continue

            # ------------------------------------------------
            # Smooth.
            # ------------------------------------------------

            network_points = smooth_lane(
                network_points
            )

            # ------------------------------------------------
            # Densify.
            # ------------------------------------------------

            network_points = interpolate_lane(
                network_points
            )

            # ------------------------------------------------
            # Convert to original image.
            # ------------------------------------------------

            image_points = (
                network_to_original(
                    network_points,
                    scale,
                    pad_left,
                    pad_top,
                    orig_w,
                    orig_h
                )
            )

            if len(image_points) < 3:

                continue

            # ------------------------------------------------
            # Convert RGB -> BGR for OpenCV.
            # ------------------------------------------------

            rgb_color = LANE_COLORS[
                lane_idx
            ]

            bgr_color = (
                rgb_color[2],
                rgb_color[1],
                rgb_color[0]
            )

            # ------------------------------------------------
            # Draw.
            # ------------------------------------------------

            draw_lane(
                vis,
                image_points,
                bgr_color
            )

            # ------------------------------------------------
            # Write coordinates.
            # ------------------------------------------------

            for x, y in image_points:

                fp.write(
                    f"{x} {y} "
                )

            fp.write(
                "\n"
            )

            lanes_drawn += 1

    # --------------------------------------------------------
    # Save.
    # --------------------------------------------------------

    output_path = (
        output_dir
        / f"{stem}_overlay.jpg"
    )

    ok = cv2.imwrite(
        str(output_path),
        vis,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95
        ]
    )

    if not ok:

        raise RuntimeError(
            f"Could not save {output_path}"
        )

    print(
        f"  Saved: {output_path}"
    )

    print(
        f"  Lanes drawn: {lanes_drawn}"
    )


# ============================================================
# Find images
# ============================================================

def find_images():

    images = []

    for ext in IMAGE_EXTENSIONS:

        images.extend(
            glob.glob(
                str(
                    INPUT_DIR
                    / f"*{ext}"
                )
            )
        )

        images.extend(
            glob.glob(
                str(
                    INPUT_DIR
                    / f"*{ext.upper()}"
                )
            )
        )

    return sorted(
        set(images)
    )


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    images = find_images()

    print(
        "=" * 60
    )

    print(
        "        SCNN -- CULane (PyTorch)"
    )

    print(
        "        Robust Lane Decoder"
    )

    print(
        "=" * 60
    )

    print(
        f"Device       : {DEVICE}"
    )

    print(
        f"Input Folder : {INPUT_DIR}"
    )

    print(
        f"Output Folder: {OUTPUT_DIR}"
    )

    print(
        f"Images Found : {len(images)}"
    )

    print()

    if not images:

        print(
            "No supported images found."
        )

        return

    # --------------------------------------------------------
    # Load model once.
    # --------------------------------------------------------

    model = load_model()

    processed = 0
    failed = 0

    # --------------------------------------------------------
    # Process.
    # --------------------------------------------------------

    for image_path in images:

        try:

            process_image(
                model,
                image_path,
                OUTPUT_DIR
            )

            processed += 1

        except Exception as e:

            print(
                f"  ERROR while processing "
                f"{os.path.basename(image_path)}: "
                f"{type(e).__name__}: {e}"
            )

            failed += 1

        print()

    # --------------------------------------------------------
    # Summary.
    # --------------------------------------------------------

    print(
        "=" * 60
    )

    print(
        f"Done. Processed: {processed} "
        f" Failed: {failed} "
        f" Total: {len(images)}"
    )

    print(
        "=" * 60
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
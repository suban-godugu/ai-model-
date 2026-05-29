"""
Wafer Die Analytics v6
-----------------------------------------
FINAL STABLE VERSION

FEATURES:
✓ Adaptive X/Y pitch
✓ Projection-based pitch estimation
✓ Real die blob refinement
✓ Automatic offset alignment
✓ Wafer masking
✓ Die lattice reconstruction
✓ Good vs Fail die classification
✓ Visualization overlay
✓ NO CSV EXPORT

SUPPORTED:
✓ Donut
✓ Center
✓ Edge
✓ Edge-Loc
✓ Scratch
✓ Random
✓ Near-Full
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# CONFIG
# =====================================================

IMG_SIZE = 256

HSV_FAIL_LOWER = np.array([10, 40, 80])
HSV_FAIL_UPPER = np.array([80, 255, 255])

HSV_GOOD_LOWER = np.array([35, 25, 25])
HSV_GOOD_UPPER = np.array([95, 255, 255])

MIN_CELL_DIE_PIXELS = 3
MIN_WAFER_COVERAGE = 0.55

EDGE_MARGIN = 3

PITCH_MIN = 4
PITCH_MAX = 10

EXPECTED_DIES_ACROSS = 52

MAX_DIES_SANITY = 3500

# =====================================================
# LOAD IMAGE
# =====================================================

def load_rgb(path):

    img = cv2.imread(path)

    if img is None:
        raise ValueError(f"Could not load image: {path}")

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    img = cv2.resize(
        img,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_NEAREST
    )

    return img

# =====================================================
# BUILD MASKS
# =====================================================

def build_masks(rgb):

    hsv = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2HSV
    )

    fail_mask = cv2.inRange(
        hsv,
        HSV_FAIL_LOWER,
        HSV_FAIL_UPPER
    )

    good_mask = cv2.inRange(
        hsv,
        HSV_GOOD_LOWER,
        HSV_GOOD_UPPER
    )

    kernel = np.ones((2, 2), np.uint8)

    fail_mask = cv2.morphologyEx(
        fail_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    good_mask = cv2.morphologyEx(
        good_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    die_mask = cv2.bitwise_or(
        fail_mask,
        good_mask
    )

    return fail_mask, good_mask, die_mask

# =====================================================
# DETECT WAFER CIRCLE
# =====================================================

def wafer_circle(die_mask):

    contours, _ = cv2.findContours(
        die_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:

        c = IMG_SIZE // 2

        return c, c, IMG_SIZE // 2 - 4

    largest = max(
        contours,
        key=cv2.contourArea
    )

    (x, y), r = cv2.minEnclosingCircle(
        largest
    )

    return int(x), int(y), int(r)

# =====================================================
# AUTOCORRELATION PITCH
# =====================================================

def autocorr_fundamental_pitch(signal):

    s = signal.astype(np.float32)

    s -= s.mean()

    if np.std(s) < 1e-6:
        return 5

    corr = np.correlate(
        s,
        s,
        mode="full"
    )

    corr = corr[corr.size // 2:]

    best_lag = 5
    best_val = -1

    for lag in range(
        PITCH_MIN,
        PITCH_MAX + 1
    ):

        if lag >= len(corr):
            break

        val = corr[lag]

        if val > best_val:

            best_val = val
            best_lag = lag

    return best_lag

# =====================================================
# INITIAL PITCH ESTIMATION
# =====================================================

def estimate_pitch(
    die_mask,
    cx,
    cy,
    r
):

    h, w = die_mask.shape

    yy, xx = np.ogrid[:h, :w]

    inside = (
        (xx - cx) ** 2 +
        (yy - cy) ** 2
    ) <= (r - 2) ** 2

    dm = (die_mask > 0) & inside

    row_proj = dm.sum(axis=1).astype(np.float32)
    col_proj = dm.sum(axis=0).astype(np.float32)

    pr = autocorr_fundamental_pitch(row_proj)
    pc = autocorr_fundamental_pitch(col_proj)

    p_ac = int(np.median([pr, pc]))

    p_prior = int(
        np.clip(
            round((2 * r) / EXPECTED_DIES_ACROSS),
            PITCH_MIN,
            PITCH_MAX
        )
    )

    pitch = int(
        round(
            0.5 * p_ac +
            0.5 * p_prior
        )
    )

    return int(
        np.clip(
            pitch,
            PITCH_MIN,
            PITCH_MAX
        )
    )

# =====================================================
# REFINE PITCH
# =====================================================

def refine_pitch_from_components(
    good_mask,
    initial_pitch
):

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        good_mask,
        connectivity=8
    )

    widths = []
    heights = []

    for i in range(1, num_labels):

        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        area = stats[i, cv2.CC_STAT_AREA]

        if area < 5:
            continue

        if w < 2 or h < 2:
            continue

        if w > 20 or h > 20:
            continue

        aspect = w / max(h, 1)

        if aspect < 0.4 or aspect > 2.5:
            continue

        widths.append(w)
        heights.append(h)

    if len(widths) == 0:

        return initial_pitch, initial_pitch

    median_w = int(np.median(widths))
    median_h = int(np.median(heights))

    pitch_x = int(
        round(
            (initial_pitch + median_w) / 2
        )
    )

    pitch_y = int(
        round(
            (initial_pitch + median_h) / 2
        )
    )

    pitch_x = max(
        PITCH_MIN,
        min(pitch_x, PITCH_MAX)
    )

    pitch_y = max(
        PITCH_MIN,
        min(pitch_y, PITCH_MAX)
    )

    return pitch_x, pitch_y

# =====================================================
# CELL SCORE
# =====================================================

def cell_score(
    fail_mask,
    good_mask,
    die_mask,
    x1,
    y1,
    x2,
    y2
):

    roi_die = die_mask[y1:y2, x1:x2] > 0

    if roi_die.size == 0:
        return -1

    coverage = roi_die.mean()

    if coverage < MIN_WAFER_COVERAGE:
        return -1

    fp = int(
        np.sum(
            (fail_mask[y1:y2, x1:x2] > 0) &
            roi_die
        )
    )

    gp = int(
        np.sum(
            (good_mask[y1:y2, x1:x2] > 0) &
            roi_die
        )
    )

    total = fp + gp

    if total < MIN_CELL_DIE_PIXELS:
        return -1

    purity = max(fp, gp) / total

    return purity * coverage

# =====================================================
# BEST OFFSET SEARCH
# =====================================================

def find_best_offset(
    fail_mask,
    good_mask,
    die_mask,
    cx,
    cy,
    r,
    pitch_x,
    pitch_y
):

    h, w = die_mask.shape

    best_ox = 0
    best_oy = 0
    best_score = -1

    for oy in range(pitch_y):

        for ox in range(pitch_x):

            score_sum = 0
            count = 0

            for y1 in range(
                oy,
                h - pitch_y,
                pitch_y
            ):

                for x1 in range(
                    ox,
                    w - pitch_x,
                    pitch_x
                ):

                    x2 = x1 + pitch_x
                    y2 = y1 + pitch_y

                    ccx = (x1 + x2) // 2
                    ccy = (y1 + y2) // 2

                    dist = np.sqrt(
                        (ccx - cx) ** 2 +
                        (ccy - cy) ** 2
                    )

                    if dist > r - EDGE_MARGIN:
                        continue

                    sc = cell_score(
                        fail_mask,
                        good_mask,
                        die_mask,
                        x1,
                        y1,
                        x2,
                        y2
                    )

                    if sc >= 0:

                        score_sum += sc
                        count += 1

            avg = score_sum / max(count, 1)

            if avg > best_score:

                best_score = avg
                best_ox = ox
                best_oy = oy

    return best_ox, best_oy

# =====================================================
# CLASSIFY CELL
# =====================================================

def classify_cell(
    fail_mask,
    good_mask,
    die_mask,
    x1,
    y1,
    x2,
    y2
):

    roi_die = die_mask[y1:y2, x1:x2] > 0

    if roi_die.size == 0:
        return None

    if roi_die.mean() < MIN_WAFER_COVERAGE:
        return None

    fp = int(
        np.sum(
            (fail_mask[y1:y2, x1:x2] > 0) &
            roi_die
        )
    )

    gp = int(
        np.sum(
            (good_mask[y1:y2, x1:x2] > 0) &
            roi_die
        )
    )

    total = fp + gp

    if total < MIN_CELL_DIE_PIXELS:
        return None

    label = "FAIL" if fp > gp else "GOOD"

    ys, xs = np.where(roi_die)

    if len(xs) == 0:
        return None

    tx1 = x1 + xs.min()
    tx2 = x1 + xs.max()

    ty1 = y1 + ys.min()
    ty2 = y1 + ys.max()

    cx = (tx1 + tx2) // 2
    cy = (ty1 + ty2) // 2

    return {
        "center_x": cx,
        "center_y": cy,
        "x1": tx1,
        "y1": ty1,
        "x2": tx2,
        "y2": ty2,
        "label": label
    }

# =====================================================
# EXTRACT DIES
# =====================================================

def extract_dies(
    fail_mask,
    good_mask,
    die_mask,
    cx,
    cy,
    r,
    pitch_x,
    pitch_y,
    ox,
    oy
):

    h, w = die_mask.shape

    dies = []

    for y1 in range(
        oy,
        h - pitch_y,
        pitch_y
    ):

        for x1 in range(
            ox,
            w - pitch_x,
            pitch_x
        ):

            x2 = x1 + pitch_x
            y2 = y1 + pitch_y

            ccx = (x1 + x2) // 2
            ccy = (y1 + y2) // 2

            dist = np.sqrt(
                (ccx - cx) ** 2 +
                (ccy - cy) ** 2
            )

            if dist > r - EDGE_MARGIN:
                continue

            res = classify_cell(
                fail_mask,
                good_mask,
                die_mask,
                x1,
                y1,
                x2,
                y2
            )

            if res is not None:
                dies.append(res)

    return dies

# =====================================================
# ANALYZE WAFER
# =====================================================

def analyze_wafer(rgb):

    fail_mask, good_mask, die_mask = build_masks(rgb)

    cx, cy, r = wafer_circle(die_mask)

    initial_pitch = estimate_pitch(
        die_mask,
        cx,
        cy,
        r
    )

    pitch_x, pitch_y = refine_pitch_from_components(
        good_mask,
        initial_pitch
    )

    ox, oy = find_best_offset(
        fail_mask,
        good_mask,
        die_mask,
        cx,
        cy,
        r,
        pitch_x,
        pitch_y
    )

    dies = extract_dies(
        fail_mask,
        good_mask,
        die_mask,
        cx,
        cy,
        r,
        pitch_x,
        pitch_y,
        ox,
        oy
    )

    good = sum(
        1 for d in dies
        if d["label"] == "GOOD"
    )

    fail = sum(
        1 for d in dies
        if d["label"] == "FAIL"
    )

    total = good + fail

    vision_yield = (
        good / total * 100
    ) if total else 0

    meta = {
        "pitch_x": pitch_x,
        "pitch_y": pitch_y,
        "offset_x": ox,
        "offset_y": oy,
        "good": good,
        "fail": fail,
        "total": total,
        "vision_yield": vision_yield,
        "cx": cx,
        "cy": cy,
        "radius": r
    }

    return dies, meta

# =====================================================
# DRAW RESULTS
# =====================================================

def draw_debug(rgb, dies, meta):

    vis = rgb.copy()

    cv2.circle(
        vis,
        (meta["cx"], meta["cy"]),
        meta["radius"],
        (255, 255, 255),
        1
    )

    for d in dies:

        color = (
            (255, 120, 0)
            if d["label"] == "FAIL"
            else
            (0, 255, 0)
        )

        cv2.rectangle(
            vis,
            (d["x1"], d["y1"]),
            (d["x2"], d["y2"]),
            color,
            1
        )

    return vis

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    print("\n========== WAFER DIE ANALYTICS v6 ==========\n")

    image_path = input(
        "Enter FULL wafer image path:\n\nPath: "
    ).strip().strip('"')

    if not os.path.exists(image_path):

        print("\nERROR: Invalid image path")
        exit()

    rgb = load_rgb(image_path)

    dies, meta = analyze_wafer(rgb)

    print("\n========== RESULTS ==========\n")

    print(f"Pitch X       : {meta['pitch_x']}")
    print(f"Pitch Y       : {meta['pitch_y']}")

    print(f"Offset        : ({meta['offset_x']}, {meta['offset_y']})")

    print(f"\nGOOD dies     : {meta['good']}")
    print(f"FAIL dies     : {meta['fail']}")
    print(f"TOTAL dies    : {meta['total']}")

    print(f"\nVISION YIELD  : {meta['vision_yield']:.2f}%")

    # =============================================
    # SHOW SAMPLE DATA ONLY
    # =============================================

    df = pd.DataFrame(dies)

    print("\n========== SAMPLE DIE DATA ==========\n")

    print(
        df.head(20)
    )

    vis = draw_debug(
        rgb,
        dies,
        meta
    )

    plt.figure(figsize=(10, 10))

    plt.imshow(vis)

    plt.title(
        f"GOOD:{meta['good']} | "
        f"FAIL:{meta['fail']} | "
        f"PITCH_X:{meta['pitch_x']} | "
        f"PITCH_Y:{meta['pitch_y']} | "
        f"YIELD:{meta['vision_yield']:.2f}%"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.show()
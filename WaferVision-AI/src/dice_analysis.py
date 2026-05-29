import cv2
import numpy as np

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
# WAFER CIRCLE
# =====================================================

def wafer_circle(die_mask):

    contours, _ = cv2.findContours(
        die_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:

        c = IMG_SIZE // 2

        return c, c, IMG_SIZE // 2 - 5

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

def autocorr_pitch(signal):

    signal = signal.astype(np.float32)

    signal -= signal.mean()

    if np.std(signal) < 1e-6:
        return 5

    corr = np.correlate(
        signal,
        signal,
        mode="full"
    )

    corr = corr[corr.size // 2:]

    best_lag = 5
    best_val = -1

    for lag in range(
        PITCH_MIN,
        PITCH_MAX + 1
    ):

        if lag < len(corr):

            if corr[lag] > best_val:

                best_val = corr[lag]
                best_lag = lag

    return best_lag


# =====================================================
# ESTIMATE PITCH
# =====================================================

def estimate_pitch(
    die_mask,
    radius
):

    row_proj = die_mask.sum(axis=1)
    col_proj = die_mask.sum(axis=0)

    pr = autocorr_pitch(row_proj)
    pc = autocorr_pitch(col_proj)

    p_auto = int(np.median([pr, pc]))

    p_prior = int(
        round(
            (2 * radius) /
            EXPECTED_DIES_ACROSS
        )
    )

    pitch = int(
        round(
            (p_auto + p_prior) / 2
        )
    )

    pitch = max(
        PITCH_MIN,
        min(pitch, PITCH_MAX)
    )

    return pitch


# =====================================================
# FIND BEST OFFSET
# =====================================================

def find_best_offset(
    die_mask,
    cx,
    cy,
    radius,
    pitch
):

    h, w = die_mask.shape

    best_ox = 0
    best_oy = 0

    best_score = -1

    for oy in range(pitch):

        for ox in range(pitch):

            score = 0

            for y in range(
                oy,
                h - pitch,
                pitch
            ):

                for x in range(
                    ox,
                    w - pitch,
                    pitch
                ):

                    mx = x + pitch // 2
                    my = y + pitch // 2

                    dist = np.sqrt(
                        (mx - cx) ** 2 +
                        (my - cy) ** 2
                    )

                    if dist > radius - EDGE_MARGIN:
                        continue

                    roi = die_mask[
                        y:y + pitch,
                        x:x + pitch
                    ]

                    score += np.mean(roi > 0)

            if score > best_score:

                best_score = score

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

    roi_die = die_mask[
        y1:y2,
        x1:x2
    ] > 0

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

    label = (
        "FAIL"
        if fp > gp
        else "GOOD"
    )

    return {
        "label": label,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "fail_px": fp,
        "good_px": gp
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
    radius,
    pitch,
    ox,
    oy
):

    h, w = die_mask.shape

    dies = []

    row = 0

    for y in range(
        oy,
        h - pitch,
        pitch
    ):

        col = 0

        for x in range(
            ox,
            w - pitch,
            pitch
        ):

            mx = x + pitch // 2
            my = y + pitch // 2

            dist = np.sqrt(
                (mx - cx) ** 2 +
                (my - cy) ** 2
            )

            if dist > radius - EDGE_MARGIN:

                col += 1
                continue

            res = classify_cell(
                fail_mask,
                good_mask,
                die_mask,
                x,
                y,
                x + pitch,
                y + pitch
            )

            if res is not None:

                dies.append({
                    "die_row": row,
                    "die_col": col,
                    **res
                })

            col += 1

        row += 1

    return dies


# =====================================================
# ANALYZE WAFER
# =====================================================

def analyze_wafer(rgb):

    fail_mask, good_mask, die_mask = build_masks(rgb)

    cx, cy, radius = wafer_circle(die_mask)

    pitch = estimate_pitch(
        die_mask,
        radius
    )

    ox, oy = find_best_offset(
        die_mask,
        cx,
        cy,
        radius,
        pitch
    )

    dies = extract_dies(
        fail_mask,
        good_mask,
        die_mask,
        cx,
        cy,
        radius,
        pitch,
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

    yield_percent = (
        good / total * 100
    ) if total else 0

    fail_map = np.zeros(
        (IMG_SIZE, IMG_SIZE),
        dtype=np.float32
    )

    for d in dies:

        if d["label"] == "FAIL":

            fail_map[
                d["y1"]:d["y2"],
                d["x1"]:d["x2"]
            ] = 1.0

    result = {

        "dies": dies,

        "good": good,

        "fail": fail,

        "total": total,

        "yield": yield_percent,

        "pitch": pitch,

        "offset": (ox, oy),

        "map": fail_map,

        "wafer_center": (cx, cy),

        "radius": radius
    }

    return result


# =====================================================
# DEBUG VISUALIZATION
# =====================================================

def draw_debug(
    rgb,
    dies
):

    vis = rgb.copy()

    overlay = vis.copy()

    for d in dies:

        if d["label"] == "GOOD":

            fill = (0, 255, 0)
            border = (0, 120, 0)

        else:

            fill = (255, 120, 0)
            border = (255, 255, 255)

        cv2.rectangle(
            overlay,
            (d["x1"], d["y1"]),
            (d["x2"], d["y2"]),
            fill,
            -1
        )

        cv2.rectangle(
            vis,
            (d["x1"], d["y1"]),
            (d["x2"], d["y2"]),
            border,
            1
        )

    vis = cv2.addWeighted(
        overlay,
        0.45,
        vis,
        0.55,
        0
    )

    return vis


# =====================================================
# EXPORTS FOR DASHBOARD
# =====================================================

die_analysis = analyze_wafer
die_analysis_v4 = analyze_wafer
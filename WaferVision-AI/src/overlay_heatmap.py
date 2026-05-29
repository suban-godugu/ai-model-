import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
OUT_SIZE = 256
CX, CY, R = OUT_SIZE // 2, OUT_SIZE // 2, 120

GRID_SIZE = 48              # finer grid => less blocky than 32 (try 32/48/64)
DENSITY_SIGMA = 12          # extra smooth after upscale
P_LO, P_HI = 5, 95          # percentile contrast inside wafer
DISPLAY_GAMMA = 0.75

HOTSPOT_THRESH = 0.72       # on display-normalized heatmap (0..1)
HOTSPOT_MIN_AREA = 80       # px^2 on 256 canvas (lower than 250)
HOTSPOT_MAX_AREA = 200_000  # allow large ring/donut zones (was 12000)

# If one giant critical blob, also report "global critical" flag
GLOBAL_CRITICAL_FRAC = 0.35  # fraction of wafer disk above HOTSPOT_THRESH


def imread_unicode(path: str):
    path = path.strip().strip('"')
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return cv2.imread(path)
        return img
    except Exception:
        return cv2.imread(path)


def wafer_mask(h, w):
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, (w // 2, h // 2), R, 255, -1)
    return m


def defect_mask_wafermap(bgr, wafer_m):
    """
    WM-811K-style maps: yellow/green defect pixels vs teal background.
    Avoids adaptiveThreshold marking the whole textured wafer as defect.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hsv_m = cv2.inRange(hsv, (15, 35, 35), (55, 255, 255))

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    _, _, b = cv2.split(lab)
    _, lab_m = cv2.threshold(b, 150, 255, cv2.THRESH_BINARY)

    m = cv2.bitwise_or(hsv_m, lab_m)
    m = cv2.bitwise_and(m, wafer_m)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return m


def grid_density_from_binary(binary_u8, wafer_m, grid_size):
    h, w = binary_u8.shape
    cell_h = h // grid_size
    cell_w = w // grid_size
    hh, ww = cell_h * grid_size, cell_w * grid_size

    d = (binary_u8[:hh, :ww] > 0).astype(np.float32)
    wm = (wafer_m[:hh, :ww] > 0).astype(np.float32)

    # block reduce: sum defect / sum valid wafer pixels per cell
    d_blk = d.reshape(grid_size, cell_h, grid_size, cell_w).transpose(0, 2, 1, 3)
    wm_blk = wm.reshape(grid_size, cell_h, grid_size, cell_w).transpose(0, 2, 1, 3)

    defect_sum = d_blk.sum(axis=(2, 3))
    valid_sum = wm_blk.sum(axis=(2, 3)) + 1e-6
    heat_small = defect_sum / valid_sum  # 0..1 per cell

    # upscale with LINEAR first (less ringing than cubic on coarse grid)
    heat_up = cv2.resize(heat_small, (w, h), interpolation=cv2.INTER_LINEAR)
    heat_up = heat_up * (wafer_m > 0).astype(np.float32)
    return heat_small, heat_up


def robust_normalize01(x, wafer_m):
    disk = wafer_m > 0
    vals = x[disk].astype(np.float32)
    if vals.size < 50:
        return np.zeros_like(x, dtype=np.float32)
    lo = float(np.percentile(vals, P_LO))
    hi = float(np.percentile(vals, P_HI))
    if hi - lo < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    y = np.clip((x.astype(np.float32) - lo) / (hi - lo + 1e-8), 0, 1)
    y[~disk] = 0
    return y


def find_hotspot_contours(heatmap01, wafer_m):
    disk = wafer_m > 0
    crit = ((heatmap01 >= HOTSPOT_THRESH) & disk).astype(np.uint8) * 255

    # clean speckle, keep zones
    crit = cv2.morphologyEx(crit, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    crit = cv2.morphologyEx(crit, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    contours, _ = cv2.findContours(crit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = []
    for c in contours:
        a = cv2.contourArea(c)
        if HOTSPOT_MIN_AREA <= a <= HOTSPOT_MAX_AREA:
            valid.append(c)
    return valid, crit


def main():
    image_path = input("\nEnter wafer image path: ")
    bgr = imread_unicode(image_path)
    if bgr is None:
        raise ValueError("Could not read image.")

    bgr = cv2.resize(bgr, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)
    wafer_m = wafer_mask(OUT_SIZE, OUT_SIZE)
    original = bgr.copy()

    # --- defect evidence (color), not grayscale adaptiveThreshold ---
    binary = defect_mask_wafermap(bgr, wafer_m)

    # --- grid density ---
    heat_small, heat_up = grid_density_from_binary(binary, wafer_m, GRID_SIZE)

    # smooth continuous field
    heat_smooth = cv2.GaussianBlur(heat_up, (0, 0), sigmaX=DENSITY_SIGMA, sigmaY=DENSITY_SIGMA)

    # display heatmap with robust contrast (fixes flat/all-red)
    heat_disp = robust_normalize01(heat_smooth, wafer_m)
    heat_disp = np.power(heat_disp, DISPLAY_GAMMA)

    disk = wafer_m > 0
    disk_px = int(np.sum(disk))

    # true defect fraction from binary mask
    defect_pixel_frac = float(np.sum((binary > 0) & disk)) / max(disk_px, 1) * 100.0

    # grid-cell analytics (what your old prints actually were)
    crit_cells = int(np.sum(heat_small > 0.75))
    med_cells = int(np.sum((heat_small > 0.4) & (heat_small <= 0.75)))
    low_cells = int(np.sum(heat_small <= 0.4))

    # pixel-level analytics on display heatmap
    crit_px = int(np.sum((heat_disp > 0.75) & disk))
    med_px = int(np.sum((heat_disp > 0.40) & (heat_disp <= 0.75) & disk))
    low_px = int(np.sum((heat_disp <= 0.40) & disk))

    hotspot_score = float(np.mean(heat_disp[disk]) * 100.0)
    global_critical = (crit_px / max(disk_px, 1)) >= GLOBAL_CRITICAL_FRAC

    # hotspot zones / boxes
    valid_contours, crit_map = find_hotspot_contours(heat_disp, wafer_m)

    # color overlay
    heat_u8 = np.clip(heat_disp * 255, 0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_color = cv2.bitwise_and(heat_color, heat_color, mask=wafer_m)

    overlay = cv2.addWeighted(original, 0.65, heat_color, 0.35, 0)

    for c in valid_contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(overlay, "CRITICAL", (x, max(0, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # if ring is one huge zone, draw its bbox anyway
    if len(valid_contours) == 0 and global_critical:
        ys, xs = np.where((heat_disp >= HOTSPOT_THRESH) & disk)
        if xs.size > 0:
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(overlay, "GLOBAL CRITICAL", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    print("\n===== HOTSPOT ANALYTICS (OPTIMIZED) =====")
    print(f"Defect pixel fraction (mask): {defect_pixel_frac:.2f}%")
    print(f"Hotspot score (display mean): {hotspot_score:.2f}%")
    print(f"Grid cells critical (>0.75):  {crit_cells} / {GRID_SIZE*GRID_SIZE}")
    print(f"Grid cells medium:            {med_cells}")
    print(f"Grid cells low:               {low_cells}")
    print(f"Pixels critical (display):    {crit_px}")
    print(f"Pixels medium (display):      {med_px}")
    print(f"Pixels low (display):         {low_px}")
    print(f"Clustered hotspot zones:      {len(valid_contours)}")
    print(f"Global critical pattern:      {global_critical}")

    plt.figure(figsize=(9, 9))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Wafer Hotspot Analytics (optimized)", fontsize=14, fontweight="bold")
    plt.figtext(
        0.02, 0.02,
        "Blue   = Low (relative, p5–p95)\n"
        "Green  = Medium\n"
        "Yellow = High\n"
        "Red    = Critical (relative)\n"
        "Boxes  = hotspot components\n\n"
        "Uses color defect mask + grid density.",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.88),
    )
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
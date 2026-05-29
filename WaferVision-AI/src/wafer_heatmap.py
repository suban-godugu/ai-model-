import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# =========================
# CONFIG
# =========================

OUT_SIZE = 256
CX, CY, R = OUT_SIZE // 2, OUT_SIZE // 2, int(OUT_SIZE * 0.47)  # ~120/256

# Density smoothing (sigma in pixels, after building a soft defect map)
DENSITY_SIGMA = 10

# Display normalization (percentiles inside wafer disk)
P_LO, P_HI = 2, 98

# Gamma < 1 boosts midtones (more visible structure). Increase toward 1.0 if too "rainbow".
DISPLAY_GAMMA = 0.85

# Hotspot extraction on *display heatmap* (0..1), after percentile norm
HOTSPOT_THRESH = 0.78
HOTSPOT_MIN_AREA = 120          # px^2 at 256 (tune)
HOTSPOT_MAX_AREA = 80_000     # px^2 at 256 (tune)

# Morphology on defect binary before density (noise control)
OPEN_K = 3
CLOSE_K = 5


def imread_unicode(path: str):
    path = path.strip().strip('"')
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return cv2.imread(path, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return cv2.imread(path, cv2.IMREAD_COLOR)


def wafer_mask(h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, (w // 2, h // 2), R, 255, -1)
    return m


def defect_mask_color_wafermap(bgr: np.ndarray, wafer_m: np.ndarray) -> np.ndarray:
    """
    For WM-811K-like maps: defects are bright yellow/green-ish vs teal background.
    This avoids adaptiveThreshold turning the whole textured wafer into 'defect'.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hsv, (15, 35, 35), (55, 255, 255))

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    _, _, b = cv2.split(lab)
    _, lab_mask = cv2.threshold(b, 150, 255, cv2.THRESH_BINARY)

    m = cv2.bitwise_or(hsv_mask, lab_mask)
    m = cv2.bitwise_and(m, wafer_m)

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_K, OPEN_K))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_K, CLOSE_K))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close)
    return m


def robust_normalize01(x: np.ndarray, wafer_m: np.ndarray) -> np.ndarray:
    """Percentile normalize using only wafer disk pixels (fixes 'all red')."""
    disk = wafer_m > 0
    vals = x[disk].astype(np.float32)
    if vals.size < 50:
        return np.zeros_like(x, dtype=np.float32)

    lo = float(np.percentile(vals, P_LO))
    hi = float(np.percentile(vals, P_HI))
    if hi - lo < 1e-6:
        return np.zeros_like(x, dtype=np.float32)

    y = (x.astype(np.float32) - lo) / (hi - lo + 1e-8)
    y = np.clip(y, 0.0, 1.0)
    y[disk == 0] = 0.0
    return y


def grid_density(defect_bin: np.ndarray, wafer_m: np.ndarray, grid: int = 32) -> np.ndarray:
    """
    Optional interpretable density: fraction of defect pixels per grid cell, upsampled.
    Great when the wafer is 'Near-Full' noise — reduces speckle dominance.
    """
    h, w = defect_bin.shape
    cell_h, cell_w = h // grid, w // grid
    hh, ww = cell_h * grid, cell_w * grid

    d = (defect_bin[:hh, :ww] > 0).astype(np.float32)
    wm = (wafer_m[:hh, :ww] > 0).astype(np.float32)

    d = d.reshape(grid, cell_h, grid, cell_w).transpose(0, 2, 1, 3).reshape(grid, grid, -1)
    wm = wm.reshape(grid, cell_h, grid, cell_w).transpose(0, 2, 1, 3).reshape(grid, grid, -1)

    frac = d.sum(axis=2) / (wm.sum(axis=2) + 1e-6)
    frac_up = cv2.resize(frac, (w, h), interpolation=cv2.INTER_CUBIC)
    frac_up = frac_up * (wafer_m > 0).astype(np.float32)
    return frac_up


def main():
    image_path = input("\nEnter wafer image path: ")
    bgr = imread_unicode(image_path)
    if bgr is None:
        raise ValueError("Invalid image path / could not read image.")

    bgr = cv2.resize(bgr, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)
    wafer_m = wafer_mask(OUT_SIZE, OUT_SIZE)

    # Defect evidence map (0/255) -> float soft density
    defect_bin = defect_mask_color_wafermap(bgr, wafer_m)

    # Choose ONE density basis (toggle):
    USE_GRID_DENSITY = True  # recommended for noisy / near-full patterns

    if USE_GRID_DENSITY:
        raw = grid_density(defect_bin, wafer_m, grid=32)
        density = cv2.GaussianBlur(raw, (0, 0), sigmaX=DENSITY_SIGMA, sigmaY=DENSITY_SIGMA)
    else:
        raw = (defect_bin > 0).astype(np.float32)
        density = cv2.GaussianBlur(raw, (0, 0), sigmaX=DENSITY_SIGMA, sigmaY=DENSITY_SIGMA)

    heatmap01 = robust_normalize01(density, wafer_m)
    heatmap01 = np.power(heatmap01, DISPLAY_GAMMA)

    # Analytics (clear naming)
    disk = wafer_m > 0
    disk_area = int(np.sum(disk))

    defect_frac = float(np.sum((defect_bin > 0) & disk)) / max(disk_area, 1)  # true defect pixel fraction
    mean_density_01 = float(np.mean(heatmap01[disk]))  # display-normalized mean

    crit_pixels = int(np.sum((heatmap01 > 0.75) & disk))
    med_pixels = int(np.sum((heatmap01 > 0.40) & (heatmap01 <= 0.75) & disk))
    low_pixels = int(np.sum((heatmap01 <= 0.40) & disk))

    # Hotspot zones from threshold map (not from raw defect noise)
    crit_map = ((heatmap01 >= HOTSPOT_THRESH) & disk).astype(np.uint8) * 255
    crit_map = cv2.morphologyEx(crit_map, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    crit_map = cv2.morphologyEx(crit_map, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(crit_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = []
    for c in contours:
        a = cv2.contourArea(c)
        if HOTSPOT_MIN_AREA <= a <= HOTSPOT_MAX_AREA:
            valid.append(c)

    # Visualization base
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray, gray, mask=wafer_m)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    heat_u8 = np.clip(heatmap01 * 255.0, 0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_color = cv2.bitwise_and(heat_color, heat_color, mask=wafer_m)

    overlay = cv2.addWeighted(base, 0.62, heat_color, 0.38, 0)

    for c in valid:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(overlay, "HOTSPOT", (x, max(0, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

    print("\n===== WAfer DENSITY ANALYTICS (OPTIMIZED) =====")
    print(f"Defect pixel fraction (mask): {defect_frac * 100:.2f}% of wafer disk")
    print(f"Mean display density (01):    {mean_density_01 * 100:.2f}%")
    print(f"Pixels heatmap > 0.75:        {crit_pixels}")
    print(f"Pixels 0.40–0.75:           {med_pixels}")
    print(f"Pixels <= 0.40:             {low_pixels}")
    print(f"Hotspot zones (components): {len(valid)}")

    plt.figure(figsize=(9, 9))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Continuous Wafer Density Analytics (optimized)", fontsize=14, fontweight="bold")
    plt.figtext(
        0.02, 0.02,
        "Blue   = Low (relative)\n"
        "Green  = Medium\n"
        "Yellow = High\n"
        "Red    = Critical (relative)\n\n"
        "Note: colors are percentile-scaled inside wafer disk,\n"
        "not absolute fab yield.",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.88),
    )
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
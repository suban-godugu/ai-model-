import os
import cv2
import numpy as np
from tqdm import tqdm

IMAGE_DIR = "data/images"
MASK_DIR = "data/masks"
os.makedirs(MASK_DIR, exist_ok=True)

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def imread_unicode(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        return cv2.imread(path)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def imwrite_unicode(path: str, img: np.ndarray) -> bool:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    elif ext == ".png":
        ok, buf = cv2.imencode(".png", img)
    else:
        ok, buf = cv2.imencode(".png", img)
    if not ok:
        return False
    try:
        buf.tofile(path)
        return True
    except Exception:
        return cv2.imwrite(path, img)


def detect_class(filename: str) -> str:
    # Keep your tokens; order is fine for these names.
    for token, name in [
        ("Donut", "Donut"),
        ("Scratch", "Scratch"),
        ("Edge-Loc", "Edge-Loc"),
        ("Edge-Ring", "Edge-Ring"),
        ("Random", "Random"),
        ("Near-Full", "Near-Full"),
        ("Center", "Center"),
        ("Local", "Local"),
        ("Normal", "Normal"),
    ]:
        if token in filename:
            return name
    return "Other"


def wafer_disk_mask(h: int, w: int, cx: int = 112, cy: int = 112, r: int = 104) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, (cx, cy), r, 255, -1)
    return m


def hsv_yellow_mask_bgr(img_bgr_224: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr_224, cv2.COLOR_BGR2HSV)
    lower = np.array([15, 35, 35], np.uint8)
    upper = np.array([50, 255, 255], np.uint8)
    return cv2.inRange(hsv, lower, upper)


def lab_b_mask_bgr(img_bgr_224: np.ndarray, thr: int = 150) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr_224, cv2.COLOR_BGR2LAB)
    _, _, b = cv2.split(lab)
    _, m = cv2.threshold(b, thr, 255, cv2.THRESH_BINARY)
    return m


def strict_non_defect_seed_bgr(img_bgr_224: np.ndarray, wafer_m: np.ndarray) -> np.ndarray:
    """
    Pixels that are very unlikely to be yellow defect.
    Used to flood-fill 'hole' regions from wafer center (good for Donut centers).
    """
    hsv = cv2.cvtColor(img_bgr_224, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Tight: low saturation OR low value => not the bright defect paint
    not_bright_defect = cv2.bitwise_or(cv2.inRange(s, 0, 60), cv2.inRange(v, 0, 70))

    # Also exclude obvious yellow hue band with enough saturation/value
    yellowish = cv2.inRange(hsv, np.array([15, 40, 40], np.uint8), np.array([50, 255, 255], np.uint8))
    seed = cv2.bitwise_and(not_bright_defect, cv2.bitwise_not(yellowish))
    seed = cv2.bitwise_and(seed, wafer_m)
    return seed


def largest_k_components(binary_u8: np.ndarray, k: int = 1) -> np.ndarray:
    """Keep largest k connected components (removes salt noise)."""
    if binary_u8.max() == 0:
        return binary_u8
    num, labels, stats, _ = cv2.connectedComponentsWithStats((binary_u8 > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        return binary_u8

    areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num)]
    areas.sort(key=lambda x: x[1], reverse=True)
    keep = {i for i, _ in areas[: max(1, k)]}

    out = np.zeros_like(binary_u8)
    for i in keep:
        out[labels == i] = 255
    return out


def donut_recover_ring(defect_u8: np.ndarray, img_bgr_224: np.ndarray, wafer_m: np.ndarray) -> np.ndarray:
    """
    If HSV/LAB merged into a blob, recover a ring-like mask by carving an interior 'hole'
    region reachable from wafer center through strict non-defect pixels.
    """
    defect = cv2.bitwise_and(defect_u8, wafer_m)
    if defect.max() == 0:
        return defect

    h, w = defect.shape[:2]
    cx, cy = w // 2, h // 2

    seed = strict_non_defect_seed_bgr(img_bgr_224, wafer_m)

    # Flood from center if center looks like non-defect interior
    flood = np.zeros((h + 2, w + 2), np.uint8)
    interior = seed.copy()

    if int(interior[cy, cx]) == 0:
        # Center isn't a safe seed; fall back to largest component mask only
        return largest_k_components(defect, k=1)

    cv2.floodFill(interior, flood, (cx, cy), 255)
    hole = interior  # flooded region

    # Expand hole slightly to cut bridges between ring legs
    k = np.ones((5, 5), np.uint8)
    hole_d = cv2.dilate(hole, k, iterations=1)

    ring = cv2.bitwise_and(defect, cv2.bitwise_not(hole_d))

    # If carving removed everything, revert to largest defect component (safer than empty)
    if ring.max() == 0:
        return largest_k_components(defect, k=1)

    # Keep main ring structure, drop speckles
    ring = largest_k_components(ring, k=1)
    return ring


def scratch_keep_thin_components(defect_u8: np.ndarray, wafer_m: np.ndarray) -> np.ndarray:
    defect = cv2.bitwise_and(defect_u8, wafer_m)

    # Light open/dilate to reduce noise but preserve thin structures
    k2 = np.ones((2, 2), np.uint8)
    defect = cv2.morphologyEx(defect, cv2.MORPH_OPEN, k2)
    defect = cv2.dilate(defect, k2, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats((defect > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        return defect

    out = np.zeros_like(defect)
    for i in range(1, num):
        x, y, ww, hh, area = stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3], stats[i, 4]
        if area < 6:
            continue
        aspect = ww / float(hh + 1e-6)
        inv_aspect = hh / float(ww + 1e-6)
        ar = max(aspect, inv_aspect)

        # Keep elongated components OR moderately sized blobs (scratch can be short)
        if ar >= 2.2 or area >= 40:
            out[labels == i] = 255

    return out


def fuse_masks(hsv_m: np.ndarray, lab_m: np.ndarray, defect_class: str) -> np.ndarray:
    """
    Key optimization:
    - Donut / Edge-Ring: prefer HSV AND LAB (reduces false bridging into the hole)
    - Near-Full: OR is OK (want recall)
    - Scratch: HSV-first + small LAB contribution (optional)
    """
    if defect_class in ("Donut", "Edge-Ring"):
        m = cv2.bitwise_and(hsv_m, lab_m)
        # If AND is too sparse, fall back progressively
        if int(cv2.countNonZero(m)) < 80:  # tune if needed
            m2 = cv2.bitwise_and(hsv_m, cv2.dilate(lab_m, np.ones((3, 3), np.uint8), iterations=1))
            m = m2
        return m

    if defect_class == "Scratch":
        # Scratches are often yellow-ish; LAB can add speckle—use mostly HSV
        m = cv2.bitwise_or(hsv_m, cv2.bitwise_and(lab_m, cv2.dilate(hsv_m, np.ones((3, 3), np.uint8), iterations=1)))
        return m

    if defect_class == "Near-Full":
        return cv2.bitwise_or(hsv_m, lab_m)

    # Default: OR (recall), then components cleanup will remove speckle
    return cv2.bitwise_or(hsv_m, lab_m)


def morphology_for_class(mask_u8: np.ndarray, defect_class: str) -> np.ndarray:
    m = mask_u8
    if defect_class == "Scratch":
        k = np.ones((2, 2), np.uint8)
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    if defect_class == "Donut":
        k = np.ones((3, 3), np.uint8)  # smaller than 5x5 to avoid closing the hole
        return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    if defect_class == "Edge-Ring":
        k = np.ones((4, 4), np.uint8)
        return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    if defect_class == "Near-Full":
        k = np.ones((6, 6), np.uint8)
        return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    if defect_class == "Random":
        k = np.ones((3, 3), np.uint8)
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, k)

    k = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)


def create_mask(img_bgr: np.ndarray, defect_class: str) -> np.ndarray:
    img = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_AREA)
    wafer_m = wafer_disk_mask(224, 224)

    if defect_class == "Normal":
        return np.zeros((224, 224), np.uint8)

    hsv_m = hsv_yellow_mask_bgr(img)
    lab_m = lab_b_mask_bgr(img, thr=150)

    mask = fuse_masks(hsv_m, lab_m, defect_class)
    mask = cv2.bitwise_and(mask, wafer_m)
    mask = morphology_for_class(mask, defect_class)

    if defect_class == "Scratch":
        mask = scratch_keep_thin_components(mask, wafer_m)
        mask[mask > 0] = 255
        return mask

    if defect_class == "Donut":
        mask = donut_recover_ring(mask, img, wafer_m)
        mask[mask > 0] = 255
        return mask

    # Generic: keep largest component(s)
    if defect_class in ("Near-Full", "Edge-Ring", "Center", "Edge-Loc", "Local", "Random", "Other"):
        k_keep = 2 if defect_class == "Random" else 1
        mask = largest_k_components(mask, k=k_keep)

    # Area gating (class-aware)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(mask)
    for c in contours:
        a = cv2.contourArea(c)
        if a <= 1:
            continue
        if defect_class == "Near-Full":
            if a > 50:
                cv2.drawContours(clean, [c], -1, 255, -1)
        else:
            if 20 < a < 30000:
                cv2.drawContours(clean, [c], -1, 255, -1)

    clean[clean > 0] = 255
    return clean


def main():
    files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(VALID_EXTENSIONS)]
    processed = failed = 0
    fails = []

    for f in tqdm(files, desc="masks"):
        p = os.path.join(IMAGE_DIR, f)
        img = imread_unicode(p)
        if img is None:
            failed += 1
            fails.append(f)
            continue

        cls = detect_class(f)
        m = create_mask(img, cls)
        out = os.path.join(MASK_DIR, f)

        if imwrite_unicode(out, m):
            processed += 1
        else:
            failed += 1
            fails.append(f)

    print(f"OK={processed}, FAIL={failed}, TOTAL={len(files)}")
    if fails[:20]:
        print("Failures (sample):", fails[:20])


if __name__ == "__main__":
    main()
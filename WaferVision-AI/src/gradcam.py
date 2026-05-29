import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from preprocess import test_transform
from model import get_model


# =========================================================
# CONFIG
# =========================================================

classes = [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Local",
    "Near-Full",
    "Normal",
    "Random",
    "Scratch",
]

WAFER_CX, WAFER_CY, WAFER_R = 112, 112, 105

USE_TWO_SCALE_CAM = True

HEATMAP_ALPHA = 0.55
BINARY_ALPHA = 0.35
ORIGINAL_WEIGHT_IN_BINARY_BLEND = 0.55

# NOTE: Near-Full must use a MUCH lower percentile than "local" defects,
# otherwise CAM looks like tiny hotspots (your screenshot).
THRESH_PERCENTILE = {
    "Center": 92,
    "Donut": 93,
    "Edge-Loc": 90,
    "Edge-Ring": 92,
    "Local": 90,
    # key change:
    "Near-Full": 72,     # try 68–78 if needed
    "Normal": 98.5,
    "Random": 85,
    "Scratch": 88,
    "Other": 92,
}

FIXED_THRESHOLDS = {
    "Center": 0.50,
    "Donut": 0.55,
    "Edge-Loc": 0.45,
    "Edge-Ring": 0.50,
    "Local": 0.45,
    "Near-Full": 0.35,   # lower because activations are spread out / weaker
    "Normal": 0.85,
    "Random": 0.35,
    "Scratch": 0.40,
}

USE_ADAPTIVE_PERCENTILE_THRESH = True

# Hybrid wafer-map defect extraction (important for Near-Full)
USE_COLOR_MASK_FOR_CLASSES = {"Near-Full", "Random"}  # add/remove as you validate
COLOR_MASK_DILATE_ITERS_NEARFULL = 2


def imread_unicode(path: str):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return cv2.imread(path)
        return img
    except Exception:
        return cv2.imread(path)


def make_wafer_mask(h=224, w=224):
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(m, (WAFER_CX, WAFER_CY), WAFER_R, 255, -1)
    return m


def apply_wafer_mask_rgb(image_rgb: np.ndarray, wafer_mask: np.ndarray) -> np.ndarray:
    m3 = wafer_mask.astype(bool)
    out = image_rgb.copy()
    out[~m3] = 0
    return out


def normalize_cam(cam: np.ndarray, wafer_mask: np.ndarray) -> np.ndarray:
    cam = cam.astype(np.float32)
    m = wafer_mask.astype(bool)
    vals = cam[m]
    if vals.size == 0:
        return np.zeros_like(cam, dtype=np.float32)
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo < 1e-8:
        out = np.zeros_like(cam, dtype=np.float32)
        return out
    out = (cam - lo) / (hi - lo + 1e-8)
    out = out * (wafer_mask.astype(np.float32) / 255.0)
    return out


def compute_gradcam_plusplus(model, input_tensor, target_class: int, target_layers):
    cam_engine = GradCAMPlusPlus(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam_engine(input_tensor=input_tensor, targets=targets)

    if isinstance(grayscale_cam, list):
        grayscale_cam = grayscale_cam[0]
    grayscale_cam = np.asarray(grayscale_cam, dtype=np.float32)
    if grayscale_cam.ndim == 3:
        grayscale_cam = grayscale_cam[0]
    return grayscale_cam


def fuse_two_layer_cams(model, input_tensor, target_class: int):
    cam3 = compute_gradcam_plusplus(model, input_tensor, target_class, [model.layer3[-1]])
    cam4 = compute_gradcam_plusplus(model, input_tensor, target_class, [model.layer4[-1]])
    cam3 = cv2.resize(cam3, (224, 224), interpolation=cv2.INTER_CUBIC)
    cam4 = cv2.resize(cam4, (224, 224), interpolation=cv2.INTER_CUBIC)
    return 0.35 * cam3 + 0.65 * cam4


def adaptive_threshold(cam_norm: np.ndarray, wafer_mask: np.ndarray, predicted_class: str) -> float:
    m = wafer_mask.astype(bool)
    vals = cam_norm[m]
    if vals.size < 50:
        return float(FIXED_THRESHOLDS.get(predicted_class, 0.50))

    p = float(THRESH_PERCENTILE.get(predicted_class, THRESH_PERCENTILE["Other"]))
    thr = float(np.percentile(vals, p))
    return float(np.clip(thr, 0.03, 0.95))


def morphology_kernels(predicted_class: str):
    if predicted_class == "Donut":
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    if predicted_class == "Scratch":
        return cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))
    if predicted_class == "Random":
        return np.ones((2, 2), np.uint8)
    if predicted_class == "Near-Full":
        # larger connectivity helps merge near-full coverage
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return np.ones((3, 3), np.uint8)


def color_defect_mask_bgr(image_bgr_224: np.ndarray, wafer_mask_u8: np.ndarray, predicted_class: str) -> np.ndarray:
    """
    Wafer-map style defect extraction (yellow-ish defect pixels).
    This is the main fix for Near-Full looking 'sparse' in GradCAM.
    """
    hsv = cv2.cvtColor(image_bgr_224, cv2.COLOR_BGR2HSV)
    lower = np.array([15, 35, 35], np.uint8)
    upper = np.array([55, 255, 255], np.uint8)
    hsv_mask = cv2.inRange(hsv, lower, upper)

    lab = cv2.cvtColor(image_bgr_224, cv2.COLOR_BGR2LAB)
    _, _, b = cv2.split(lab)
    _, lab_mask = cv2.threshold(b, 150, 255, cv2.THRESH_BINARY)

    # Near-full maps often need recall: OR is appropriate
    m = cv2.bitwise_or(hsv_mask, lab_mask)
    m = cv2.bitwise_and(m, wafer_mask_u8)

    if predicted_class == "Near-Full":
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
        m = cv2.dilate(m, k, iterations=COLOR_MASK_DILATE_ITERS_NEARFULL)
    else:
        k = np.ones((3, 3), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

    m = (m > 0).astype(np.uint8) * 255
    return m


def cam_binary_mask(cam_norm: np.ndarray, wafer_mask_u8: np.ndarray, predicted_class: str) -> np.ndarray:
    if USE_ADAPTIVE_PERCENTILE_THRESH:
        thr = adaptive_threshold(cam_norm, wafer_mask_u8, predicted_class)
    else:
        thr = float(FIXED_THRESHOLDS.get(predicted_class, 0.50))

    defect = np.zeros_like(cam_norm, dtype=np.uint8)
    defect[cam_norm >= thr] = 255

    kernel = morphology_kernels(predicted_class)
    defect = cv2.morphologyEx(defect, cv2.MORPH_OPEN, kernel)
    defect = cv2.morphologyEx(defect, cv2.MORPH_CLOSE, kernel)

    # Near-full: extra closing to merge scattered CAM islands
    if predicted_class == "Near-Full":
        kbig = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        defect = cv2.morphologyEx(defect, cv2.MORPH_CLOSE, kbig, iterations=2)

    defect = cv2.bitwise_and(defect, wafer_mask_u8)
    return defect, thr


def donut_ring_recovery(defect_mask_u8: np.ndarray, cam_norm: np.ndarray, wafer_mask_u8: np.ndarray) -> np.ndarray:
    wafer = wafer_mask_u8.astype(bool)
    defect = (defect_mask_u8 > 0) & wafer

    low = (cam_norm < np.percentile(cam_norm[wafer], 35)) & wafer
    interior = np.zeros_like(defect_mask_u8, dtype=np.uint8)
    interior[low.astype(np.uint8) > 0] = 255

    h, w = defect_mask_u8.shape
    cx, cy = WAFER_CX, WAFER_CY
    if int(interior[cy, cx]) == 0:
        return defect_mask_u8

    flood = np.zeros((h + 2, w + 2), np.uint8)
    interior_flood = interior.copy()
    cv2.floodFill(interior_flood, flood, (cx, cy), 255)

    hole_u8 = (interior_flood > 0).astype(np.uint8) * 255
    hole_d = cv2.dilate(hole_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)

    ring = (defect.astype(np.uint8) * 255)
    ring = cv2.bitwise_and(ring, cv2.bitwise_not(hole_d))
    if int(ring.max()) == 0:
        return defect_mask_u8
    return ring


def connected_components_cleanup(defect_mask_u8: np.ndarray, predicted_class: str):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(defect_mask_u8, connectivity=8)
    clean = np.zeros_like(defect_mask_u8)

    max_area = {
        "Near-Full": 200000,  # effectively "no cap" at 224
        "Random": 200000,
        "Donut": 20000,
        "Edge-Ring": 20000,
        "Center": 20000,
        "Edge-Loc": 20000,
        "Local": 20000,
        "Scratch": 20000,
        "Normal": 20000,
    }.get(predicted_class, 20000)

    min_area = 25 if predicted_class != "Scratch" else 10

    critical_regions = 0
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        clean[labels == i] = 255
        critical_regions += 1

    return clean, critical_regions


def mask_area_fraction(mask_u8: np.ndarray, wafer_mask_u8: np.ndarray) -> float:
    wafer_pixels = int(np.sum(wafer_mask_u8 > 0))
    if wafer_pixels <= 0:
        return 0.0
    defect_pixels = int(np.sum((mask_u8 > 0) & (wafer_mask_u8 > 0)))
    return defect_pixels / wafer_pixels * 100.0


def fuse_hybrid_defect_mask(cam_mask_u8, color_mask_u8, predicted_class: str):
    if predicted_class in USE_COLOR_MASK_FOR_CLASSES:
        fused = cv2.bitwise_or(cam_mask_u8, color_mask_u8)
        return fused, "hybrid(CAM+color)"
    return cam_mask_u8, "CAM-only"


def nearfull_auto_fallback(cam_mask_u8, color_mask_u8, wafer_mask_u8, predicted_class: str):
    """
    If CAM still covers too little for Near-Full, prefer the color mask (more faithful to 'coverage').
    """
    if predicted_class != "Near-Full":
        return cam_mask_u8, "no-fallback"

    cam_frac = mask_area_fraction(cam_mask_u8, wafer_mask_u8)
    col_frac = mask_area_fraction(color_mask_u8, wafer_mask_u8)

    # Heuristic: near-full should usually imply substantial coverage on wafer-map images
    if cam_frac < 12.0 and col_frac > cam_frac + 3.0:
        return color_mask_u8, f"fallback=color (CAM={cam_frac:.2f}% < color={col_frac:.2f}%)"

    return cv2.bitwise_or(cam_mask_u8, color_mask_u8), "hybrid(CAM+color)"


def render_heatmap_overlay(image_rgb: np.ndarray, cam_norm: np.ndarray, wafer_mask_u8: np.ndarray) -> np.ndarray:
    cam_u8 = np.clip(cam_norm * 255.0, 0, 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

    m = (wafer_mask_u8 > 0).astype(np.float32)[..., None]
    a = HEATMAP_ALPHA * m
    out = (image_rgb.astype(np.float32) * (1.0 - a) + heat_rgb.astype(np.float32) * a).clip(0, 255).astype(np.uint8)
    out[wafer_mask_u8 == 0] = 0
    return out


def render_soft_defect_tint(image_rgb: np.ndarray, defect_mask_u8: np.ndarray, wafer_mask_u8: np.ndarray) -> np.ndarray:
    m = (defect_mask_u8 > 0) & (wafer_mask_u8 > 0)
    tint = image_rgb.astype(np.float32)
    red = np.array([255, 0, 0], dtype=np.float32)
    tint[m] = tint[m] * (1.0 - BINARY_ALPHA) + red * BINARY_ALPHA
    tint[wafer_mask_u8 == 0] = 0
    return tint.clip(0, 255).astype(np.uint8)


def risk_level(defect_density: float, predicted_class: str) -> str:
    # Near-full: risk should not be "LOW" if coverage is huge
    if predicted_class == "Near-Full" and defect_density > 35:
        return "HIGH" if defect_density < 70 else "VERY HIGH"

    if defect_density > 50:
        return "VERY HIGH"
    if defect_density > 25:
        return "HIGH"
    if defect_density > 10:
        return "MEDIUM"
    return "LOW"


def main():
    device = torch.device("cpu")

    model = get_model(len(classes))
    state = torch.load("wafer_model.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()

    image_path = input("\nEnter wafer image path: ").strip().strip('"')
    image_bgr = imread_unicode(image_path)
    if image_bgr is None:
        raise ValueError("\nInvalid image path (could not read image).")

    image_bgr = cv2.resize(image_bgr, (224, 224), interpolation=cv2.INTER_AREA)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    wafer_mask = make_wafer_mask()
    image_rgb_masked = apply_wafer_mask_rgb(image_rgb, wafer_mask)

    pil_image = Image.fromarray(image_rgb_masked)
    input_tensor = test_transform(pil_image).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, predicted = torch.max(probs, 1)

    predicted_class = classes[predicted.item()]
    confidence_score = float(confidence.item() * 100.0)
    print(f"\nPredicted Class : {predicted_class}")
    print(f"Confidence      : {confidence_score:.2f}%")

    # GradCAM++
    if USE_TWO_SCALE_CAM:
        fused_cam = fuse_two_layer_cams(model, input_tensor, predicted.item())
    else:
        fused_cam = compute_gradcam_plusplus(model, input_tensor, predicted.item(), [model.layer4[-1]])
        fused_cam = cv2.resize(fused_cam, (224, 224), interpolation=cv2.INTER_CUBIC)

    fused_cam = np.maximum(fused_cam, 0).astype(np.float32)
    fused_cam = cv2.bilateralFilter(fused_cam, 7, 50, 50)

    cam_norm = normalize_cam(fused_cam, wafer_mask)

    cam_mask, cam_thr = cam_binary_mask(cam_norm, wafer_mask, predicted_class)

    color_mask = color_defect_mask_bgr(image_bgr, wafer_mask, predicted_class)

    if predicted_class == "Near-Full":
        defect_mask, fusion_note = nearfull_auto_fallback(cam_mask, color_mask, wafer_mask, predicted_class)
    else:
        defect_mask, fusion_note = fuse_hybrid_defect_mask(cam_mask, color_mask, predicted_class)

    if predicted_class == "Donut":
        defect_mask = donut_ring_recovery(defect_mask, cam_norm, wafer_mask)

    defect_mask, critical_regions = connected_components_cleanup(defect_mask, predicted_class)

    defect_density = mask_area_fraction(defect_mask, wafer_mask)
    vision_clear_proxy = max(0.0, 100.0 - defect_density)
    risk = risk_level(defect_density, predicted_class)

    heat_overlay = render_heatmap_overlay(image_rgb_masked, cam_norm, wafer_mask)
    tinted = render_soft_defect_tint(image_rgb_masked, defect_mask, wafer_mask)

    bin_vis = np.zeros_like(image_rgb)
    bin_vis[:] = [0, 0, 255]
    bin_vis[defect_mask > 0] = [255, 0, 0]
    bin_vis = cv2.bitwise_and(bin_vis, bin_vis, mask=wafer_mask)
    bin_blend = cv2.addWeighted(image_rgb_masked, ORIGINAL_WEIGHT_IN_BINARY_BLEND, bin_vis, 1.0 - ORIGINAL_WEIGHT_IN_BINARY_BLEND, 0)

    contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) > 25:
            cv2.drawContours(heat_overlay, [c], -1, (255, 255, 255), 1)
            cv2.drawContours(tinted, [c], -1, (255, 255, 255), 1)
            cv2.drawContours(bin_blend, [c], -1, (255, 255, 255), 1)

    cv2.circle(heat_overlay, (WAFER_CX, WAFER_CY), WAFER_R, (255, 255, 255), 2)
    cv2.circle(tinted, (WAFER_CX, WAFER_CY), WAFER_R, (255, 255, 255), 2)
    cv2.circle(bin_blend, (WAFER_CX, WAFER_CY), WAFER_R, (255, 255, 255), 2)

    # Optional debug panel numbers
    cam_only_frac = mask_area_fraction(cam_mask, wafer_mask)
    color_only_frac = mask_area_fraction(color_mask, wafer_mask)

    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    plt.imshow(heat_overlay)
    plt.title(
        "ResNet50 + GradCAM++ (continuous heatmap)\n"
        f"Class: {predicted_class} | Confidence: {confidence_score:.2f}%",
        fontsize=14,
        fontweight="bold",
    )
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(tinted)
    plt.title("Hybrid defect mask (recommended for Near-Full)\n(white contours)", fontsize=14, fontweight="bold")
    plt.axis("off")

    metrics_text = (
        f"Fusion Mode       : {fusion_note}\n\n"
        f"Critical Regions  : {critical_regions}\n\n"
        f"Vision Defect Area: {defect_density:.2f}% of wafer disk\n"
        f"  - CAM-only area : {cam_only_frac:.2f}%\n"
        f"  - Color-only    : {color_only_frac:.2f}%\n\n"
        f"Vision Clear Proxy: {vision_clear_proxy:.2f}% (NOT tester yield)\n\n"
        f"Risk Level         : {risk}\n\n"
        f"CAM Thr (percentile mode): {cam_thr:.3f}\n"
        f"(Near-Full uses hybrid CAM+HSV/LAB by design)\n"
    )

    plt.figtext(
        0.02,
        0.02,
        metrics_text,
        fontsize=10,
        color="white",
        bbox=dict(facecolor="black", alpha=0.90, edgecolor="white"),
    )

    plt.figtext(
        0.62,
        0.02,
        "Why Near-Full looked wrong before:\n"
        "GradCAM is sparse by nature for global patterns.\n"
        "Hybrid adds wafer-map color defect pixels.\n\n"
        "Tune knobs:\n"
        "- THRESH_PERCENTILE['Near-Full'] (68–78)\n"
        "- COLOR_MASK_DILATE_ITERS_NEARFULL\n"
        "- HSV/LAB thresholds in color_defect_mask_bgr()",
        fontsize=9,
        color="white",
        bbox=dict(facecolor="black", alpha=0.90, edgecolor="white"),
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
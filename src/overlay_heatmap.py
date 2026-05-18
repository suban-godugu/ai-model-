import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===================================
# INPUT IMAGE
# ===================================

image_path = input("\nEnter wafer image path: ")

# ===================================
# LOAD IMAGE
# ===================================

image = cv2.imread(image_path)

# ===================================
# RESIZE IMAGE
# ===================================

image = cv2.resize(image, (256, 256))

original = image.copy()

# ===================================
# GRAYSCALE
# ===================================

gray = cv2.cvtColor(

    image,

    cv2.COLOR_BGR2GRAY
)

# ===================================
# ADAPTIVE DEFECT EXTRACTION
# ===================================

binary = cv2.adaptiveThreshold(

    gray,

    255,

    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,

    cv2.THRESH_BINARY,

    21,

    5
)

# ===================================
# INVERT FOR DEFECT ENHANCEMENT
# ===================================

binary = cv2.bitwise_not(binary)

# ===================================
# REMOVE SMALL NOISE
# ===================================

kernel = np.ones((3,3), np.uint8)

binary = cv2.morphologyEx(

    binary,

    cv2.MORPH_OPEN,

    kernel
)

# ===================================
# CREATE CIRCULAR MASK
# ===================================

mask = np.zeros((256, 256), dtype=np.uint8)

cv2.circle(

    mask,

    (128, 128),

    120,

    255,

    -1
)

# ===================================
# APPLY MASK
# ===================================

binary = cv2.bitwise_and(

    binary,

    binary,

    mask=mask
)

# ===================================
# GRID SETTINGS
# ===================================

GRID_SIZE = 32

cell_h = gray.shape[0] // GRID_SIZE
cell_w = gray.shape[1] // GRID_SIZE

# ===================================
# HEATMAP STORAGE
# ===================================

heatmap = np.zeros((GRID_SIZE, GRID_SIZE))

# ===================================
# DENSITY ESTIMATION
# ===================================

for i in range(GRID_SIZE):

    for j in range(GRID_SIZE):

        y1 = i * cell_h
        y2 = (i + 1) * cell_h

        x1 = j * cell_w
        x2 = (j + 1) * cell_w

        cell = binary[y1:y2, x1:x2]

        defect_pixels = np.sum(cell == 255)

        density = defect_pixels / (cell_h * cell_w)

        heatmap[i, j] = density

# ===================================
# SMOOTH NONLINEAR NORMALIZATION
# ===================================

heatmap = np.power(

    heatmap,

    0.6
)

if np.max(heatmap) != 0:

    heatmap = heatmap / np.max(heatmap)

# ===================================
# HOTSPOT SCORE
# ===================================

hotspot_score = np.mean(heatmap) * 100

# ===================================
# RISK ANALYSIS
# ===================================

critical_regions = np.sum(heatmap > 0.75)

medium_regions = np.sum(

    (heatmap > 0.4) & (heatmap <= 0.75)
)

low_regions = np.sum(heatmap <= 0.4)

# ===================================
# UPSCALE HEATMAP
# ===================================

heatmap_resized = cv2.resize(

    heatmap,

    (256, 256),

    interpolation=cv2.INTER_CUBIC
)

# ===================================
# ADVANCED SMOOTHING
# ===================================

heatmap_smooth = cv2.GaussianBlur(

    heatmap_resized,

    (21, 21),

    0
)

# ===================================
# HEATMAP COLORIZATION
# ===================================

heatmap_colored = cv2.applyColorMap(

    np.uint8(255 * heatmap_smooth),

    cv2.COLORMAP_JET
)

# ===================================
# APPLY MASK TO HEATMAP
# ===================================

heatmap_colored = cv2.bitwise_and(

    heatmap_colored,

    heatmap_colored,

    mask=mask
)

# ===================================
# CRITICAL HOTSPOT EXTRACTION
# ===================================

critical_map = np.uint8(

    heatmap_smooth > 0.72
) * 255

# ===================================
# MERGE NEARBY REGIONS
# ===================================

kernel = np.ones((9,9), np.uint8)

critical_map = cv2.dilate(

    critical_map,

    kernel,

    iterations=2
)

# ===================================
# FIND HOTSPOT CONTOURS
# ===================================

contours, _ = cv2.findContours(

    critical_map,

    cv2.RETR_EXTERNAL,

    cv2.CHAIN_APPROX_SIMPLE
)

# ===================================
# DRAW CLUSTERED HOTSPOTS
# ===================================

valid_contours = 0

for contour in contours:

    area = cv2.contourArea(contour)

    # REMOVE GIANT FALSE BOUNDARY BOXES

    if area > 250 and area < 12000:

        valid_contours += 1

        x, y, w, h = cv2.boundingRect(contour)

        # INDUSTRIAL BOUNDING BOX

        cv2.rectangle(

            image,

            (x, y),

            (x + w, y + h),

            (0, 0, 255),

            2
        )

        # HOTSPOT LABEL

        cv2.putText(

            image,

            "CRITICAL",

            (x, y - 8),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (0, 0, 255),

            2
        )

# ===================================
# INDUSTRY-LEVEL OVERLAY
# ===================================

overlay = cv2.addWeighted(

    image,

    0.65,

    heatmap_colored,

    0.35,

    0
)

# ===================================
# ANALYTICS SUMMARY
# ===================================

print("\n===== HOTSPOT ANALYTICS =====")

print(f"\nHotspot Score: {hotspot_score:.2f}%")

print(f"Critical Regions: {critical_regions}")

print(f"Medium Risk Regions: {medium_regions}")

print(f"Low Risk Regions: {low_regions}")

print(f"Clustered Hotspot Zones: {valid_contours}")

# ===================================
# DISPLAY RESULT
# ===================================

plt.figure(figsize=(9,9))

plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

plt.title(

    "Wafer Hotspot Analytics",

    fontsize=14,

    fontweight='bold'
)

# ===================================
# LEGEND
# ===================================

plt.figtext(

    0.02,

    0.02,

    "Blue   = Low Density\n"
    "Green  = Medium Density\n"
    "Yellow = High Density\n"
    "Red    = Critical Hotspot\n"
    "Boxes  = Clustered Critical Zones",

    fontsize=10,

    bbox=dict(

        facecolor='white',

        alpha=0.85
    )
)

plt.axis('off')

plt.tight_layout()

plt.show()
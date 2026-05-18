import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===================================
# IMAGE INPUT
# ===================================

image_path = input("\nEnter wafer image path: ")

# ===================================
# LOAD IMAGE
# ===================================

image = cv2.imread(

    image_path,

    cv2.IMREAD_GRAYSCALE
)

# ===================================
# RESIZE IMAGE
# ===================================

image = cv2.resize(

    image,

    (256, 256)
)

# ===================================
# CREATE CIRCULAR WAFER MASK
# ===================================

mask = np.zeros(

    (256, 256),

    dtype=np.uint8
)

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

image = cv2.bitwise_and(

    image,

    image,

    mask=mask
)

# ===================================
# ADAPTIVE DEFECT EXTRACTION
# ===================================

binary = cv2.adaptiveThreshold(

    image,

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
# EXTRACT DEFECT COORDINATES
# ===================================

points = np.column_stack(

    np.where(binary > 0)
)

# ===================================
# EMPTY HEATMAP
# ===================================

heatmap = np.zeros(

    (256, 256),

    dtype=np.float32
)

# ===================================
# GAUSSIAN DENSITY PROPAGATION
# ===================================

for point in points:

    y, x = point

    cv2.circle(

        heatmap,

        (x, y),

        8,

        1,

        -1
    )

# ===================================
# CONTINUOUS DENSITY SMOOTHING
# ===================================

heatmap = cv2.GaussianBlur(

    heatmap,

    (0, 0),

    sigmaX=15,

    sigmaY=15
)

# ===================================
# NORMALIZE HEATMAP
# ===================================

heatmap = (

    heatmap - heatmap.min()

) / (

    heatmap.max() - heatmap.min() + 1e-8
)

# ===================================
# INDUSTRY-STYLE CONTRAST BOOST
# ===================================

heatmap = np.power(

    heatmap,

    0.65
)

# ===================================
# APPLY CIRCULAR MASK
# ===================================

heatmap = heatmap * (

    mask / 255.0
)

# ===================================
# HOTSPOT ANALYTICS
# ===================================

hotspot_score = np.mean(heatmap) * 100

critical_regions = np.sum(heatmap > 0.75)

medium_regions = np.sum(

    (heatmap > 0.4) & (heatmap <= 0.75)
)

low_regions = np.sum(heatmap <= 0.4)

# ===================================
# CREATE CRITICAL REGION MAP
# ===================================

critical_map = np.uint8(

    heatmap > 0.72
) * 255

# ===================================
# MERGE NEARBY HOTSPOTS
# ===================================

kernel = np.ones((7,7), np.uint8)

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
# CONVERT TO COLOR IMAGE
# ===================================

image_color = cv2.cvtColor(

    image,

    cv2.COLOR_GRAY2BGR
)

# ===================================
# DRAW CLUSTERED HOTSPOTS
# ===================================

valid_contours = 0

for contour in contours:

    area = cv2.contourArea(contour)

    if area > 150 and area < 12000:

        valid_contours += 1

        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(

            image_color,

            (x, y),

            (x + w, y + h),

            (0, 0, 255),

            2
        )

        cv2.putText(

            image_color,

            "HOTSPOT",

            (x, y - 8),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.5,

            (0, 0, 255),

            2
        )

# ===================================
# HIGH-QUALITY HEATMAP COLORING
# ===================================

heatmap_colored = cv2.applyColorMap(

    np.uint8(255 * heatmap),

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
# INDUSTRY-STYLE OVERLAY
# ===================================

overlay = cv2.addWeighted(

    image_color,

    0.60,

    heatmap_colored,

    0.40,

    0
)

# ===================================
# PRINT ANALYTICS
# ===================================

print("\n===== ADVANCED HOTSPOT ANALYTICS =====")

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

    "Continuous Wafer Density Analytics",

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
    "Red    = Critical Hotspot",

    fontsize=10,

    bbox=dict(

        facecolor='white',

        alpha=0.85
    )
)

plt.axis('off')

plt.tight_layout()

plt.show()
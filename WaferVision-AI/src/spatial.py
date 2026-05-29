import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

# ===================================
# IMAGE INPUT
# ===================================

image_path = input(

    "\nEnter wafer image path: "
)

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
# STRICT DEFECT EXTRACTION
# ===================================

_, binary = cv2.threshold(

    image,

    210,

    255,

    cv2.THRESH_BINARY
)

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
# FIND DEFECT CONTOURS
# ===================================

contours, _ = cv2.findContours(

    binary,

    cv2.RETR_EXTERNAL,

    cv2.CHAIN_APPROX_SIMPLE
)

# ===================================
# OUTPUT IMAGE
# ===================================

output = cv2.cvtColor(

    image,

    cv2.COLOR_GRAY2BGR
)

# ===================================
# ANALYTICS VARIABLES
# ===================================

centroids = []

total_defect_area = 0

sector_counts = {

    "Center": 0,
    "Top": 0,
    "Bottom": 0,
    "Left": 0,
    "Right": 0,
    "Edge-Ring": 0
}

# ===================================
# GLOBAL CENTER
# ===================================

cx_global = 128
cy_global = 128

# ===================================
# PROCESS DEFECT REGIONS
# ===================================

for contour in contours:

    area = cv2.contourArea(contour)

    # IGNORE VERY SMALL REGIONS

    if area > 5 and area < 4000:

        total_defect_area += area

        M = cv2.moments(contour)

        if M["m00"] != 0:

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            centroids.append((cx, cy))

            # ===================================
            # DRAW CENTROID
            # ===================================

            cv2.circle(

                output,

                (cx, cy),

                3,

                (0, 0, 255),

                -1
            )

            # ===================================
            # DRAW DEFECT CONTOUR
            # ===================================

            cv2.drawContours(

                output,

                [contour],

                -1,

                (0, 255, 255),

                1
            )

            # ===================================
            # RADIAL DISTANCE
            # ===================================

            distance = np.sqrt(

                (cx - cx_global) ** 2 +
                (cy - cy_global) ** 2
            )

            # ===================================
            # SECTOR ANALYSIS
            # ===================================

            if distance < 40:

                sector_counts["Center"] += 1

            elif distance > 95:

                sector_counts["Edge-Ring"] += 1

            elif cy < 90:

                sector_counts["Top"] += 1

            elif cy > 166:

                sector_counts["Bottom"] += 1

            elif cx < 90:

                sector_counts["Left"] += 1

            elif cx > 166:

                sector_counts["Right"] += 1

# ===================================
# SPATIAL SPREAD ANALYSIS
# ===================================

if len(centroids) > 0:

    centroid_array = np.array(centroids)

    spread_x = np.std(centroid_array[:,0])

    spread_y = np.std(centroid_array[:,1])

    # ===================================
    # CLUSTER COMPACTNESS
    # ===================================

    if len(centroids) > 1:

        pairwise_distances = pdist(centroid_array)

        compactness = np.mean(pairwise_distances)

    else:

        compactness = 0

else:

    spread_x = 0
    spread_y = 0
    compactness = 0

# ===================================
# DEFECT DENSITY
# ===================================

wafer_area = np.pi * (120 ** 2)

defect_density = min(

    (total_defect_area / wafer_area) * 100,

    100
)

# ===================================
# RADIAL RISK SCORE
# ===================================

radial_risk_score = (

    sector_counts["Center"] * 2.0 +
    sector_counts["Edge-Ring"] * 1.8 +
    sector_counts["Top"] * 1.0 +
    sector_counts["Bottom"] * 1.0 +
    sector_counts["Left"] * 1.0 +
    sector_counts["Right"] * 1.0
)

# ===================================
# SPREAD SCORE
# ===================================

spread_score = (

    spread_x + spread_y
) / 2

# ===================================
# AUTOMATED WAFER RISK
# ===================================

if radial_risk_score > 120 or defect_density > 25:

    wafer_risk = "HIGH RISK"

elif radial_risk_score > 60 or defect_density > 12:

    wafer_risk = "MEDIUM RISK"

else:

    wafer_risk = "LOW RISK"

# ===================================
# DRAW RADIAL RINGS
# ===================================

cv2.circle(

    output,

    (128, 128),

    40,

    (255, 255, 255),

    1
)

cv2.circle(

    output,

    (128, 128),

    80,

    (255, 255, 255),

    1
)

cv2.circle(

    output,

    (128, 128),

    120,

    (255, 255, 0),

    2
)

# ===================================
# DRAW WAFER CENTER
# ===================================

cv2.circle(

    output,

    (128, 128),

    5,

    (255, 0, 0),

    -1
)

# ===================================
# PRINT ANALYTICS
# ===================================

print("\n========== INDUSTRIAL SPATIAL ANALYTICS ==========")

print(f"\nTotal Defect Regions     : {len(centroids)}")

print(f"Total Defect Area        : {total_defect_area:.2f}")

print(f"Defect Density           : {defect_density:.2f}%")

print(f"Spatial Spread X         : {spread_x:.2f}")

print(f"Spatial Spread Y         : {spread_y:.2f}")

print(f"Cluster Compactness      : {compactness:.2f}")

print(f"Radial Risk Score        : {radial_risk_score:.2f}")

print(f"Spread Score             : {spread_score:.2f}")

print(f"\nFinal Wafer Risk         : {wafer_risk}")

print("\n========== SECTOR DISTRIBUTION ==========")

for sector, count in sector_counts.items():

    print(f"{sector:<12}: {count}")

# ===================================
# DISPLAY OUTPUT
# ===================================

plt.figure(figsize=(9,9))

plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

plt.title(

    f"Industrial Spatial Analytics\n"
    f"Risk Level: {wafer_risk}",

    fontsize=14,

    fontweight='bold'
)

# ===================================
# LEGEND
# ===================================

plt.figtext(

    0.02,

    0.02,

    "Red Dots     = Defect Centroids\n"
    "Yellow Lines = Defect Boundaries\n"
    "Blue Dot     = Wafer Center\n"
    "White Rings  = Radial Zones\n"
    "Outer Ring   = Edge Risk Zone",

    fontsize=10,

    bbox=dict(

        facecolor='white',

        alpha=0.85
    )
)

plt.axis('off')

plt.tight_layout()

plt.show()
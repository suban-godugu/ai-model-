import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===================================
# IMAGE INPUT
# ===================================

image_path = input(

    "\nEnter wafer image path: "
)

# ===================================
# LOAD IMAGE
# ===================================

image = cv2.imread(image_path)

if image is None:

    print("\nERROR: Image not found.")
    exit()

# ===================================
# HIGH RESOLUTION
# ===================================

image = cv2.resize(

    image,

    (1024, 1024)
)

output = image.copy()

# ===================================
# CONVERT TO HSV
# ===================================

hsv = cv2.cvtColor(

    image,

    cv2.COLOR_BGR2HSV
)

# ===================================
# CREATE CIRCULAR WAFER MASK
# ===================================

mask = np.zeros(

    (1024, 1024),

    dtype=np.uint8
)

cv2.circle(

    mask,

    (512, 512),

    470,

    255,

    -1
)

# ===================================
# GRID SETTINGS
# ===================================

GRID_SIZE = 64

cell_h = 1024 // GRID_SIZE
cell_w = 1024 // GRID_SIZE

# ===================================
# ANALYTICS VARIABLES
# ===================================

total_dice = 0
good_dice = 0
defect_dice = 0

# ===================================
# LOOP THROUGH GRID CELLS
# ===================================

for row in range(GRID_SIZE):

    for col in range(GRID_SIZE):

        x1 = col * cell_w
        y1 = row * cell_h

        x2 = x1 + cell_w
        y2 = y1 + cell_h

        # ===================================
        # CELL CENTER
        # ===================================

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # ===================================
        # IGNORE OUTSIDE WAFER
        # ===================================

        if mask[center_y, center_x] == 0:

            continue

        # ===================================
        # EXTRACT CELL
        # ===================================

        cell = hsv[y1:y2, x1:x2]

        # ===================================
        # CELL BRIGHTNESS
        # ===================================

        mean_v = np.mean(cell[:,:,2])

        # ===================================
        # IGNORE EMPTY CELLS
        # ===================================

        if mean_v < 25:

            continue

        total_dice += 1

        # ===================================
        # YELLOW DEFECT DETECTION
        # ===================================

        yellow_mask = cv2.inRange(

            cell,

            (15, 40, 80),

            (45, 255, 255)
        )

        yellow_ratio = (

            np.sum(yellow_mask > 0)
            / yellow_mask.size
        )

        # ===================================
        # DEFECT CLASSIFICATION
        # ===================================

        is_defect = yellow_ratio > 0.18

        # ===================================
        # DRAW RESULTS
        # ===================================

        if is_defect:

            defect_dice += 1

            color = (255, 0, 0)

        else:

            good_dice += 1

            color = (0, 255, 0)

        # ===================================
        # DRAW DICE GRID
        # ===================================

        cv2.rectangle(

            output,

            (x1, y1),

            (x2, y2),

            color,

            1
        )

# ===================================
# YIELD CALCULATION
# ===================================

yield_percentage = (

    good_dice / total_dice
) * 100 if total_dice > 0 else 0

# ===================================
# APPLY WAFER MASK TO OUTPUT
# ===================================

output = cv2.bitwise_and(

    output,

    output,

    mask=mask
)

# ===================================
# DRAW WAFER BOUNDARY
# ===================================

cv2.circle(

    output,

    (512, 512),

    470,

    (255, 255, 255),

    2
)

# ===================================
# PRINT ANALYTICS
# ===================================

print("\n========== INDUSTRIAL DIE ANALYTICS ==========")

print(f"\nTotal Dice       : {total_dice}")

print(f"Good Dice        : {good_dice}")

print(f"Defect Dice      : {defect_dice}")

print(f"Yield Percentage : {yield_percentage:.2f}%")

# ===================================
# DISPLAY OUTPUT
# ===================================

plt.figure(figsize=(12,12))

plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

plt.title(

    f"Industry-Level Wafer Die Analytics\n"
    f"Total: {total_dice} | "
    f"Good: {good_dice} | "
    f"Defect: {defect_dice}",

    fontsize=15,

    fontweight='bold'
)

# ===================================
# LEGEND
# ===================================

plt.figtext(

    0.02,

    0.02,

    "GREEN = Good Dice\n"
    "BLUE  = Defective Dice\n"
    "Grid  = Individual Die Segmentation",

    fontsize=10,

    bbox=dict(

        facecolor='white',

        alpha=0.85
    )
)

plt.axis('off')

plt.tight_layout()

plt.show()
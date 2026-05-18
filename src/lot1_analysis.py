import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# INDUSTRIAL LOT1 WAFER ANALYTICS
# =========================================================

print(

    "\n========== LOT1 WAFER DATASET ANALYTICS =========="
)

# =========================================================
# DEFAULT DATASET PATH
# =========================================================

dataset_path = r"C:\Users\Keerthan\OneDrive\Desktop\wafer-spatial-ai\data\wafer_dataset\wafer_dataset\train"

# =========================================================
# CHECK DATASET
# =========================================================

if not os.path.exists(dataset_path):

    print("\nERROR: Dataset path not found.")
    exit()

# =========================================================
# LOT NAME
# =========================================================

lot_name = "LOT1"

# =========================================================
# DEFECT TYPES
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
    "Scratch"
]

# =========================================================
# STORAGE
# =========================================================

counts = {}

total_wafers = 0

# =========================================================
# COUNT ALL WAFERS
# =========================================================

for defect_type in classes:

    class_path = os.path.join(

        dataset_path,

        defect_type
    )

    # =====================================================
    # CHECK CLASS FOLDER
    # =====================================================

    if not os.path.exists(class_path):

        counts[defect_type] = 0
        continue

    # =====================================================
    # IMAGE FILES
    # =====================================================

    image_files = [

        f for f in os.listdir(class_path)

        if f.lower().endswith(

            ('.png', '.jpg', '.jpeg')
        )
    ]

    # =====================================================
    # COUNT
    # =====================================================

    count = len(image_files)

    counts[defect_type] = count

    total_wafers += count

# =========================================================
# PERCENTAGE CALCULATION
# =========================================================

percentages = {}

for defect_type in classes:

    count = counts[defect_type]

    if total_wafers > 0:

        percentages[defect_type] = (

            count / total_wafers
        ) * 100

    else:

        percentages[defect_type] = 0

# =========================================================
# PRINT RESULTS
# =========================================================

print(f"\nLot Name            : {lot_name}")

print(f"Dataset Path        : {dataset_path}")

print(f"\nTotal Wafers        : {total_wafers}")

print(

    "\n========== DEFECT DISTRIBUTION =========="
)

for defect_type in classes:

    print(

        f"{defect_type:12} : "
        f"{counts[defect_type]:6} wafers | "
        f"{percentages[defect_type]:6.2f}%"
    )

# =========================================================
# CREATE DATAFRAME
# =========================================================

report_df = pd.DataFrame({

    "Defect Type": classes,

    "Wafer Count": [

        counts[c] for c in classes
    ],

    "Percentage": [

        percentages[c] for c in classes
    ]
})

# =========================================================
# SAVE CSV REPORT
# =========================================================

csv_name = "LOT1_wafer_report.csv"

report_df.to_csv(

    csv_name,

    index=False
)

print(

    f"\nCSV Report Saved : {csv_name}"
)

# =========================================================
# BAR CHART
# =========================================================

plt.figure(figsize=(12,6))

bars = plt.bar(

    classes,

    [counts[c] for c in classes]
)

# =========================================================
# TITLE
# =========================================================

plt.title(

    "LOT1 Wafer Defect Distribution",

    fontsize=16,

    fontweight='bold'
)

# =========================================================
# AXIS LABELS
# =========================================================

plt.xlabel("Defect Type")

plt.ylabel("Number of Wafers")

# =========================================================
# VALUE LABELS
# =========================================================

for bar in bars:

    height = bar.get_height()

    plt.text(

        bar.get_x() + bar.get_width()/2,

        height + 5,

        f'{int(height)}',

        ha='center',

        fontsize=9
    )

# =========================================================
# GRID
# =========================================================

plt.grid(

    axis='y',

    linestyle='--',

    alpha=0.4
)

plt.tight_layout()

# =========================================================
# DISPLAY
# =========================================================

plt.show()
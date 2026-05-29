# =====================================================
# DASHBOARD LOT MANAGER
# FULLY AUTOMATIC AI + LOT ANALYSIS
# =====================================================

# FLOW:
#
# Upload Image
#       ↓
# AI Predicts Defect Class
#       ↓
# Auto Assign Lot
#       ↓
# Run Die Analysis
#       ↓
# Update Lot Statistics
#       ↓
# Display Yield
#
# =====================================================

# =====================================================
# IMPORTS
# =====================================================

import os

# ==========================================
# IMPORT DIE ANALYSIS ENGINE
# ==========================================

from dice_analysis import (
    load_rgb,
    analyze_wafer
)

# ==========================================
# IMPORT AI CLASSIFIER
# ==========================================

from predict import predict_image

# =====================================================
# LOT MAPPING
# =====================================================

LOT_MAPPING = {

    "Center": "LOT_1",

    "Donut": "LOT_2",

    "Edge-Loc": "LOT_3",

    "Edge-Ring": "LOT_4",

    "Scratch": "LOT_5",

    "Near-full": "LOT_6",

    "Random": "LOT_7",

    "Loc": "LOT_8",

    "Normal": "LOT_9"
}

# =====================================================
# LOT DATABASE
# =====================================================

lot_database = {}

# =====================================================
# GET LOT NAME
# =====================================================

def get_lot_name(defect_class):

    return LOT_MAPPING.get(
        defect_class,
        "UNKNOWN_LOT"
    )

# =====================================================
# INITIALIZE LOT
# =====================================================

def initialize_lot(lot_name):

    if lot_name not in lot_database:

        lot_database[lot_name] = {

            "wafer_count": 0,

            "good_dies": 0,

            "fail_dies": 0,

            "yield": 0.0,

            "wafers": []
        }

# =====================================================
# UPDATE LOT DATABASE
# =====================================================

def update_lot(
    lot_name,
    wafer_name,
    good,
    fail
):

    initialize_lot(lot_name)

    # ==========================================
    # UPDATE COUNTS
    # ==========================================

    lot_database[lot_name]["wafer_count"] += 1

    lot_database[lot_name]["good_dies"] += good

    lot_database[lot_name]["fail_dies"] += fail

    # ==========================================
    # CALCULATE YIELD
    # ==========================================

    total = (

        lot_database[lot_name]["good_dies"]

        +

        lot_database[lot_name]["fail_dies"]
    )

    if total > 0:

        lot_database[lot_name]["yield"] = (

            lot_database[lot_name]["good_dies"]

            / total

        ) * 100

    # ==========================================
    # STORE WAFER
    # ==========================================

    lot_database[lot_name]["wafers"].append({

        "wafer": wafer_name,

        "good": good,

        "fail": fail
    })

# =====================================================
# DISPLAY LOT SUMMARY
# =====================================================

def display_lot_summary(lot_name):

    lot = lot_database[lot_name]

    print("\n===================================")

    print(f"LOT SUMMARY : {lot_name}")

    print("===================================")

    print(
        f"\nTOTAL WAFERS : "
        f"{lot['wafer_count']}"
    )

    print(
        f"\nTOTAL GOOD DIES : "
        f"{lot['good_dies']}"
    )

    print(
        f"TOTAL FAIL DIES : "
        f"{lot['fail_dies']}"
    )

    total = (
        lot["good_dies"]
        +
        lot["fail_dies"]
    )

    print(
        f"\nTOTAL DIES : "
        f"{total}"
    )

    print(
        f"\nLOT YIELD : "
        f"{lot['yield']:.2f}%"
    )

# =====================================================
# PROCESS SINGLE WAFER
# =====================================================

def process_wafer(image_path):

    print("\n===================================")

    print("PROCESSING WAFER")

    print("===================================")

    print(f"\nImage Path : {image_path}")

    # ==========================================
    # AI PREDICTION
    # ==========================================

    predicted_class = predict_image(
        image_path
    )

    print(
        f"\nAI Predicted Class : "
        f"{predicted_class}"
    )

    # ==========================================
    # ASSIGN LOT
    # ==========================================

    lot_name = get_lot_name(
        predicted_class
    )

    print(
        f"Assigned Lot : "
        f"{lot_name}"
    )

    # ==========================================
    # LOAD IMAGE
    # ==========================================

    rgb = load_rgb(image_path)

    # ==========================================
    # RUN DIE ANALYSIS
    # ==========================================

    dies, meta = analyze_wafer(rgb)

    # ==========================================
    # EXTRACT RESULTS
    # ==========================================

    good = meta["good"]

    fail = meta["fail"]

    total = good + fail

    wafer_yield = (

        good / total * 100

    ) if total else 0

    # ==========================================
    # UPDATE LOT DATABASE
    # ==========================================

    wafer_name = os.path.basename(
        image_path
    )

    update_lot(

        lot_name,

        wafer_name,

        good,

        fail
    )

    # ==========================================
    # DISPLAY WAFER RESULT
    # ==========================================

    print("\n========== WAFER RESULT ==========")

    print(f"\nGOOD DIES : {good}")

    print(f"FAIL DIES : {fail}")

    print(f"TOTAL DIES : {total}")

    print(
        f"\nWAFER YIELD : "
        f"{wafer_yield:.2f}%"
    )

    # ==========================================
    # DISPLAY LOT SUMMARY
    # ==========================================

    display_lot_summary(
        lot_name
    )

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    print(
        "\n========== AI DASHBOARD LOT MANAGER ==========\n"
    )

    # ==========================================
    # ENTER IMAGE PATH
    # ==========================================

    image_path = input(

        "Enter wafer image path:\n\nPath: "

    ).strip().strip('"')

    # ==========================================
    # CHECK IMAGE
    # ==========================================

    if not os.path.exists(image_path):

        print("\nERROR: Image not found")

        exit()

    # ==========================================
    # PROCESS WAFER
    # ==========================================

    process_wafer(image_path)

    print("\n===================================")

    print("PROCESS COMPLETED")

    print("===================================")
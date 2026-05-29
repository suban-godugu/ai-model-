import os

# =====================================================
# IMPORT SINGLE WAFER ENGINE
# =====================================================

from dice_analysis import (
    load_rgb,
    analyze_wafer
)

# =====================================================
# LOT NAME MAPPING
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
# LOT ANALYSIS
# =====================================================

if __name__ == "__main__":

    print("\n========== LOT ANALYSIS ==========\n")

    lot_path = input(
        "Enter LOT folder path:\n\nPath: "
    ).strip().strip('"')

    # ==========================================
    # CHECK PATH
    # ==========================================

    if not os.path.exists(lot_path):

        print("\nERROR: Invalid folder path")

        exit()

    # ==========================================
    # GET LOT NAME
    # ==========================================

    folder_name = os.path.basename(lot_path)

    lot_name = LOT_MAPPING.get(
        folder_name,
        "UNKNOWN_LOT"
    )

    print(f"\nDetected Lot: {lot_name}")

    print(f"Defect Type : {folder_name}")

    # ==========================================
    # FIND ALL IMAGES
    # ==========================================

    image_files = []

    for file in os.listdir(lot_path):

        if file.lower().endswith(
            (
                ".png",
                ".jpg",
                ".jpeg",
                ".bmp"
            )
        ):

            image_files.append(file)

    # ==========================================
    # NO IMAGES
    # ==========================================

    if len(image_files) == 0:

        print("\nNo wafer images found")

        exit()

    # ==========================================
    # LOT TOTALS
    # ==========================================

    lot_good = 0

    lot_fail = 0

    wafer_results = []

    # ==========================================
    # PROCESS EACH WAFER
    # ==========================================

    for idx, file in enumerate(image_files):

        try:

            full_path = os.path.join(
                lot_path,
                file
            )

            print(
                f"\n[{idx+1}/{len(image_files)}] "
                f"Processing: {file}"
            )

            # ==================================
            # LOAD IMAGE
            # ==================================

            rgb = load_rgb(full_path)

            # ==================================
            # RUN DIE ANALYSIS
            # ==================================

            dies, meta = analyze_wafer(rgb)

            good = meta["good"]

            fail = meta["fail"]

            total = good + fail

            wafer_yield = (
                good / total * 100
            ) if total else 0

            # ==================================
            # UPDATE LOT TOTALS
            # ==================================

            lot_good += good

            lot_fail += fail

            # ==================================
            # STORE WAFER RESULT
            # ==================================

            wafer_results.append({

                "wafer": file,

                "good": good,

                "fail": fail,

                "yield": wafer_yield
            })

            print(
                f"GOOD={good} | "
                f"FAIL={fail} | "
                f"YIELD={wafer_yield:.2f}%"
            )

        except Exception as e:

            print(f"\nERROR PROCESSING: {file}")

            print(e)

    # ==========================================
    # FINAL LOT RESULTS
    # ==========================================

    total_dies = lot_good + lot_fail

    lot_yield = (
        lot_good / total_dies * 100
    ) if total_dies else 0

    print("\n===================================")

    print("FINAL LOT RESULTS")

    print("===================================")

    print(f"\nLOT NAME : {lot_name}")

    print(f"DEFECT TYPE : {folder_name}")

    print(f"\nTOTAL WAFERS : {len(wafer_results)}")

    print(f"\nTOTAL GOOD DIES : {lot_good}")

    print(f"TOTAL FAIL DIES : {lot_fail}")

    print(f"\nTOTAL DIES : {total_dies}")

    print(f"\nLOT YIELD : {lot_yield:.2f}%")

    # ==========================================
    # BEST WAFER
    # ==========================================

    best = max(
        wafer_results,
        key=lambda x: x["yield"]
    )

    # ==========================================
    # WORST WAFER
    # ==========================================

    worst = min(
        wafer_results,
        key=lambda x: x["yield"]
    )

    print("\n========== BEST WAFER ==========\n")

    print(best)

    print("\n========== WORST WAFER ==========\n")

    print(worst)
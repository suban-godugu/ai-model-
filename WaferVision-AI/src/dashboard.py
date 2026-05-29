# REVISED DASHBOARD.PY
# =========================================================
# SEMICONDUCTOR WAFER ANALYTICS DASHBOARD
# INDUSTRIAL SIDE-BY-SIDE VISUALIZATION VERSION
# =========================================================

import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import requests
import tempfile

from dice_analysis import analyze_wafer

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Wafer Spatial AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #0E1117;
}

.block-container {
    padding-top: 1rem;
    padding-left: 1.5rem;
    padding-right: 1.5rem;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
}

h1, h2, h3 {
    color: white;
}

.metric-card {
    background: linear-gradient(145deg, #182234, #101826);
    border: 1px solid #26354d;
    border-radius: 14px;
    padding: 12px;
    margin-bottom: 12px;
}

.metric-title {
    color: #8ea3c7;
    font-size: 14px;
    margin-bottom: 4px;
}

.metric-value {
    color: white;
    font-size: 28px;
    font-weight: 700;
}

.overlay-title {
    text-align: center;
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 10px;
    color: white;
}

.small-sub {
    text-align: center;
    color: #9ca3af;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================

st.markdown("""
# SEMICONDUCTOR WAFER ANALYTICS DASHBOARD
### AI Defect Classification • Overlay Heatmap • Fail Density • LOT Intelligence
""")

# =========================================================
# CONFIG
# =========================================================

IMG_SIZE = 224

WAFER_CX = IMG_SIZE // 2
WAFER_CY = IMG_SIZE // 2
WAFER_R = 102

API_URL = "http://127.0.0.1:8000/predict"

LOT_MAPPING = {
    "Center": "LOT_1",
    "Donut": "LOT_2",
    "Edge-Loc": "LOT_3",
    "Edge-Ring": "LOT_4",
    "Scratch": "LOT_5",
    "Near-Full": "LOT_6",
    "Random": "LOT_7",
    "Local": "LOT_8",
    "Normal": "LOT_9",
}

# =========================================================
# WAFER MASK
# =========================================================

Y, X = np.ogrid[:IMG_SIZE, :IMG_SIZE]

_dist = np.sqrt(
    (X - WAFER_CX) ** 2 +
    (Y - WAFER_CY) ** 2
)

WAFER_MASK = (_dist <= WAFER_R).astype(np.uint8)

# =========================================================
# APPLY MASK
# =========================================================


def apply_mask_rgb(rgb):

    out = rgb.copy()

    out[WAFER_MASK == 0] = 0

    return out

# =========================================================
# API PREDICTION
# =========================================================


def predict_via_api(image_array):

    try:

        with tempfile.NamedTemporaryFile(
            suffix=".png",
            delete=False
        ) as tmp:

            temp_path = tmp.name

        Image.fromarray(image_array).save(temp_path)

        with open(temp_path, "rb") as f:

            response = requests.post(
                API_URL,
                files={"file": f},
                timeout=60
            )

        if response.status_code != 200:
            raise Exception(f"API Error {response.status_code}")

        result = response.json()

        predicted_class = result["class"]
        confidence = result["confidence"]

        return predicted_class, confidence

    except Exception as e:

        st.error(f"FastAPI prediction failed: {e}")

        return "Unknown", 0.0

# =========================================================
# INDUSTRIAL OVERLAY HEATMAP
# =========================================================


def build_overlay(masked_rgb, fail_map, pitch, offset, blur_ksize=3):

    heat = np.nan_to_num(
        fail_map,
        nan=0.0
    ).astype(np.float32)

    heat = cv2.resize(
        heat,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_NEAREST
    )

    if blur_ksize > 0:

        k = blur_ksize

        if k % 2 == 0:
            k += 1

        heat = cv2.GaussianBlur(
            heat,
            (k, k),
            0
        )

    heat = np.clip(heat, 0, 1)

    heat_u8 = (heat * 255).astype(np.uint8)

    heat_color = cv2.applyColorMap(
        heat_u8,
        cv2.COLORMAP_INFERNO
    )

    heat_color = cv2.cvtColor(
        heat_color,
        cv2.COLOR_BGR2RGB
    )

    overlay = masked_rgb.copy()

    fail_mask = heat > 0.25

    overlay[fail_mask] = (
        0.82 * overlay[fail_mask] +
        0.18 * heat_color[fail_mask]
    ).astype(np.uint8)

    edges = cv2.Canny(
        heat_u8,
        80,
        150
    )

    overlay[edges > 0] = [180, 220, 255]

    cv2.circle(
        overlay,
        (WAFER_CX, WAFER_CY),
        WAFER_R,
        (255, 255, 255),
        2
    )

    ox, oy = offset

    pitch = max(1, int(pitch))

    for x in range(int(ox), IMG_SIZE, pitch):

        overlay[:, x] = (
            overlay[:, x] * 0.97
        ).astype(np.uint8)

    for y in range(int(oy), IMG_SIZE, pitch):

        overlay[y, :] = (
            overlay[y, :] * 0.97
        ).astype(np.uint8)

    return overlay

# =========================================================
# FAIL DENSITY MAP
# =========================================================


def build_density_map(fail_map):

    density = np.nan_to_num(
        fail_map,
        nan=0.0
    ).astype(np.float32)

    density = cv2.resize(
        density,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_CUBIC
    )

    density = cv2.GaussianBlur(
        density,
        (25, 25),
        0
    )

    density = np.clip(density, 0, 1)

    density_u8 = (density * 255).astype(np.uint8)

    density_color = cv2.applyColorMap(
        density_u8,
        cv2.COLORMAP_JET
    )

    density_color = cv2.cvtColor(
        density_color,
        cv2.COLOR_BGR2RGB
    )

    density_color[WAFER_MASK == 0] = 0

    cv2.circle(
        density_color,
        (WAFER_CX, WAFER_CY),
        WAFER_R,
        (255, 255, 255),
        2
    )

    return density_color

# =========================================================
# AI ATTENTION MAP
# =========================================================


def build_attention_map(masked_rgb):

    gray = cv2.cvtColor(
        masked_rgb,
        cv2.COLOR_RGB2GRAY
    )

    attention = cv2.GaussianBlur(
        gray,
        (31, 31),
        0
    )

    attention = cv2.normalize(
        attention,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    attention_color = cv2.applyColorMap(
        attention.astype(np.uint8),
        cv2.COLORMAP_MAGMA
    )

    attention_color = cv2.cvtColor(
        attention_color,
        cv2.COLOR_BGR2RGB
    )

    result = cv2.addWeighted(
        masked_rgb,
        0.65,
        attention_color,
        0.35,
        0
    )

    cv2.circle(
        result,
        (WAFER_CX, WAFER_CY),
        WAFER_R,
        (255, 255, 255),
        2
    )

    return result

# =========================================================
# PLOTLY FIGURE
# =========================================================


def plotly_figure(image, title, subtitle=""):

    fig = go.Figure()

    fig.add_trace(go.Image(z=image))

    fig.update_layout(

        template="plotly_dark",

        height=620,

        margin=dict(
            l=5,
            r=5,
            t=60,
            b=5
        ),

        paper_bgcolor="#0E1117",

        plot_bgcolor="#0E1117",

        title=dict(
            text=f"<b>{title}</b><br><sup>{subtitle}</sup>",
            x=0.5,
            xanchor="center",
        )
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    return fig

# =========================================================
# SESSION STATE
# =========================================================

if "lot_database" not in st.session_state:

    st.session_state.lot_database = {
        "LOT_1": {"defect_type": "Center", "wafers": []},
        "LOT_2": {"defect_type": "Donut", "wafers": []},
        "LOT_3": {"defect_type": "Edge-Loc", "wafers": []},
        "LOT_4": {"defect_type": "Edge-Ring", "wafers": []},
        "LOT_5": {"defect_type": "Scratch", "wafers": []},
        "LOT_6": {"defect_type": "Near-Full", "wafers": []},
        "LOT_7": {"defect_type": "Random", "wafers": []},
        "LOT_8": {"defect_type": "Local", "wafers": []},
        "LOT_9": {"defect_type": "Normal", "wafers": []},
    }

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Visualization")

blur_slider = st.sidebar.slider(
    "Heatmap smoothness",
    0,
    15,
    3,
    1
)

show_attention = st.sidebar.toggle(
    "Show AI attention map",
    False
)

uploaded_files = st.file_uploader(
    "Upload Wafer Images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

# =========================================================
# PROCESS IMAGES
# =========================================================

if uploaded_files:

    for uploaded_file in uploaded_files:

        try:

            pil_img = Image.open(uploaded_file).convert("RGB")

            rgb = np.array(pil_img)

            rgb = cv2.resize(
                rgb,
                (IMG_SIZE, IMG_SIZE)
            )

            masked_rgb = apply_mask_rgb(rgb)

            predicted_class, confidence = predict_via_api(masked_rgb)

            assigned_lot = LOT_MAPPING.get(
                predicted_class,
                "LOT_UNKNOWN"
            )

            result = analyze_wafer(masked_rgb)

            overlay = build_overlay(
                masked_rgb,
                result["map"],
                result["pitch"],
                result["offset"],
                blur_slider,
            )

            density_map = build_density_map(result["map"])

            attention_map = build_attention_map(masked_rgb)

            wafer_data = {
                "name": uploaded_file.name,
                "overlay": overlay,
                "density": density_map,
                "attention": attention_map,
                "class": predicted_class,
                "confidence": confidence,
                "lot": assigned_lot,
                "good": result["good"],
                "fail": result["fail"],
                "total": result["total"],
                "yield": result["yield"],
            }

            existing = [
                w["name"]
                for w in st.session_state.lot_database[
                    assigned_lot
                ]["wafers"]
            ]

            if uploaded_file.name not in existing:

                st.session_state.lot_database[
                    assigned_lot
                ]["wafers"].append(wafer_data)

        except Exception as e:

            st.error(
                f"Error processing {uploaded_file.name}: {e}"
            )

# =========================================================
# LOT SELECTION
# =========================================================

selected_lot = st.sidebar.selectbox(
    "Select LOT",
    list(st.session_state.lot_database.keys())
)

st.sidebar.markdown("---")
st.sidebar.subheader("LOT STATUS")

for lot_name, lot_data in st.session_state.lot_database.items():

    st.sidebar.write(
        f"{lot_name} ({lot_data['defect_type']}) : "
        f"{len(lot_data['wafers'])} wafers"
    )

lot_info = st.session_state.lot_database[selected_lot]
wafers = lot_info["wafers"]

# =========================================================
# DISPLAY
# =========================================================

st.header(f"{selected_lot} ANALYSIS")

st.caption(
    f"Defect Type: {lot_info['defect_type']}"
)

if len(wafers) == 0:

    st.warning("No wafers in this LOT")

else:

    for idx, wafer in enumerate(wafers):

        st.markdown("---")

        st.subheader(wafer["name"])

        main_col, side_col = st.columns([5, 2])

        with main_col:

            viz1, viz2 = st.columns(2)

            with viz1:

                st.markdown(
                    "<div class='overlay-title'>Overlay Analytics</div>",
                    unsafe_allow_html=True
                )

                fig = plotly_figure(
                    wafer["overlay"],
                    wafer["name"],
                    f"{wafer['class']} ({wafer['confidence']:.2f}%)"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key=f"overlay_{selected_lot}_{idx}"
                )

            with viz2:

                st.markdown(
                    "<div class='overlay-title'>Fail Density Map</div>",
                    unsafe_allow_html=True
                )

                density_fig = plotly_figure(
                    wafer["density"],
                    "Defect Density Distribution",
                    "Thermal-style fail concentration visualization"
                )

                st.plotly_chart(
                    density_fig,
                    use_container_width=True,
                    key=f"density_{selected_lot}_{idx}"
                )

            if show_attention:

                st.subheader("AI Attention Map")

                attention_fig = plotly_figure(
                    wafer["attention"],
                    "AI Visual Attention",
                    "CNN attention-style visualization"
                )

                st.plotly_chart(
                    attention_fig,
                    use_container_width=True,
                    key=f"attention_{selected_lot}_{idx}"
                )

        with side_col:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Defect Type</div>
                    <div class="metric-value">{wafer['class']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Assigned LOT</div>
                    <div class="metric-value">{wafer['lot']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.metric(
                "Confidence",
                f"{wafer['confidence']:.2f}%"
            )

            st.metric("Good Dies", wafer["good"])
            st.metric("Fail Dies", wafer["fail"])
            st.metric("Total Dies", wafer["total"])
            st.metric("Yield", f"{wafer['yield']:.2f}%")

# =========================================================
# LOT SUMMARY
# =========================================================

if len(wafers) > 0:

    total_good = sum(w["good"] for w in wafers)
    total_fail = sum(w["fail"] for w in wafers)
    total_dies = sum(w["total"] for w in wafers)

    lot_yield = (
        total_good / total_dies
    ) * 100.0

    st.markdown("---")

    st.subheader(f"{selected_lot} SUMMARY")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Wafers", len(wafers))
    c2.metric("Good Dies", total_good)
    c3.metric("Fail Dies", total_fail)
    c4.metric("LOT Yield", f"{lot_yield:.2f}%")
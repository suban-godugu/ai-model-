import streamlit as st
import cv2
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from PIL import Image
from sklearn.cluster import DBSCAN

from pytorch_grad_cam import GradCAMPlusPlus

from preprocess import test_transform
from model import get_model

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(

    page_title="WaferVision AI",

    layout="wide",

    initial_sidebar_state="expanded"
)

# =========================================================
# DARK UI STYLE
# =========================================================

st.markdown("""

<style>

.stApp {

    background-color: #0a0f1f;
    color: white;
}

[data-testid="stSidebar"] {

    background-color: #111827;
}

.metric-box {

    background: #151c2c;

    padding: 18px;

    border-radius: 15px;

    border: 1px solid #24314d;

    box-shadow: 0px 0px 12px rgba(0,255,255,0.06);
}

</style>

""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================

st.title("WaferVision AI Control Center")

st.markdown(
    "Industry-Level Semiconductor Spatial Intelligence Platform"
)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Control Panel")

overlay_type = st.sidebar.selectbox(

    "Overlay Mode",

    [

        "Grad-CAM++",
        "Cost Heatmap",
        "Fail Density",
        "Critical Hotspots"
    ]
)

show_grid = st.sidebar.toggle(

    "Show Die Grid",

    value=True
)

show_clusters = st.sidebar.toggle(

    "Show Defect Clusters",

    value=True
)

show_3d = st.sidebar.toggle(

    "3D Surface",

    value=False
)

opacity = st.sidebar.slider(

    "Overlay Opacity",

    0.1,

    1.0,

    0.60
)

# =========================================================
# CLASSES
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
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = get_model(len(classes))

    model.load_state_dict(

        torch.load(

            "wafer_model.pth",

            map_location=torch.device("cpu")
        )
    )

    model.eval()

    return model

model = load_model()

# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_files = st.file_uploader(

    "Upload Wafer Images",

    type=["png", "jpg", "jpeg"],

    accept_multiple_files=True
)

# =========================================================
# PROCESS FILES
# =========================================================

if uploaded_files:

    lot_summary = []

    total_hotspots = 0

    total_risk = 0

    for uploaded_file in uploaded_files:

        st.markdown("---")

        st.subheader(f"Wafer Analysis: {uploaded_file.name}")

        # =================================================
        # LOAD IMAGE
        # =================================================

        image = Image.open(uploaded_file).convert("RGB")

        image_np = np.array(image)

        image_np = cv2.resize(

            image_np,

            (256,256)
        )

        # =================================================
        # WAFER MASK
        # =================================================

        mask = np.zeros((256,256), dtype=np.uint8)

        cv2.circle(

            mask,

            (128,128),

            118,

            255,

            -1
        )

        image_np = cv2.bitwise_and(

            image_np,

            image_np,

            mask=mask
        )

        # =================================================
        # TRANSFORM
        # =================================================

        input_tensor = test_transform(

            Image.fromarray(image_np)

        ).unsqueeze(0)

        # =================================================
        # PREDICTION
        # =================================================

        with torch.no_grad():

            output = model(input_tensor)

            probabilities = torch.nn.functional.softmax(

                output,

                dim=1
            )

            confidence, predicted = torch.max(

                probabilities,

                1
            )

        predicted_class = classes[predicted.item()]

        confidence_score = confidence.item() * 100

        # =================================================
        # RISK
        # =================================================

        if predicted_class == "Normal":

            risk = "LOW"

            risk_score = 25

        elif confidence_score > 95:

            risk = "HIGH"

            risk_score = 90

        else:

            risk = "MEDIUM"

            risk_score = 60

        total_risk += risk_score

        # =================================================
        # GRAD CAM
        # =================================================

        target_layers = [model.layer4[-1]]

        cam = GradCAMPlusPlus(

            model=model,

            target_layers=target_layers
        )

        grayscale_cam = cam(

            input_tensor=input_tensor

        )[0]

        grayscale_cam = cv2.resize(

            grayscale_cam,

            (256,256),

            interpolation=cv2.INTER_CUBIC
        )

        grayscale_cam = cv2.GaussianBlur(

            grayscale_cam,

            (11,11),

            0
        )

        grayscale_cam = (

            grayscale_cam -

            grayscale_cam.min()

        ) / (

            grayscale_cam.max()

            -

            grayscale_cam.min()

            +

            1e-8
        )

        grayscale_cam = grayscale_cam * (

            mask / 255.0
        )

        # =================================================
        # OVERLAY TYPES
        # =================================================

        if overlay_type == "Cost Heatmap":

            display_map = grayscale_cam * 100

        elif overlay_type == "Fail Density":

            display_map = grayscale_cam * 1.5

        elif overlay_type == "Critical Hotspots":

            display_map = np.where(

                grayscale_cam > 0.6,

                grayscale_cam,

                0
            )

        else:

            display_map = grayscale_cam

        # =================================================
        # CLUSTER DETECTION
        # =================================================

        hotspot_points = np.column_stack(

            np.where(grayscale_cam > 0.70)
        )

        total_hotspots += len(hotspot_points)

        cluster_x = []
        cluster_y = []

        if len(hotspot_points) > 10:

            clustering = DBSCAN(

                eps=6,

                min_samples=5

            ).fit(hotspot_points)

            labels = clustering.labels_

            unique_labels = set(labels)

            for label in unique_labels:

                if label == -1:

                    continue

                cluster = hotspot_points[

                    labels == label
                ]

                centroid = np.mean(

                    cluster,

                    axis=0
                )

                cluster_y.append(centroid[0])

                cluster_x.append(centroid[1])

        # =================================================
        # PLOTLY FIGURE
        # =================================================

        if show_3d:

            fig = go.Figure(

                data=[go.Surface(

                    z=display_map,

                    colorscale='Turbo'
                )]
            )

            fig.update_layout(

                template="plotly_dark",

                height=700
            )

        else:

            fig = go.Figure()

            # =============================================
            # ORIGINAL WAFER
            # =============================================

            fig.add_trace(

                go.Image(

                    z=image_np
                )
            )

            # =============================================
            # HEATMAP
            # =============================================

            fig.add_trace(

                go.Heatmap(

                    z=display_map,

                    colorscale='Turbo',

                    opacity=opacity,

                    showscale=True
                )
            )

            # =============================================
            # DIE GRID
            # =============================================

            if show_grid:

                grid_size = 8

                for x in range(0,256,grid_size):

                    fig.add_shape(

                        type="line",

                        x0=x,

                        y0=0,

                        x1=x,

                        y1=256,

                        line=dict(

                            color="rgba(255,255,255,0.15)",

                            width=1
                        )
                    )

                for y in range(0,256,grid_size):

                    fig.add_shape(

                        type="line",

                        x0=0,

                        y0=y,

                        x1=256,

                        y1=y,

                        line=dict(

                            color="rgba(255,255,255,0.15)",

                            width=1
                        )
                    )

            # =============================================
            # DEFECT CLUSTERS
            # =============================================

            if show_clusters and len(cluster_x) > 0:

                fig.add_trace(

                    go.Scatter(

                        x=cluster_x,

                        y=cluster_y,

                        mode='markers',

                        marker=dict(

                            size=14,

                            color='red',

                            symbol='x'
                        ),

                        name="Critical Cluster"
                    )
                )

            # =============================================
            # WAFER BORDER
            # =============================================

            theta = np.linspace(

                0,

                2*np.pi,

                400
            )

            x = 128 + 118*np.cos(theta)

            y = 128 + 118*np.sin(theta)

            fig.add_trace(

                go.Scatter(

                    x=x,

                    y=y,

                    mode='lines',

                    line=dict(

                        color='white',

                        width=3
                    ),

                    showlegend=False
                )
            )

            fig.update_layout(

                template="plotly_dark",

                height=750,

                xaxis=dict(

                    visible=False
                ),

                yaxis=dict(

                    visible=False,

                    scaleanchor="x"
                )
            )

        # =================================================
        # DISPLAY
        # =================================================

        col1, col2 = st.columns([3,1])

        with col1:

            st.plotly_chart(

                fig,

                use_container_width=True
            )

        with col2:

            st.markdown("## Wafer Metrics")

            st.metric(

                "Predicted Class",

                predicted_class
            )

            st.metric(

                "Confidence",

                f"{confidence_score:.2f}%"
            )

            st.metric(

                "Risk Level",

                risk
            )

            hotspot_pixels = int(

                np.sum(grayscale_cam > 0.70)
            )

            st.metric(

                "Hotspot Pixels",

                hotspot_pixels
            )

            attention_score = float(

                np.mean(grayscale_cam) * 100
            )

            st.metric(

                "Attention Score",

                f"{attention_score:.2f}%"
            )

            st.metric(

                "Critical Clusters",

                len(cluster_x)
            )

            yield_score = 100 - attention_score

            st.metric(

                "Estimated Yield",

                f"{yield_score:.2f}%"
            )

        # =================================================
        # STORE SUMMARY
        # =================================================

        lot_summary.append({

            "Wafer": uploaded_file.name,

            "Prediction": predicted_class,

            "Confidence": f"{confidence_score:.2f}%",

            "Risk": risk,

            "Clusters": len(cluster_x)
        })

    # =====================================================
    # LOT SUMMARY
    # =====================================================

    st.markdown("---")

    st.header("LOT-Level Semiconductor Analytics")

    avg_risk = total_risk / len(uploaded_files)

    summary_df = pd.DataFrame(lot_summary)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    kpi1.metric(

        "Total Wafers",

        len(uploaded_files)
    )

    kpi2.metric(

        "Total Hotspots",

        total_hotspots
    )

    kpi3.metric(

        "Average Risk Score",

        f"{avg_risk:.2f}"
    )

    kpi4.metric(

        "LOT Yield",

        f"{100 - avg_risk:.2f}%"
    )

    st.dataframe(

        summary_df,

        width='stretch'
    )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(

    "WaferVision AI | Spatial Semiconductor Intelligence System"
)
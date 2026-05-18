import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image

from preprocess import test_transform
from model import get_model

# ===================================
# CLASSES
# ===================================

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

# ===================================
# LOAD MODEL
# ===================================

model = get_model(len(classes))

model.load_state_dict(

    torch.load(

        "wafer_model.pth",

        map_location=torch.device("cpu")
    )
)

model.eval()

# ===================================
# IMAGE INPUT
# ===================================

image_path = input(

    "\nEnter image path: "
)

# ===================================
# LOAD ORIGINAL IMAGE
# ===================================

original_image = cv2.imread(image_path)

# ===================================
# RESIZE IMAGE
# ===================================

original_image = cv2.resize(

    original_image,

    (224, 224)
)

# ===================================
# CREATE CIRCULAR WAFER MASK
# ===================================

mask = np.zeros(

    (224, 224),

    dtype=np.uint8
)

cv2.circle(

    mask,

    (112, 112),

    105,

    255,

    -1
)

# ===================================
# APPLY MASK TO IMAGE
# ===================================

original_image = cv2.bitwise_and(

    original_image,

    original_image,

    mask=mask
)

# ===================================
# CONVERT BGR → RGB
# ===================================

original_image_rgb = cv2.cvtColor(

    original_image,

    cv2.COLOR_BGR2RGB
)

# ===================================
# LOAD PIL IMAGE
# ===================================

image = Image.open(image_path).convert("RGB")

# ===================================
# APPLY TRANSFORM
# ===================================

input_tensor = test_transform(image).unsqueeze(0)

# ===================================
# MODEL PREDICTION
# ===================================

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

print(

    f"\nPredicted Class: {predicted_class}"
)

print(

    f"Confidence Score: {confidence_score:.2f}%"
)

# ===================================
# TARGET LAYER
# ===================================

target_layers = [

    model.layer4[-1]
]

# ===================================
# INITIALIZE GRAD-CAM++
# ===================================

cam = GradCAMPlusPlus(

    model=model,

    target_layers=target_layers
)

# ===================================
# GENERATE GRAD-CAM
# ===================================

grayscale_cam = cam(

    input_tensor=input_tensor
)

grayscale_cam = grayscale_cam[0]

# ===================================
# RESIZE CAM
# ===================================

grayscale_cam = cv2.resize(

    grayscale_cam,

    (224, 224),

    interpolation=cv2.INTER_CUBIC
)

# ===================================
# SMOOTH HEATMAP
# ===================================

grayscale_cam = cv2.GaussianBlur(

    grayscale_cam,

    (11, 11),

    0
)

# ===================================
# NORMALIZE HEATMAP
# ===================================

grayscale_cam = (

    grayscale_cam - grayscale_cam.min()

) / (

    grayscale_cam.max() - grayscale_cam.min() + 1e-8
)

# ===================================
# CONTRAST BOOST
# ===================================

grayscale_cam = np.power(

    grayscale_cam,

    0.7
)

# ===================================
# REMOVE LOW ATTENTION NOISE
# ===================================

threshold = np.percentile(

    grayscale_cam,

    60
)

grayscale_cam[

    grayscale_cam < threshold

] *= 0.25

# ===================================
# APPLY CIRCULAR MASK TO CAM
# ===================================

mask_float = mask.astype(np.float32) / 255.0

grayscale_cam = grayscale_cam * mask_float

# ===================================
# NORMALIZE RGB IMAGE
# ===================================

rgb_image = original_image_rgb.astype(

    np.float32

) / 255.0

# ===================================
# CREATE VISUALIZATION
# ===================================

visualization = show_cam_on_image(

    rgb_image,

    grayscale_cam,

    use_rgb=True,

    image_weight=0.55
)

# ===================================
# APPLY MASK TO FINAL OUTPUT
# ===================================

visualization = cv2.bitwise_and(

    visualization,

    visualization,

    mask=mask
)

# ===================================
# ADD WAFER BORDER
# ===================================

cv2.circle(

    visualization,

    (112, 112),

    105,

    (255, 255, 255),

    2
)

# ===================================
# SHARPEN OUTPUT
# ===================================

kernel = np.array([

    [-1, -1, -1],
    [-1,  9, -1],
    [-1, -1, -1]

])

visualization = cv2.filter2D(

    visualization,

    -1,

    kernel
)

# ===================================
# DISPLAY OUTPUT
# ===================================

plt.figure(figsize=(10,10))

plt.imshow(visualization)

plt.title(

    f"Industry-Level Optimized Grad-CAM++\n"
    f"Class: {predicted_class} | "
    f"Confidence: {confidence_score:.2f}%",

    fontsize=14,

    fontweight='bold'
)

# ===================================
# LEGEND
# ===================================

plt.figtext(

    0.02,

    0.02,

    "RED / YELLOW : Strong AI Attention\n"
    "GREEN         : Moderate Attention\n"
    "BLUE          : Weak Attention",

    fontsize=10,

    bbox=dict(

        facecolor='white',

        alpha=0.85
    )
)

# ===================================
# REMOVE AXES
# ===================================

plt.axis('off')

plt.tight_layout()

# ===================================
# SHOW OUTPUT
# ===================================

plt.show()
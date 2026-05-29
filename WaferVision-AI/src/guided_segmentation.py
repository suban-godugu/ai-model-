import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

from torchvision import models
from torchvision import transforms

# ==========================================
# DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing Device: {device}")

# ==========================================
# LOAD RESNET50 CLASSIFIER
# ==========================================

classifier = models.resnet50()

classifier.fc = torch.nn.Linear(
    classifier.fc.in_features,
    9
)

classifier.load_state_dict(
    torch.load(
        "wafer_model.pth",
        map_location=device
    )
)

classifier = classifier.to(device)
classifier.eval()

# ==========================================
# LOAD U-NET SEGMENTER
# ==========================================

segmenter = smp.Unet(

    encoder_name="resnet34",

    encoder_weights=None,

    in_channels=3,

    classes=1
)

segmenter.load_state_dict(
    torch.load(
        "models/unet_wafer.pth",
        map_location=device
    )
)

segmenter = segmenter.to(device)
segmenter.eval()

print("\nModels Loaded Successfully.")

# ==========================================
# CLASS NAMES
# ==========================================

class_names = [

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

# ==========================================
# INPUT IMAGE
# ==========================================

image_path = input(
    "\nEnter wafer image path: "
)

# ==========================================
# LOAD IMAGE
# ==========================================

image = cv2.imread(image_path)

if image is None:
    raise ValueError("\nInvalid image path.")

image = cv2.resize(
    image,
    (224,224)
)

image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

# ==========================================
# CREATE WAFER MASK
# ==========================================

wafer_mask = np.zeros(
    (224,224),
    dtype=np.uint8
)

cv2.circle(
    wafer_mask,
    (112,112),
    104,
    255,
    -1
)

# ==========================================
# PREPROCESS
# ==========================================

transform = transforms.Compose([
    transforms.ToTensor()
])

input_tensor = transform(
    image_rgb
).unsqueeze(0).to(device)

# ==========================================
# GRADCAM HOOKS
# ==========================================

gradients = []
activations = []

def backward_hook(module, grad_input, grad_output):

    gradients.append(
        grad_output[0]
    )

def forward_hook(module, input, output):

    activations.append(output)

target_layer = classifier.layer4[-1]

target_layer.register_forward_hook(
    forward_hook
)

target_layer.register_full_backward_hook(
    backward_hook
)

# ==========================================
# CLASSIFICATION
# ==========================================

output = classifier(input_tensor)

predicted_class = torch.argmax(
    output,
    dim=1
).item()

confidence = torch.softmax(
    output,
    dim=1
)[0][predicted_class].item()

# ==========================================
# BACKPROP
# ==========================================

classifier.zero_grad()

output[0, predicted_class].backward()

# ==========================================
# GRADCAM
# ==========================================

grads = gradients[0]

acts = activations[0]

weights = torch.mean(
    grads,
    dim=[2,3],
    keepdim=True
)

cam = torch.sum(
    weights * acts,
    dim=1
).squeeze()

cam = torch.relu(cam)

cam = cam.detach().cpu().numpy()

cam = cv2.resize(
    cam,
    (224,224)
)

cam = (cam - cam.min()) / (
    cam.max() - cam.min() + 1e-8
)

# ==========================================
# CLASS-AWARE ROI THRESHOLD
# ==========================================

predicted_name = class_names[predicted_class]

if predicted_name == "Scratch":

    threshold = 0.30

elif predicted_name == "Donut":

    threshold = 0.45

elif predicted_name == "Near-Full":

    threshold = 0.55

elif predicted_name == "Edge-Loc":

    threshold = 0.40

elif predicted_name == "Edge-Ring":

    threshold = 0.42

elif predicted_name == "Random":

    threshold = 0.45

else:

    threshold = 0.50

# ==========================================
# ROI EXTRACTION
# ==========================================

roi_mask = np.zeros_like(cam)

roi_mask[cam > threshold] = 1

roi_mask = (
    roi_mask * 255
).astype(np.uint8)

# ==========================================
# APPLY WAFER MASK
# ==========================================

roi_mask = cv2.bitwise_and(
    roi_mask,
    wafer_mask
)

# ==========================================
# CLASS-AWARE MORPHOLOGY
# ==========================================

if predicted_name == "Scratch":

    kernel = np.ones((2,2), np.uint8)

    roi_mask = cv2.morphologyEx(
        roi_mask,
        cv2.MORPH_OPEN,
        kernel
    )

elif predicted_name == "Donut":

    kernel = np.ones((5,5), np.uint8)

    roi_mask = cv2.morphologyEx(
        roi_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

elif predicted_name == "Edge-Loc":

    kernel = np.ones((4,4), np.uint8)

    roi_mask = cv2.dilate(
        roi_mask,
        kernel,
        iterations=1
    )

elif predicted_name == "Near-Full":

    kernel = np.ones((7,7), np.uint8)

    roi_mask = cv2.morphologyEx(
        roi_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

else:

    kernel = np.ones((3,3), np.uint8)

    roi_mask = cv2.morphologyEx(
        roi_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

# ==========================================
# APPLY ROI TO IMAGE
# ==========================================

guided_image = image_rgb.copy()

guided_image[
    roi_mask == 0
] = 0

# ==========================================
# U-NET INPUT
# ==========================================

guided_input = guided_image.astype(
    np.float32
) / 255.0

guided_input = np.transpose(
    guided_input,
    (2,0,1)
)

guided_input = torch.tensor(
    guided_input,
    dtype=torch.float32
).unsqueeze(0).to(device)

# ==========================================
# U-NET PREDICTION
# ==========================================

with torch.no_grad():

    seg_output = segmenter(
        guided_input
    )

    seg_output = torch.sigmoid(
        seg_output
    )

# ==========================================
# SEGMENTATION MASK
# ==========================================

seg_mask = seg_output.squeeze().cpu().numpy()

binary_mask = np.zeros_like(seg_mask)

binary_mask[seg_mask > 0.55] = 1

binary_mask = (
    binary_mask * 255
).astype(np.uint8)

# ==========================================
# APPLY WAFER MASK
# ==========================================

binary_mask = cv2.bitwise_and(
    binary_mask,
    wafer_mask
)

# ==========================================
# FINAL CLEANING
# ==========================================

binary_mask = cv2.morphologyEx(
    binary_mask,
    cv2.MORPH_OPEN,
    np.ones((3,3), np.uint8)
)

# ==========================================
# CONTOURS
# ==========================================

contours, _ = cv2.findContours(
    binary_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# ==========================================
# REMOVE SMALL NOISE
# ==========================================

clean_mask = np.zeros_like(binary_mask)

for contour in contours:

    area = cv2.contourArea(contour)

    if area > 20:

        cv2.drawContours(
            clean_mask,
            [contour],
            -1,
            255,
            -1
        )

binary_mask = clean_mask

# ==========================================
# OVERLAY
# ==========================================

overlay = image_rgb.copy()

overlay[
    binary_mask > 0
] = [255,0,0]

result = cv2.addWeighted(
    image_rgb,
    0.75,
    overlay,
    0.45,
    0
)

# ==========================================
# DRAW CONTOURS
# ==========================================

contours, _ = cv2.findContours(
    binary_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

for contour in contours:

    area = cv2.contourArea(contour)

    if area > 20:

        cv2.drawContours(
            result,
            [contour],
            -1,
            (255,255,255),
            1
        )

# ==========================================
# METRICS
# ==========================================

defect_pixels = np.sum(
    binary_mask > 0
)

wafer_pixels = np.sum(
    wafer_mask > 0
)

defect_density = (
    defect_pixels / wafer_pixels
) * 100

yield_percentage = 100 - defect_density

# ==========================================
# RISK LEVEL
# ==========================================

if defect_density > 40:

    risk = "VERY HIGH"

elif defect_density > 20:

    risk = "HIGH"

elif defect_density > 8:

    risk = "MEDIUM"

else:

    risk = "LOW"

# ==========================================
# DISPLAY
# ==========================================

plt.figure(figsize=(12,12))

plt.imshow(result)

plt.title(

    f"Industrial Guided Segmentation\n"
    f"Class: {predicted_name} | "
    f"Confidence: {confidence*100:.2f}%",

    fontsize=16,

    fontweight='bold'
)

# ==========================================
# METRICS PANEL
# ==========================================

plt.figtext(

    0.72,
    0.10,

    f"Defect Density : {defect_density:.2f}%\n\n"
    f"Estimated Yield : {yield_percentage:.2f}%\n\n"
    f"Risk Level : {risk}",

    fontsize=12,

    color='white',

    bbox=dict(
        facecolor='black',
        alpha=0.9,
        edgecolor='white'
    )
)

# ==========================================
# LEGEND
# ==========================================

plt.figtext(

    0.72,
    0.03,

    "RED : Guided Defect Region",

    fontsize=11,

    color='white',

    bbox=dict(
        facecolor='black',
        alpha=0.9,
        edgecolor='white'
    )
)

# ==========================================
# FINAL DISPLAY
# ==========================================

plt.axis('off')

plt.tight_layout()

plt.show()
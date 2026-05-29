import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

# ==========================================
# DEVICE
# ==========================================

device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"
)

print(f"\nUsing Device: {device}")

# ==========================================
# LOAD U-NET MODEL
# ==========================================

model = smp.Unet(

    encoder_name="resnet34",

    encoder_weights=None,

    in_channels=3,

    classes=1
)

model.load_state_dict(

    torch.load(

        "models/unet_wafer.pth",

        map_location=device
    )
)

model = model.to(device)

model.eval()

print("\nU-Net Model Loaded Successfully.")

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

    raise ValueError(
        "\nInvalid image path."
    )

# ==========================================
# RESIZE
# ==========================================

original = cv2.resize(
    image,
    (224,224)
)

image_rgb = cv2.cvtColor(
    original,
    cv2.COLOR_BGR2RGB
)

# ==========================================
# NORMALIZATION
# ==========================================

image_norm = image_rgb.astype(np.float32) / 255.0

# ==========================================
# CHANNEL FIRST
# ==========================================

image_input = np.transpose(
    image_norm,
    (2,0,1)
)

image_input = torch.tensor(
    image_input,
    dtype=torch.float32
).unsqueeze(0)

image_input = image_input.to(device)

# ==========================================
# PREDICTION
# ==========================================

with torch.no_grad():

    output = model(image_input)

    output = torch.sigmoid(output)

# ==========================================
# CONVERT TO MASK
# ==========================================

mask = output.squeeze().cpu().numpy()

# ==========================================
# THRESHOLD
# ==========================================

binary_mask = np.zeros_like(mask)

binary_mask[mask > 0.5] = 1

# ==========================================
# REMOVE SMALL NOISE
# ==========================================

binary_mask = (binary_mask * 255).astype(np.uint8)

kernel = np.ones((3,3), np.uint8)

binary_mask = cv2.morphologyEx(
    binary_mask,
    cv2.MORPH_OPEN,
    kernel
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
# CREATE OVERLAY
# ==========================================

overlay = image_rgb.copy()

# RED = DEFECTS

overlay[
    binary_mask > 0
] = [255,0,0]

# ==========================================
# BLEND
# ==========================================

result = cv2.addWeighted(
    image_rgb,
    0.65,
    overlay,
    0.35,
    0
)

# ==========================================
# DRAW CONTOURS
# ==========================================

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

total_pixels = 224 * 224

defect_density = (
    defect_pixels / total_pixels
) * 100

yield_percentage = 100 - defect_density

# ==========================================
# RISK LEVEL
# ==========================================

if defect_density > 50:

    risk = "VERY HIGH"

elif defect_density > 25:

    risk = "HIGH"

elif defect_density > 10:

    risk = "MEDIUM"

else:

    risk = "LOW"

# ==========================================
# DISPLAY
# ==========================================

plt.figure(figsize=(12,12))

plt.imshow(result)

plt.title(

    "Industrial U-Net Segmentation",

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

    "RED : Defect Region",

    fontsize=11,

    color='white',

    bbox=dict(
        facecolor='black',
        alpha=0.9,
        edgecolor='white'
    )
)

plt.axis('off')

plt.tight_layout()

plt.show()
# =====================================================
# WAFER DEFECT CLASSIFIER
# API + DASHBOARD SAFE VERSION
# =====================================================

import torch
import torch.nn as nn
import numpy as np

from torchvision import transforms
from torchvision import models

from PIL import Image

# =====================================================
# DEVICE
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\n==================================================")
print(f"DEVICE : {device}")
print("==================================================\n")

# =====================================================
# EXACT TRAINING CLASS ORDER
# =====================================================

CLASS_NAMES = [

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

# =====================================================
# MODEL
# =====================================================

NUM_CLASSES = len(CLASS_NAMES)

model = models.resnet50(
    weights=None
)

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    NUM_CLASSES
)

# =====================================================
# LOAD MODEL
# =====================================================

MODEL_PATH = "models/wafer_model.pth"

print("\n==================================================")
print("LOADING MODEL")
print("==================================================")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint
)

model.to(device)

model.eval()

print(f"\nMODEL LOADED : {MODEL_PATH}")

print("\nCLASS MAPPING")

for idx, cls in enumerate(CLASS_NAMES):

    print(f"[{idx}] {cls}")

print("==================================================\n")

# =====================================================
# EXACT TRAINING TRANSFORM
# MUST MATCH TRAIN.PY
# =====================================================

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[0.485, 0.456, 0.406],

        std=[0.229, 0.224, 0.225]
    )
])

# =====================================================
# INTERNAL INFERENCE
# =====================================================

def run_inference(image):

    tensor = transform(image)

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(device)

    with torch.no_grad():

        outputs = model(tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            1
        )

    predicted_index = predicted.item()

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence_score = (
        confidence.item() * 100.0
    )

    all_probs = {}

    for idx, cls in enumerate(CLASS_NAMES):

        prob = probabilities[0][idx].item() * 100.0

        all_probs[cls] = round(prob, 2)

    return {
        "predicted_class": predicted_class,
        "confidence_score": confidence_score,
        "predicted_index": predicted_index,
        "all_probabilities": all_probs
    }

# =====================================================
# MAIN PREDICTION FUNCTION
# DASHBOARD SAFE
# =====================================================

def predict_image(image_input):

    # =============================================
    # INPUT HANDLING
    # =============================================

    if isinstance(image_input, np.ndarray):

        image = Image.fromarray(
            image_input.astype(np.uint8)
        ).convert("RGB")

        input_type = "NumPy RGB Array"

    elif isinstance(image_input, str):

        image = Image.open(
            image_input
        ).convert("RGB")

        input_type = "Image Path"

    elif isinstance(image_input, Image.Image):

        image = image_input.convert("RGB")

        input_type = "PIL Image"

    else:

        raise TypeError(
            "predict_image expects path, PIL image, or numpy array"
        )

    # =============================================
    # MODEL INFERENCE
    # =============================================

    result = run_inference(image)

    predicted_class = result["predicted_class"]

    confidence_score = result["confidence_score"]

    predicted_index = result["predicted_index"]

    probabilities = result["all_probabilities"]

    # =============================================
    # INDUSTRIAL DEBUG LOG
    # =============================================

    print("\n==================================================")
    print("           WAFER AI PREDICTION RESULT")
    print("==================================================")

    print(f"INPUT TYPE         : {input_type}")

    print(f"IMAGE SIZE         : {image.size}")

    print(f"CLASS INDEX        : {predicted_index}")

    print(f"DEFECT TYPE        : {predicted_class}")

    print(f"CONFIDENCE SCORE   : {confidence_score:.2f}%")

    print("\n==================================================")
    print("ALL CLASS PROBABILITIES")
    print("==================================================")

    for cls, prob in probabilities.items():

        marker = (
            " <-- MAX"
            if cls == predicted_class
            else ""
        )

        print(f"{cls:<15} : {prob:>7.2f}%{marker}")

    print("==================================================\n")

    # =============================================
    # KEEP DASHBOARD COMPATIBILITY
    # =============================================

    return predicted_class, confidence_score

# =====================================================
# API FUNCTION
# =====================================================

def predict_api(image_input):

    if isinstance(image_input, np.ndarray):

        image = Image.fromarray(
            image_input.astype(np.uint8)
        ).convert("RGB")

    elif isinstance(image_input, str):

        image = Image.open(
            image_input
        ).convert("RGB")

    elif isinstance(image_input, Image.Image):

        image = image_input.convert("RGB")

    else:

        raise TypeError(
            "predict_api expects path, PIL image, or numpy array"
        )

    result = run_inference(image)

    return {

        "class": result["predicted_class"],

        "confidence": round(
            result["confidence_score"],
            2
        ),

        "class_index": result["predicted_index"],

        "probabilities": result["all_probabilities"]
    }

# =====================================================
# TERMINAL TESTING
# =====================================================

if __name__ == "__main__":

    print("\n==================================================")
    print("         WAFER DEFECT PREDICTION TEST")
    print("==================================================")

    image_path = input(
        "\nEnter wafer image path:\n\nPath: "
    )

    predicted_class, confidence = predict_image(
        image_path
    )

    print("\n==================================================")
    print("FINAL RESULT")
    print("==================================================")

    print(f"DEFECT TYPE      : {predicted_class}")

    print(f"CONFIDENCE SCORE : {confidence:.2f}%")

    print("==================================================\n")
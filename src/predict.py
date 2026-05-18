import torch

from PIL import Image

from preprocess import test_transform
from model import get_model

# -----------------------------------
# CLASSES
# -----------------------------------

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

# -----------------------------------
# LOAD MODEL
# -----------------------------------

model = get_model(len(classes))

model.load_state_dict(

    torch.load(
        "wafer_model.pth",
        map_location=torch.device('cpu')
    )
)

model.eval()

# -----------------------------------
# IMAGE INPUT
# -----------------------------------

image_path = input(

    "\nEnter image path: "
)

# -----------------------------------
# LOAD IMAGE
# -----------------------------------

image = Image.open(image_path)

# -----------------------------------
# APPLY TRANSFORM
# -----------------------------------

image = test_transform(image).unsqueeze(0)

# -----------------------------------
# PREDICTION
# -----------------------------------

with torch.no_grad():

    output = model(image)

    probabilities = torch.nn.functional.softmax(

        output,

        dim=1
    )

    confidence, predicted = torch.max(

        probabilities,

        1
    )

# -----------------------------------
# RESULTS
# -----------------------------------

predicted_class = classes[predicted.item()]

confidence_score = confidence.item() * 100

print(

    f"\nPredicted Class: {predicted_class}"
)

print(

    f"Confidence Score: {confidence_score:.2f}%"
)
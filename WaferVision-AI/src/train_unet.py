import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ==========================================
# DEVICE
# ==========================================

device = torch.device(

    "cuda" if torch.cuda.is_available()

    else "cpu"
)

print(f"\nUsing Device: {device}")

# ==========================================
# PATHS
# ==========================================

IMAGE_DIR = "data/images"
MASK_DIR = "data/masks"

# ==========================================
# DATASET
# ==========================================

class WaferDataset(Dataset):

    def __init__(self):

        self.images = os.listdir(IMAGE_DIR)

    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):

        image_name = self.images[idx]

        image_path = os.path.join(
            IMAGE_DIR,
            image_name
        )

        mask_path = os.path.join(
            MASK_DIR,
            image_name
        )

        image = cv2.imread(image_path)

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = cv2.resize(
            image,
            (224,224)
        )

        image = image.astype(np.float32) / 255.0

        mask = cv2.imread(
            mask_path,
            0
        )

        mask = cv2.resize(
            mask,
            (224,224)
        )

        mask = mask.astype(np.float32) / 255.0

        image = np.transpose(
            image,
            (2,0,1)
        )

        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        mask = torch.tensor(
            mask,
            dtype=torch.float32
        ).unsqueeze(0)

        return image, mask

# ==========================================
# DATASET
# ==========================================

dataset = WaferDataset()

loader = DataLoader(

    dataset,

    batch_size=8,

    shuffle=True
)

# ==========================================
# U-NET MODEL
# ==========================================

model = smp.Unet(

    encoder_name="resnet34",

    encoder_weights="imagenet",

    in_channels=3,

    classes=1
)

model = model.to(device)

# ==========================================
# LOSS
# ==========================================

loss_fn = torch.nn.BCEWithLogitsLoss()

# ==========================================
# OPTIMIZER
# ==========================================

optimizer = torch.optim.Adam(

    model.parameters(),

    lr=1e-4
)

# ==========================================
# TRAINING
# ==========================================

EPOCHS = 15

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    loop = tqdm(loader)

    for images, masks in loop:

        images = images.to(device)

        masks = masks.to(device)

        outputs = model(images)

        loss = loss_fn(
            outputs,
            masks
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        loop.set_description(
            f"Epoch [{epoch+1}/{EPOCHS}]"
        )

        loop.set_postfix(
            loss=loss.item()
        )

    avg_loss = total_loss / len(loader)

    print(
        f"\nEpoch {epoch+1} Loss: {avg_loss:.4f}"
    )

# ==========================================
# SAVE MODEL
# ==========================================

os.makedirs(
    "models",
    exist_ok=True
)

torch.save(

    model.state_dict(),

    "models/unet_wafer.pth"
)

print("\nU-Net Model Saved Successfully.")
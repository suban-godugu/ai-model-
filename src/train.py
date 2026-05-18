import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets
from torchvision import transforms
from torchvision import models

from torch.utils.data import DataLoader

# =========================================
# DEVICE CONFIGURATION
# =========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing Device: {device}")

# =========================================
# DATASET PATHS
# =========================================

train_dir = "data/wafer_dataset/wafer_dataset/train"

valid_dir = "data/wafer_dataset/wafer_dataset/valid"

test_dir = "data/wafer_dataset/wafer_dataset/test"

# =========================================
# CHECK PATHS
# =========================================

print("\nChecking Dataset Paths...\n")

print("Train Exists :", os.path.exists(train_dir))
print("Valid Exists :", os.path.exists(valid_dir))
print("Test Exists  :", os.path.exists(test_dir))

# =========================================
# DATA AUGMENTATION
# =========================================

train_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomVerticalFlip(),

    transforms.RandomRotation(20),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),

    transforms.ToTensor()
])

# =========================================
# VALIDATION / TEST TRANSFORM
# =========================================

test_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor()
])

# =========================================
# LOAD DATASETS
# =========================================

train_dataset = datasets.ImageFolder(
    root=train_dir,
    transform=train_transform
)

valid_dataset = datasets.ImageFolder(
    root=valid_dir,
    transform=test_transform
)

test_dataset = datasets.ImageFolder(
    root=test_dir,
    transform=test_transform
)

# =========================================
# DATA LOADERS
# =========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=16,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)

# =========================================
# PRINT DATASET INFO
# =========================================

print("\nClasses:\n")

print(train_dataset.classes)

print(f"\nTrain Images      : {len(train_dataset)}")
print(f"Validation Images : {len(valid_dataset)}")
print(f"Test Images       : {len(test_dataset)}")

# =========================================
# LOAD PRETRAINED RESNET50
# =========================================

model = models.resnet50(
    weights=models.ResNet50_Weights.DEFAULT
)

# =========================================
# MODIFY FINAL LAYER
# =========================================

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    len(train_dataset.classes)
)

model = model.to(device)

# =========================================
# LOSS FUNCTION
# =========================================

criterion = nn.CrossEntropyLoss()

# =========================================
# OPTIMIZER
# =========================================

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001
)

# =========================================
# TRAINING LOOP
# =========================================

epochs = 15

for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total

    print(f"\nEpoch [{epoch+1}/{epochs}]")
    print(f"Loss: {running_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy:.2f}%")

# =========================================
# TESTING
# =========================================

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

# =========================================
# FINAL TEST ACCURACY
# =========================================

test_accuracy = 100 * correct / total

print(f"\nFinal Test Accuracy: {test_accuracy:.2f}%")

# =========================================
# SAVE MODEL
# =========================================

torch.save(
    model.state_dict(),
    "wafer_model.pth"
)

print("\nResNet50 Model Saved Successfully!")
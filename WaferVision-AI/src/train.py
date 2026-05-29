# =====================================================
# INDUSTRIAL WAFER DEFECT TRAINING
# FINAL OPTIMIZED VERSION
# =====================================================

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from model import get_model

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
# DATA PATHS
# =====================================================

TRAIN_DIR = "data/wafer_dataset/wafer_dataset/train"

VALID_DIR = "data/wafer_dataset/wafer_dataset/valid"

# =====================================================
# HYPERPARAMETERS
# =====================================================

BATCH_SIZE = 16

EPOCHS = 15

LEARNING_RATE = 0.0001

IMG_SIZE = 224

# =====================================================
# TRANSFORMS
# =====================================================

train_transform = transforms.Compose([

    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomVerticalFlip(),

    transforms.RandomRotation(15),

    transforms.ColorJitter(
        brightness=0.08,
        contrast=0.08
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

valid_transform = transforms.Compose([

    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================================================
# DATASETS
# =====================================================

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

valid_dataset = datasets.ImageFolder(
    VALID_DIR,
    transform=valid_transform
)

# =====================================================
# CLASS INFO
# =====================================================

class_names = train_dataset.classes

num_classes = len(class_names)

print("\n==================================================")
print("CLASS MAPPING")
print("==================================================\n")

for idx, cls in enumerate(class_names):

    print(f"[{idx}] {cls}")

# =====================================================
# CLASS COUNTS
# =====================================================

from collections import Counter

labels = [label for _, label in train_dataset.samples]

count = Counter(labels)

print("\n==================================================")
print("CLASS COUNTS")
print("==================================================\n")

for idx, cls in enumerate(class_names):

    print(f"{cls:<15} : {count[idx]}")

# =====================================================
# DATALOADERS
# =====================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# =====================================================
# MODEL
# =====================================================

model = get_model(num_classes)

model.to(device)

# =====================================================
# LOSS FUNCTION
# =====================================================

criterion = nn.CrossEntropyLoss()

# =====================================================
# OPTIMIZER
# =====================================================

optimizer = optim.Adam(

    model.parameters(),

    lr=LEARNING_RATE
)

# =====================================================
# LR SCHEDULER
# =====================================================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="max",

    factor=0.5,

    patience=2
)

# =====================================================
# TRAINING LOOP
# =====================================================

best_acc = 0.0

for epoch in range(EPOCHS):

    print("\n==================================================")
    print(f"EPOCH {epoch+1}/{EPOCHS}")
    print("==================================================")

    # =================================================
    # TRAIN
    # =================================================

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

    train_acc = 100 * correct / total

    avg_loss = running_loss / len(train_loader)

    # =================================================
    # VALIDATION
    # =================================================

    model.eval()

    val_correct = 0

    val_total = 0

    y_true = []

    y_pred = []

    with torch.no_grad():

        for images, labels in valid_loader:

            images = images.to(device)

            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)

            val_correct += (predicted == labels).sum().item()

            y_true.extend(labels.cpu().numpy())

            y_pred.extend(predicted.cpu().numpy())

    val_acc = 100 * val_correct / val_total

    scheduler.step(val_acc)

    # =================================================
    # PRINT METRICS
    # =================================================

    print(f"\nLOSS          : {avg_loss:.4f}")

    print(f"TRAIN ACC     : {train_acc:.2f}%")

    print(f"VALID ACC     : {val_acc:.2f}%")

    # =================================================
    # SAVE BEST MODEL
    # =================================================

    if val_acc > best_acc:

        best_acc = val_acc

        torch.save(

            model.state_dict(),

            "models/wafer_model.pth"
        )

        print("\nBEST MODEL SAVED")

# =====================================================
# FINAL REPORT
# =====================================================

print("\n==================================================")
print("FINAL CLASSIFICATION REPORT")
print("==================================================\n")

print(classification_report(
    y_true,
    y_pred,
    target_names=class_names
))

print("\n==================================================")
print("CONFUSION MATRIX")
print("==================================================\n")

print(confusion_matrix(
    y_true,
    y_pred
))

print("\n==================================================")
print(f"BEST VALIDATION ACCURACY : {best_acc:.2f}%")
print("==================================================\n")
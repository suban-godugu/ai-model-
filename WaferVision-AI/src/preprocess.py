from torchvision import transforms

# ===================================
# TRAIN TRANSFORM
# ===================================

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

# ===================================
# TEST TRANSFORM
# ===================================

test_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor()
])
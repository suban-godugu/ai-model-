import torch.nn as nn

from torchvision import models

# ===================================
# LOAD RESNET50 MODEL
# ===================================

def get_model(num_classes):

    model = models.resnet50(

        weights=models.ResNet50_Weights.DEFAULT
    )

    # ===================================
    # MODIFY FINAL LAYER
    # ===================================

    num_features = model.fc.in_features

    model.fc = nn.Linear(

        num_features,

        num_classes
    )

    return model
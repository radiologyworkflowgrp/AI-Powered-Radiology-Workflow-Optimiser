import torch
import torch.nn as nn
from torchvision import models

class BrainTumorModel(nn.Module):
    """Brain tumor classification model based on EfficientNet"""
    
    def __init__(self, num_classes=4):
        super(BrainTumorModel, self).__init__()
        # Use EfficientNet-B0 as backbone
        self.model = models.efficientnet_b0(pretrained=False)
        # Replace final layer
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

# Test loading
print("Creating model...")
model = BrainTumorModel(num_classes=4)

print("Loading checkpoint...")
checkpoint = torch.load(r'c:\Users\saite\Downloads\cloudfinal\best_mri_model.pth', 
                       map_location='cpu', 
                       weights_only=False)

print("Loading state dict...")
model.load_state_dict(checkpoint)

print("✓ Model loaded successfully!")
print(f"Model type: EfficientNet-B0")
print(f"Number of classes: 4")

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

TUMOR_CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

class BrainTumorModel(nn.Module):
    """Brain tumor classification model based on EfficientNet
    
    Note: This model uses the efficientnet-pytorch library with grayscale input.
    """
    
    def __init__(self, num_classes=4):
        super(BrainTumorModel, self).__init__()
        # Load EfficientNet-B0 with 1 input channel (grayscale)
        self.model = EfficientNet.from_name('efficientnet-b0', num_classes=num_classes, in_channels=1)
    
    def forward(self, x):
        return self.model(x)

# Test loading
print("Loading checkpoint...")
checkpoint = torch.load(r'c:\Users\saite\Downloads\cloudfinal\best_mri_model.pth', 
                       map_location='cpu', 
                       weights_only=False)

print("Creating model wrapper...")
wrapper = BrainTumorModel(num_classes=len(TUMOR_CLASSES))

print("Loading state dict...")
wrapper.load_state_dict(checkpoint)

print("Extracting model...")
model = wrapper.model

print("✓ Model loaded successfully!")
print(f"Model type: {type(model)}")

# Test forward pass with grayscale input
print("\nTesting forward pass with grayscale input...")
dummy_input = torch.randn(1, 1, 224, 224)  # 1 channel for grayscale
with torch.no_grad():
    output = model(dummy_input)
print(f"Output shape: {output.shape}")
print(f"Number of classes: {output.shape[1]}")
print(f"✓ Forward pass successful!")
print(f"\n✓✓✓ ALL TESTS PASSED! ✓✓✓")

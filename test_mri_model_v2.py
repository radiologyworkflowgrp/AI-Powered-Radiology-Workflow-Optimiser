import torch
import torch.nn as nn

class BrainTumorModel(nn.Module):
    """Brain tumor classification model based on EfficientNet
    
    Note: This model uses the old EfficientNet implementation with _fc layer.
    The checkpoint was trained with efficientnet-pytorch library, not torchvision.
    """
    
    def __init__(self, num_classes=4):
        super(BrainTumorModel, self).__init__()
        # This will be populated when loading the checkpoint
        self.model = None
    
    def forward(self, x):
        return self.model(x)

# Test loading
print("Loading checkpoint...")
checkpoint = torch.load(r'c:\Users\saite\Downloads\cloudfinal\best_mri_model.pth', 
                       map_location='cpu', 
                       weights_only=False)

print("Creating wrapper model...")
wrapper = BrainTumorModel(num_classes=4)

print("Loading state dict into wrapper...")
wrapper.load_state_dict(checkpoint)

print("Extracting model...")
model = wrapper.model

print("✓ Model loaded successfully!")
print(f"Model type: {type(model)}")
print(f"Model has _fc layer: {hasattr(model, '_fc')}")

# Test forward pass
print("\nTesting forward pass...")
dummy_input = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    output = model(dummy_input)
print(f"Output shape: {output.shape}")
print(f"✓ Forward pass successful!")

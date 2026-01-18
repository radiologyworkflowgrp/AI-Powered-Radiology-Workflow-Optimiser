#!/usr/bin/env python3
"""
test_duoformer.py

Test script for DuoFormer model integration
Tests model loading and basic inference
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_model_loading():
    """Test if the model can be loaded"""
    print("=" * 60)
    print("Test 1: Model Loading")
    print("=" * 60)
    
    try:
        from duoformer_inference import DuoFormerInference
        
        model_path = Path(__file__).parent.parent / "best.pt"
        print(f"Model path: {model_path}")
        print(f"Model exists: {model_path.exists()}")
        print(f"Model size: {model_path.stat().st_size / (1024*1024):.2f} MB")
        
        print("\nLoading model...")
        inference = DuoFormerInference(str(model_path))
        print("✓ Model loaded successfully!")
        
        return True
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_image_preprocessing():
    """Test image preprocessing"""
    print("\n" + "=" * 60)
    print("Test 2: Image Preprocessing")
    print("=" * 60)
    
    try:
        from duoformer_inference import DuoFormerInference
        from PIL import Image
        import numpy as np
        import io
        
        # Create a dummy chest X-ray image (grayscale)
        dummy_image = Image.new('L', (512, 512), color=128)
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        dummy_image.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()
        
        print(f"Created dummy image: {len(img_bytes)} bytes")
        
        model_path = Path(__file__).parent.parent / "best.pt"
        inference = DuoFormerInference(str(model_path))
        
        print("Preprocessing image...")
        tensor = inference.preprocess_image(img_bytes)
        print(f"✓ Preprocessed tensor shape: {tensor.shape}")
        print(f"  Expected: [1, 3, 224, 224]")
        
        return True
        
    except Exception as e:
        print(f"✗ Image preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inference():
    """Test inference on dummy image"""
    print("\n" + "=" * 60)
    print("Test 3: Model Inference")
    print("=" * 60)
    
    try:
        from duoformer_inference import DuoFormerInference
        from PIL import Image
        import io
        
        # Create a dummy chest X-ray image
        dummy_image = Image.new('L', (512, 512), color=128)
        img_bytes = io.BytesIO()
        dummy_image.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()
        
        model_path = Path(__file__).parent.parent / "best.pt"
        inference = DuoFormerInference(str(model_path))
        
        print("Running inference...")
        results = inference.predict(img_bytes)
        
        print("\n✓ Inference completed!")
        print(f"\nResults:")
        print(f"  Model: {results['model']}")
        print(f"  Confidence: {results['confidence_score']:.4f}")
        print(f"  Number of predictions: {len(results['predictions'])}")
        print(f"  Number of findings: {len(results['findings'])}")
        
        print("\nFindings:")
        for finding in results['findings']:
            print(f"  - {finding['location']}: {finding['finding']}")
            print(f"    Severity: {finding['severity']}, Confidence: {finding['confidence']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("DuoFormer Integration Test Suite")
    print("=" * 60)
    
    results = []
    
    # Test 1: Model Loading
    results.append(("Model Loading", test_model_loading()))
    
    # Test 2: Image Preprocessing
    if results[0][1]:  # Only if model loaded successfully
        results.append(("Image Preprocessing", test_image_preprocessing()))
    
    # Test 3: Inference
    if results[0][1]:  # Only if model loaded successfully
        results.append(("Inference", test_inference()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())

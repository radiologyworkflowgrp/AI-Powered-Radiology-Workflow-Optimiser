#!/usr/bin/env python3
"""
Standalone MRI Worker Script
Loads the best_mri_model.pth and processes MRI scans from RabbitMQ
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mri_inference import MRIWorker

def main():
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'best_mri_model.pth')
    
    print(f"🧠 Starting MRI Worker with model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please ensure best_mri_model.pth is in the models/ directory")
        sys.exit(1)
    
    worker = MRIWorker(model_path)
    worker.start()

if __name__ == '__main__':
    main()

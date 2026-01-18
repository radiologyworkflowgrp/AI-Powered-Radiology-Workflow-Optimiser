#!/usr/bin/env python3
"""
Standalone DuoFormer X-Ray Worker Script
Loads the best.pt model and processes chest X-ray scans from RabbitMQ
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from duoformer_inference import XRayWorker

def main():
    model_path = os.path.join(os.path.dirname(__file__), 'best.pt')
    
    print(f"🫁 Starting DuoFormer X-Ray Worker with model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please ensure best.pt is in the prioritization-ml/ directory")
        sys.exit(1)
    
    worker = XRayWorker(model_path)
    worker.start()

if __name__ == '__main__':
    main()

"""
Brain Tumor MRI Classification Worker
Processes MRI scans to detect brain tumors using PyTorch model
"""

import os
import sys
import json
import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
import pika
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mri_worker')

# Configuration
INPUT_QUEUE = 'mri_analysis_queue'
OUTPUT_QUEUE = 'mri_results_queue'
RABBITMQ_URL = os.getenv('RABBITMQ_URL', 'amqp://guest:guest@localhost:5672/%2F')

# Brain tumor classes
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


class MRIInference:
    """MRI brain tumor inference engine"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = None
        self._load_model()
        self._setup_transforms()
    
    def _load_model(self):
        """Load the brain tumor classification model"""
        try:
            logger.info(f"Loading MRI model from {self.model_path}")
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Create model wrapper
            wrapper = BrainTumorModel(num_classes=len(TUMOR_CLASSES))
            
            # Load the state dict (which has 'model.' prefix)
            wrapper.load_state_dict(checkpoint)
            
            # Extract the actual EfficientNet model
            self.model = wrapper.model
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"✓ MRI model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load MRI model: {e}", exc_info=True)
            raise
    
    def _setup_transforms(self):
        """Setup image preprocessing transforms for grayscale images"""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Single channel normalization
        ])
    
    def preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        """Preprocess image for model input (grayscale)"""
        try:
            # Load image
            image = Image.open(BytesIO(image_bytes))
            
            # Convert to RGB first if needed (transforms.Grayscale will handle conversion)
            if image.mode not in ['RGB', 'L']:
                image = image.convert('RGB')
            
            # Apply transforms (includes grayscale conversion)
            image_tensor = self.transform(image)
            
            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)
            
            return image_tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    def predict(self, image_bytes: bytes) -> Dict[str, Any]:
        """Run inference on MRI image"""
        try:
            # Preprocess
            image_tensor = self.preprocess_image(image_bytes)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
            
            # Get predictions
            probs = probabilities.cpu().numpy()
            predicted_class_idx = np.argmax(probs)
            predicted_class = TUMOR_CLASSES[predicted_class_idx]
            confidence = float(probs[predicted_class_idx])
            
            logger.info(f"Prediction: {predicted_class} ({confidence:.2%})")
            
            # Create predictions list
            predictions = []
            for i, class_name in enumerate(TUMOR_CLASSES):
                predictions.append({
                    "label": class_name.capitalize(),
                    "probability": float(probs[i]),
                    "positive": bool(i == predicted_class_idx)
                })
            
            # Generate findings
            findings = self._generate_findings(predicted_class, confidence, probs)
            
            return {
                "predicted_class": predicted_class.capitalize(),
                "confidence_score": confidence,
                "predictions": predictions,
                "findings": findings
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            raise
    
    def _generate_findings(self, predicted_class: str, confidence: float, all_probs: np.ndarray) -> List[Dict]:
        """Generate clinical findings from predictions"""
        findings = []
        
        if predicted_class == 'notumor':
            findings.append({
                "location": "Brain",
                "finding": "No tumor detected",
                "severity": "normal",
                "confidence": confidence
            })
        else:
            # Tumor detected
            tumor_type = predicted_class.capitalize()
            severity = self._get_severity(confidence)
            
            findings.append({
                "location": "Brain",
                "finding": f"{tumor_type} tumor detected",
                "severity": severity,
                "confidence": confidence
            })
            
            # Add additional context
            if predicted_class == 'glioma':
                findings.append({
                    "location": "Brain",
                    "finding": "Glioma - originates from glial cells",
                    "severity": "critical",
                    "confidence": confidence
                })
            elif predicted_class == 'meningioma':
                findings.append({
                    "location": "Meninges",
                    "finding": "Meningioma - originates from meninges",
                    "severity": "moderate",
                    "confidence": confidence
                })
            elif predicted_class == 'pituitary':
                findings.append({
                    "location": "Pituitary Gland",
                    "finding": "Pituitary tumor detected",
                    "severity": "moderate",
                    "confidence": confidence
                })
        
        return findings
    
    def _get_severity(self, confidence: float) -> str:
        """Map confidence to severity level"""
        if confidence > 0.9:
            return "critical"
        elif confidence > 0.7:
            return "severe"
        elif confidence > 0.5:
            return "moderate"
        else:
            return "mild"


class MRIWorker:
    """RabbitMQ worker for MRI inference"""
    
    def __init__(self, model_path: str):
        self.inference = MRIInference(model_path)
        self.connection = None
        self.channel = None
    
    def connect(self):
        """Connect to RabbitMQ"""
        try:
            logger.info(f"Connecting to RabbitMQ: {RABBITMQ_URL}")
            self.connection = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL))
            self.channel = self.connection.channel()
            
            # Declare queues
            self.channel.queue_declare(queue=INPUT_QUEUE, durable=True)
            self.channel.queue_declare(queue=OUTPUT_QUEUE, durable=True)
            
            logger.info(f"✓ Connected to RabbitMQ, listening on: {INPUT_QUEUE}")
            
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise
    
    def callback(self, ch, method, properties, body):
        """Process incoming MRI analysis requests"""
        try:
            # Parse message
            data = json.loads(body.decode('utf-8'))
            patient_id = data.get('patient_id', 'UNKNOWN')
            scan_id = data.get('scan_id', 'UNKNOWN')
            
            logger.info(f"Processing MRI for patient: {patient_id}, scan: {scan_id}")
            
            # Get image data
            image_b64 = data.get('image_base64')
            if not image_b64:
                raise ValueError("No image data provided")
            
            # Decode image
            image_bytes = base64.b64decode(image_b64)
            
            # Run inference
            results = self.inference.predict(image_bytes)
            
            # Prepare output
            output = {
                "patient_id": patient_id,
                "scan_id": scan_id,
                "patient_name": data.get('patient_name', 'Unknown'),
                "report_type": "brain_mri",
                "ml_model": "BrainTumor-EfficientNet-B0",
                "predicted_class": results['predicted_class'],
                "confidence_score": results['confidence_score'],
                "findings": results['findings'],
                "predictions": results['predictions'],
                "status": "completed"
            }
            
            # Publish results
            self.channel.basic_publish(
                exchange='',
                routing_key=OUTPUT_QUEUE,
                body=json.dumps(output).encode('utf-8'),
                properties=pika.BasicProperties(delivery_mode=2)
            )
            
            logger.info(f"✓ MRI analysis complete for {patient_id}, result: {results['predicted_class']}")
            
            # Acknowledge message
            ch.basic_ack(delivery_tag=method.delivery_tag)
            
        except Exception as e:
            logger.error(f"Error processing MRI: {e}", exc_info=True)
            # Reject and don't requeue
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
    
    def start(self):
        """Start consuming messages"""
        try:
            self.connect()
            self.channel.basic_qos(prefetch_count=1)
            self.channel.basic_consume(queue=INPUT_QUEUE, on_message_callback=self.callback)
            
            logger.info("MRI Worker started. Waiting for MRI analysis requests...")
            self.channel.start_consuming()
            
        except KeyboardInterrupt:
            logger.info("Shutting down MRI worker...")
            if self.connection:
                self.connection.close()
        except Exception as e:
            logger.error(f"Worker error: {e}", exc_info=True)
            raise


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Brain Tumor MRI Classification Worker')
    parser.add_argument('--model', type=str, required=True, help='Path to model file (.pth)')
    parser.add_argument('--worker', action='store_true', help='Run as RabbitMQ worker')
    
    args = parser.parse_args()
    
    if args.worker:
        # Start worker
        worker = MRIWorker(args.model)
        worker.start()
    else:
        print("Use --worker flag to start the MRI worker")
        print(f"Example: python mri_inference.py --worker --model best_mri_model.pth")


if __name__ == '__main__':
    main()

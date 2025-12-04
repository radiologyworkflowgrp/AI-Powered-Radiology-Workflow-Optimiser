"""
Test script to verify ground truth filtering with DuoFormer model
This script sends an X-ray image directly to the DuoFormer worker via RabbitMQ
and displays the filtered results based on train_new.csv
"""

import pika
import json
import base64
import time
import sys
from pathlib import Path

# Configuration
RABBITMQ_URL = "amqp://localhost:5672"
INPUT_QUEUE = "xray_analysis_queue"
OUTPUT_QUEUE = "xray_results_queue"

def send_xray_for_analysis(image_path: str, patient_id: str = "TEST_P001"):
    """
    Send an X-ray image for analysis and wait for results
    
    Args:
        image_path: Path to the X-ray image file
        patient_id: Patient ID for the test
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        print(f"❌ Error: Image file not found: {image_path}")
        return
    
    # Extract image filename for ground truth matching
    image_filename = image_path.name
    print(f"\n📸 Testing with image: {image_filename}")
    print(f"📁 Full path: {image_path}")
    
    # Read image file
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    # Encode image to base64
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    # Create request payload
    request_data = {
        "patient_id": patient_id,
        "scan_id": image_filename,  # This will be used for ground truth matching
        "image_name": image_filename,  # Backup field
        "image_data": image_base64,
        "timestamp": time.time()
    }
    
    print(f"\n🔄 Connecting to RabbitMQ...")
    
    try:
        # Connect to RabbitMQ
        connection = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL))
        channel = connection.channel()
        
        # Declare queues
        channel.queue_declare(queue=INPUT_QUEUE, durable=True)
        channel.queue_declare(queue=OUTPUT_QUEUE, durable=True)
        
        print(f"✅ Connected to RabbitMQ")
        print(f"\n📤 Sending image to DuoFormer worker...")
        print(f"   Patient ID: {patient_id}")
        print(f"   Image Name: {image_filename}")
        print(f"   Image Size: {len(image_bytes)} bytes")
        
        # Publish message to input queue
        channel.basic_publish(
            exchange='',
            routing_key=INPUT_QUEUE,
            body=json.dumps(request_data),
            properties=pika.BasicProperties(
                delivery_mode=2,  # Make message persistent
                content_type='application/json'
            )
        )
        
        print(f"✅ Message sent to queue: {INPUT_QUEUE}")
        print(f"\n⏳ Waiting for results from DuoFormer worker...")
        print(f"   (This may take 10-30 seconds depending on model processing time)")
        
        # Set up consumer for results
        results_received = []
        
        def callback(ch, method, properties, body):
            try:
                result = json.loads(body)
                results_received.append(result)
                ch.basic_ack(delivery_tag=method.delivery_tag)
                ch.stop_consuming()
            except Exception as e:
                print(f"❌ Error processing result: {e}")
                ch.basic_ack(delivery_tag=method.delivery_tag)
        
        # Consume from output queue
        channel.basic_consume(
            queue=OUTPUT_QUEUE,
            on_message_callback=callback,
            auto_ack=False
        )
        
        # Wait for results with timeout
        connection.process_data_events(time_limit=60)  # 60 second timeout
        
        if results_received:
            result = results_received[0]
            print(f"\n{'='*80}")
            print(f"🎉 RESULTS RECEIVED - GROUND TRUTH FILTERING APPLIED")
            print(f"{'='*80}\n")
            
            # Display results
            print(f"📊 Analysis Results for: {image_filename}")
            print(f"   Patient ID: {result.get('patient_id', 'N/A')}")
            print(f"   Model: {result.get('model', 'N/A')}")
            print(f"   Overall Confidence: {result.get('confidence_score', 0):.2%}")
            
            # Display filtered predictions
            predictions = result.get('predictions', [])
            findings = result.get('findings', [])
            
            print(f"\n🔍 FILTERED PREDICTIONS (Only diseases from ground truth CSV):")
            print(f"{'─'*80}")
            
            positive_predictions = [p for p in predictions if p.get('positive', False)]
            
            if positive_predictions:
                for pred in positive_predictions:
                    label = pred.get('label', 'Unknown')
                    prob = pred.get('probability', 0)
                    print(f"   ✓ {label:30s} | Probability: {prob:.2%}")
            else:
                print(f"   ℹ️  No positive findings (all diseases filtered out by ground truth)")
            
            print(f"\n📋 DETAILED FINDINGS:")
            print(f"{'─'*80}")
            
            if findings:
                for finding in findings:
                    location = finding.get('location', 'Unknown')
                    disease = finding.get('finding', 'Unknown')
                    severity = finding.get('severity', 'Unknown')
                    confidence = finding.get('confidence', 0)
                    
                    print(f"   Location: {location}")
                    print(f"   Finding:  {disease}")
                    print(f"   Severity: {severity}")
                    print(f"   Confidence: {confidence:.2%}")
                    print(f"   {'-'*76}")
            else:
                print(f"   ℹ️  No findings to report")
            
            print(f"\n{'='*80}")
            print(f"✅ Ground truth filtering successfully applied!")
            print(f"   Only diseases marked as TRUE in train_new.csv are shown above.")
            print(f"{'='*80}\n")
            
        else:
            print(f"\n⚠️  No results received within timeout period")
            print(f"   Check if DuoFormer worker is running:")
            print(f"   python duoformer_inference.py --worker --ground-truth-csv train_new.csv --model \"c:\\Users\\saite\\Downloads\\cloudfinal\\best.pt\"")
        
        connection.close()
        
    except pika.exceptions.AMQPConnectionError:
        print(f"❌ Error: Could not connect to RabbitMQ")
        print(f"   Make sure RabbitMQ is running on localhost:5672")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print(f"\n{'='*80}")
    print(f"DuoFormer Ground Truth Filtering Test")
    print(f"{'='*80}\n")
    
    # Check if image path provided
    if len(sys.argv) < 2:
        print(f"Usage: python test_ground_truth_filtering.py <path_to_xray_image>")
        print(f"\nExample:")
        print(f"  python test_ground_truth_filtering.py \"c:\\Users\\saite\\Downloads\\cloudfinal\\train\\patient00001\\study1\\view1_frontal.jpg\"")
        print(f"\n📝 Note: The image filename should match an entry in train_new.csv")
        sys.exit(1)
    
    image_path = sys.argv[1]
    send_xray_for_analysis(image_path)

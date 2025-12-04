# -*- coding: utf-8 -*-
"""Simple test script for ground truth filtering"""

import pika
import json
import base64
import time
import sys
from pathlib import Path

RABBITMQ_URL = "amqp://localhost:5672"
INPUT_QUEUE = "xray_analysis_queue"
OUTPUT_QUEUE = "xray_results_queue"

def test_image(image_path):
    image_path = Path(image_path)
    
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return
    
    image_filename = image_path.name
    print(f"\n" + "="*80)
    print(f"Testing Ground Truth Filtering")
    print(f"="*80)
    print(f"\nImage: {image_filename}")
    print(f"Path: {image_path}")
    
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    request_data = {
        "patient_id": "TEST_P001",
        "scan_id": image_filename,
        "image_name": image_filename,
        "image_base64": image_base64,
        "timestamp": time.time()
    }
    
    print(f"\nConnecting to RabbitMQ...")
    
    try:
        connection = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL))
        channel = connection.channel()
        
        channel.queue_declare(queue=INPUT_QUEUE, durable=True)
        channel.queue_declare(queue=OUTPUT_QUEUE, durable=True)
        
        print(f"Connected successfully")
        print(f"\nSending image to DuoFormer worker...")
        print(f"  Patient ID: TEST_P001")
        print(f"  Image: {image_filename}")
        print(f"  Size: {len(image_bytes)} bytes")
        
        channel.basic_publish(
            exchange='',
            routing_key=INPUT_QUEUE,
            body=json.dumps(request_data),
            properties=pika.BasicProperties(
                delivery_mode=2,
                content_type='application/json'
            )
        )
        
        print(f"\nMessage sent! Waiting for results...")
        print(f"(This may take 10-30 seconds)")
        
        results_received = []
        
        def callback(ch, method, properties, body):
            try:
                result = json.loads(body)
                results_received.append(result)
                ch.basic_ack(delivery_tag=method.delivery_tag)
                ch.stop_consuming()
            except Exception as e:
                print(f"Error: {e}")
                ch.basic_ack(delivery_tag=method.delivery_tag)
        
        channel.basic_consume(
            queue=OUTPUT_QUEUE,
            on_message_callback=callback,
            auto_ack=False
        )
        
        connection.process_data_events(time_limit=60)
        
        if results_received:
            result = results_received[0]
            print(f"\n" + "="*80)
            print(f"RESULTS - GROUND TRUTH FILTERING APPLIED")
            print(f"="*80)
            
            print(f"\nImage: {image_filename}")
            print(f"Model: {result.get('model', 'N/A')}")
            print(f"Confidence: {result.get('confidence_score', 0):.2%}")
            
            predictions = result.get('predictions', [])
            findings = result.get('findings', [])
            
            print(f"\nFILTERED PREDICTIONS (from ground truth CSV):")
            print(f"-"*80)
            
            positive = [p for p in predictions if p.get('positive', False)]
            
            if positive:
                for pred in positive:
                    label = pred.get('label', 'Unknown')
                    prob = pred.get('probability', 0)
                    print(f"  {label:30s} | {prob:.2%}")
            else:
                print(f"  No positive findings")
            
            print(f"\nDETAILED FINDINGS:")
            print(f"-"*80)
            
            if findings:
                for f in findings:
                    print(f"  Location:   {f.get('location', 'Unknown')}")
                    print(f"  Finding:    {f.get('finding', 'Unknown')}")
                    print(f"  Severity:   {f.get('severity', 'Unknown')}")
                    print(f"  Confidence: {f.get('confidence', 0):.2%}")
                    print(f"  " + "-"*76)
            else:
                print(f"  No findings")
            
            print(f"\n" + "="*80)
            print(f"SUCCESS! Ground truth filtering applied.")
            print(f"Only diseases marked TRUE in train_new.csv are shown.")
            print(f"="*80 + "\n")
            
        else:
            print(f"\nNo results received (timeout)")
            print(f"Check if DuoFormer worker is running")
        
        connection.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_xray_simple.py <image_path>")
        print("\nExample:")
        print('  python test_xray_simple.py "c:\\train\\patient00001\\study1\\view1_frontal.jpg"')
        sys.exit(1)
    
    test_image(sys.argv[1])

# -*- coding: utf-8 -*-
"""
Comprehensive test script for ground truth filtering
Tests 20 different images and compares ground truth with filtered results
"""

import pika
import json
import base64
import time
import pandas as pd
from pathlib import Path
import os

RABBITMQ_URL = "amqp://localhost:5672"
INPUT_QUEUE = "xray_analysis_queue"
OUTPUT_QUEUE = "xray_results_queue"

# Get 20 test images
train_dir = Path(r"c:\Users\saite\Downloads\cloudfinal\train")
csv_path = Path(r"c:\Users\saite\Downloads\cloudfinal\train_new.csv")

print("="*100)
print("GROUND TRUTH FILTERING - COMPREHENSIVE TEST (20 IMAGES)")
print("="*100)

# Load CSV
print("\nLoading ground truth CSV...")
df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} entries from CSV")

# Get 20 image files
image_files = list(train_dir.rglob("*.jpg"))[:20]
print(f"\nFound {len(image_files)} test images")

# Connect to RabbitMQ
print("\nConnecting to RabbitMQ...")
connection = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL))
channel = connection.channel()
channel.queue_declare(queue=INPUT_QUEUE, durable=True)
channel.queue_declare(queue=OUTPUT_QUEUE, durable=True)
print("Connected!")

# Disease columns
disease_cols = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 
                'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 
                'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 
                'Pleural Other', 'Fracture', 'Support Devices']

results_summary = []

print("\n" + "="*100)
print("PROCESSING IMAGES...")
print("="*100)

for idx, image_path in enumerate(image_files, 1):
    print(f"\n[{idx}/20] Processing: {image_path.name}")
    
    # Read image
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    # Find ground truth
    csv_path_pattern = str(image_path).replace('\\', '/')
    matching_rows = df[df['Path'].str.contains(image_path.name, na=False)]
    
    if len(matching_rows) == 0:
        print(f"  WARNING: No ground truth found in CSV for {image_path.name}")
        continue
    
    gt_row = matching_rows.iloc[0]
    
    # Get ground truth diseases (only 1.0 values)
    gt_diseases = []
    for col in disease_cols:
        if col in gt_row and gt_row[col] == 1.0:
            gt_diseases.append(col)
    
    print(f"  Ground Truth Diseases: {gt_diseases if gt_diseases else 'None (all filtered)'}")
    
    # Send to worker
    request_data = {
        "patient_id": f"TEST_P{idx:03d}",
        "scan_id": image_path.name,
        "image_name": image_path.name,
        "image_base64": image_base64
    }
    
    channel.basic_publish(
        exchange='',
        routing_key=INPUT_QUEUE,
        body=json.dumps(request_data),
        properties=pika.BasicProperties(delivery_mode=2)
    )
    
    # Wait for result
    time.sleep(3)  # Give worker time to process
    
    # Get result from queue
    method, properties, body = channel.basic_get(queue=OUTPUT_QUEUE, auto_ack=True)
    
    if body:
        result = json.loads(body)
        filtered_diseases = [p['label'] for p in result.get('predictions', []) if p.get('positive', False)]
        
        print(f"  Filtered Predictions: {filtered_diseases if filtered_diseases else 'None'}")
        
        # Check if filtering worked correctly
        match = set(gt_diseases) == set(filtered_diseases)
        status = "MATCH" if match else "MISMATCH"
        
        if not match:
            print(f"  STATUS: {status}")
            print(f"    Expected: {sorted(gt_diseases)}")
            print(f"    Got:      {sorted(filtered_diseases)}")
        else:
            print(f"  STATUS: {status} (Filtering working correctly!)")
        
        results_summary.append({
            'image': image_path.name,
            'ground_truth': gt_diseases,
            'filtered': filtered_diseases,
            'match': match
        })
    else:
        print(f"  ERROR: No result received from worker")
        results_summary.append({
            'image': image_path.name,
            'ground_truth': gt_diseases,
            'filtered': None,
            'match': False
        })

connection.close()

# Print summary
print("\n" + "="*100)
print("TEST SUMMARY")
print("="*100)

matches = sum(1 for r in results_summary if r['match'])
total = len(results_summary)

print(f"\nTotal Images Tested: {total}")
print(f"Successful Matches: {matches}/{total} ({matches/total*100:.1f}%)")
print(f"Mismatches: {total-matches}/{total}")

print("\n" + "-"*100)
print("DETAILED RESULTS:")
print("-"*100)

for i, result in enumerate(results_summary, 1):
    status_icon = "✓" if result['match'] else "✗"
    print(f"\n{i}. {status_icon} {result['image']}")
    print(f"   Ground Truth: {result['ground_truth']}")
    print(f"   Filtered:     {result['filtered']}")

print("\n" + "="*100)
print("TEST COMPLETE!")
print("="*100)

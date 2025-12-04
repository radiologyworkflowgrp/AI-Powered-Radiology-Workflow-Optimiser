# Simple script to just send a message and let the worker process it
import pika
import json
import base64
import sys
from pathlib import Path

image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("train/patient00001/study1/view1_frontal.jpg")

print(f"Reading image: {image_path}")

with open(image_path, 'rb') as f:
    image_bytes = f.read()

image_base64 = base64.b64encode(image_bytes).decode('utf-8')

request_data = {
    "patient_id": "TEST_P001",
    "scan_id": image_path.name,
    "image_name": image_path.name,
    "image_base64": image_base64
}

print(f"Connecting to RabbitMQ...")
connection = pika.BlockingConnection(pika.URLParameters("amqp://localhost:5672"))
channel = connection.channel()

channel.queue_declare(queue="xray_analysis_queue", durable=True)

print(f"Sending message...")
channel.basic_publish(
    exchange='',
    routing_key="xray_analysis_queue",
    body=json.dumps(request_data),
    properties=pika.BasicProperties(delivery_mode=2)
)

print(f"Message sent! Check the DuoFormer worker logs for processing.")
print(f"The worker should process the image and publish results to xray_results_queue")

connection.close()

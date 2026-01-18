# ML Integration Guide

## Overview

The AI-Powered Radiology Workflow Optimiser uses a hybrid approach for ML services:

1.  **Report Generation**: Handled by `ml-services/modelRunner.js` (Node.js). Consumes `report_queue`.
2.  **Patient Prioritization**: Handled by `prioritization-ml/ml_priority_system_pytorch.py` (Python). Consumes `priority_queue` and outputs to `waitlist_queue`.

## Architecture

-   **RabbitMQ**: Message broker connecting services.
-   **Node.js Backend**: API and Report Generation.
-   **Python ML Worker**: PyTorch model for urgency classification.

## Setup

1.  **Install Python Dependencies**:
    ```bash
    cd integrated-backend
    python3 -m venv prioritization-ml/venv
    prioritization-ml/venv/bin/pip install -r prioritization-ml/requirements.txt
    ```

2.  **Services**:
    Make sure RabbitMQ is running:
    ```bash
    sudo systemctl start rabbitmq-server
    ```

## Running ML Services

You need to run BOTH the Node.js ML worker and the Python Priority worker.

1.  **Start Report Generator (Node.js)**:
    ```bash
    npm run dev:ml-models
    ```

2.  **Start Priority Worker (Python)**:
    ```bash
    export OUTPUT_QUEUE=waitlist_queue
    npm run start:ml-priority
    ```
    *Note: `OUTPUT_QUEUE` must be set to `waitlist_queue` to match the backend expectation.*

## Testing

Run the integration test to verify the pipeline:

```bash
node test-ml-integration.js
```

#!/bin/bash

# Function to kill all child processes on exit
cleanup() {
    echo "Stopping all services..."
    kill $(jobs -p) 2>/dev/null
    exit
}

trap cleanup SIGINT SIGTERM

echo "Starting AI-Powered Radiology Backend Services..."

# 1. Start RabbitMQ (if not already running - requires user to have started it usually, but we can try)
# echo "Ensuring RabbitMQ is running..."
# sudo systemctl start rabbitmq-server

# 2. Start Python ML Worker (Priority System)
echo "Starting Python ML Priority Worker..."
export OUTPUT_QUEUE=waitlist_queue
npm run start:ml-priority &
PID_PYTHON=$!

# 3. Start Node.js ML Service (Report Generator)
echo "Starting Node.js ML Report Generator..."
npm run dev:ml-models &
PID_ML_NODE=$!

# 4. Start Main Backend Server
echo "Starting Main Backend Server..."
npm start &
PID_SERVER=$!

echo "==================================================="
echo "All services started!"
echo "Python Worker PID: $PID_PYTHON"
echo "Node ML Worker PID: $PID_ML_NODE"
echo "Backend Server PID: $PID_SERVER"
echo "==================================================="
echo "Press Ctrl+C to stop all services."

# Wait for all background processes
wait

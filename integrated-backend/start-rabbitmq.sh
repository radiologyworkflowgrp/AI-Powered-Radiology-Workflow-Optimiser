#!/bin/bash

# RabbitMQ Startup Script for DICOM Integration

echo "🐰 Starting RabbitMQ Server..."
echo ""

# Check if RabbitMQ is installed
if ! command -v rabbitmq-server &> /dev/null; then
    echo "❌ RabbitMQ is not installed!"
    echo ""
    echo "To install RabbitMQ on Ubuntu/Pop!_OS:"
    echo "  sudo apt update"
    echo "  sudo apt install rabbitmq-server"
    echo ""
    exit 1
fi

# Start RabbitMQ
echo "Starting RabbitMQ service..."
sudo systemctl start rabbitmq-server

# Wait a moment for it to start
sleep 2

# Check status
if systemctl is-active --quiet rabbitmq-server; then
    echo "✅ RabbitMQ is now running!"
    echo ""
    
    # Enable management plugin if not already enabled
    echo "Enabling RabbitMQ Management Plugin..."
    sudo rabbitmq-plugins enable rabbitmq_management
    
    echo ""
    echo "📊 RabbitMQ Management Console: http://localhost:15672"
    echo "   Default credentials: guest / guest"
    echo ""
    echo "🔌 RabbitMQ AMQP Port: 5672"
    echo ""
    echo "✅ Ready for DICOM queue integration!"
    echo ""
    echo "Next steps:"
    echo "1. Restart your backend server (type 'rs' in nodemon)"
    echo "2. Start ML models: npm run start:ml-models"
    echo "3. Test DICOM upload functionality"
else
    echo "❌ Failed to start RabbitMQ"
    echo ""
    echo "Check the status with:"
    echo "  sudo systemctl status rabbitmq-server"
    echo ""
    echo "View logs with:"
    echo "  sudo journalctl -u rabbitmq-server -n 50"
fi

const amqp = require('amqplib');
const path = require('path');
const fs = require('fs');

const RABBITMQ_URL = 'amqp://guest:guest@localhost:5672/';
const PDF_QUEUE = 'pdf_queue';

async function testPdfGen() {
    console.log('Testing PDF Generator Isolation...');

    try {
        const connection = await amqp.connect(RABBITMQ_URL);
        const channel = await connection.createChannel();

        // Ensure queue exists
        await channel.assertQueue(PDF_QUEUE, {
            durable: true,
            arguments: {
                'x-dead-letter-exchange': 'hospital_dlx',
                'x-message-ttl': 3600000
            }
        });

        const testData = {
            patient_id: 'TEST-PDF-ONLY',
            scan_id: 'SCAN-TEST-001',
            timestamp: new Date().toISOString(),
            patient_name: 'Test Patient PDF',
            age: 30,
            gender: 'M',
            scan_type: 'CT Chest',
            priority_score: 0.8,
            urgency_level: 'High',
            findings: ['Normal lung parenchyma', 'No nodules'],
            impression: 'Normal CT Chest',
            recommendations: ['Routine follow-up'],
            ai_analysis: {
                findings: [
                    { location: 'Lungs', confidence: 0.99, severity: 'normal' }
                ]
            }
        };

        console.log('Sending test message to pdf_queue...');
        channel.sendToQueue(PDF_QUEUE, Buffer.from(JSON.stringify(testData)), {
            persistent: true
        });
        console.log('Message sent!');

        await channel.close();
        await connection.close();

        console.log('Check logs and generated_reports folder.');

    } catch (error) {
        console.error('Error:', error);
    }
}

testPdfGen();

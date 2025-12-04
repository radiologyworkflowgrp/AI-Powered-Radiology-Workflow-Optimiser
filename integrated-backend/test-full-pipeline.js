const amqp = require('amqplib');
const fs = require('fs');
const path = require('path');

const RABBITMQ_URL = 'amqp://guest:guest@localhost:5672/';
const INPUT_QUEUE = 'priority_queue';
const PDF_DIR = path.join(__dirname, 'prioritization-ml', 'generated_reports');
const TEST_TIMEOUT = 120000; // 120 seconds for full pipeline

async function testFullPipeline() {
    console.log('='.repeat(60));
    console.log('Full ML Pipeline Integration Test');
    console.log('='.repeat(60));
    console.log(`Timeout: ${TEST_TIMEOUT / 1000}s`);
    console.log(`Checking PDF output in: ${PDF_DIR}`);

    let connection;
    let timeoutId;

    try {
        // 1. Connect to RabbitMQ
        console.log('\nConnecting to RabbitMQ...');
        connection = await amqp.connect(RABBITMQ_URL);
        const channel = await connection.createChannel();
        console.log('✓ Connected to RabbitMQ');

        // 2. Generate Test Data
        const patientId = `TEST-${Math.random().toString(36).substr(2, 8).toUpperCase()}`;
        const patientData = {
            patient_id: patientId,
            patient_name: "Pipeline Test Patient",
            age: 55,
            gender: "Male",
            symptoms: [
                "chest pain",
                "shortness of breath",
                "dizziness"
            ],
            vitals: {
                heartRate: 110,
                bloodPressure: "140/90",
                temperature: 37.5,
                oxygenSaturation: 94
            },
            history: "Hypertension",
            timestamp: new Date().toISOString()
        };

        // 3. Send to Priority Queue
        console.log(`\nSending Test Patient to Priority Queue: ${patientId}`);
        channel.sendToQueue(INPUT_QUEUE, Buffer.from(JSON.stringify(patientData)), {
            persistent: true
        });
        console.log('✓ Message sent');

        // Wait a bit to ensure message is published
        await new Promise(resolve => setTimeout(resolve, 500));

        // 4. Wait for PDF Generation
        console.log('\nWaiting for PDF report generation...');

        const startTime = Date.now();

        return new Promise((resolve, reject) => {
            // Set timeout
            timeoutId = setTimeout(() => {
                reject(new Error(`TIMEOUT: No PDF generated for patient ${patientId} within ${TEST_TIMEOUT / 1000}s`));
            }, TEST_TIMEOUT);

            // Poll for PDF file
            const checkInterval = setInterval(() => {
                try {
                    if (!fs.existsSync(PDF_DIR)) {
                        console.log(`[${new Date().toISOString()}] PDF directory does not exist yet...`);
                        return;
                    }

                    const files = fs.readdirSync(PDF_DIR);
                    const pdfFile = files.find(f => f.includes(patientId) && f.endsWith('.pdf'));

                    if (pdfFile) {
                        clearInterval(checkInterval);
                        clearTimeout(timeoutId);

                        const duration = (Date.now() - startTime) / 1000;
                        console.log(`\n✓ PDF Generated successfully!`);
                        console.log(`  File: ${pdfFile}`);
                        console.log(`  Time taken: ${duration.toFixed(2)}s`);
                        console.log(`  Path: ${path.join(PDF_DIR, pdfFile)}`);

                        resolve();
                    } else {
                        // Log progress every 10 seconds
                        const elapsed = (Date.now() - startTime) / 1000;
                        if (elapsed % 10 < 1) {
                            console.log(`[${elapsed.toFixed(0)}s] Still waiting... (${files.length} files in directory)`);
                        }
                    }
                } catch (err) {
                    console.error('Error checking for PDF:', err.message);
                }
            }, 1000); // Check every second
        });

    } catch (error) {
        console.error('\n❌ Test Failed:', error.message);
        process.exit(1);
    } finally {
        if (connection) {
            await new Promise(resolve => setTimeout(resolve, 1000)); // Wait before closing
            await connection.close();
        }
    }
}

testFullPipeline();

const amqp = require('amqplib');
const { v4: uuidv4 } = require('uuid');

const RABBITMQ_URL = 'amqp://localhost';
const INPUT_QUEUE = 'priority_queue';
const OUTPUT_QUEUE = 'waitlist_queue';
const TEST_TIMEOUT = 30000; // 30 seconds

async function testIntegration() {
    console.log('='.repeat(60));
    console.log('ML Integration Test');
    console.log('='.repeat(60));
    console.log(`Timeout: ${TEST_TIMEOUT / 1000}s\n`);

    let connection;
    let timeoutId;

    try {
        console.log('Connecting to RabbitMQ...');
        connection = await amqp.connect(RABBITMQ_URL);
        const channel = await connection.createChannel();
        console.log('✓ Connected to RabbitMQ\n');

        // Use passive mode to check queues exist without modifying them
        await channel.checkQueue(INPUT_QUEUE);
        await channel.checkQueue(OUTPUT_QUEUE);

        // Set timeout
        timeoutId = setTimeout(() => {
            console.error('\n❌ TIMEOUT: No response received within 30 seconds');
            console.error('Possible issues:');
            console.error('  - Python worker not running');
            console.error('  - RabbitMQ queues misconfigured');
            console.error('  - ML model failed to load');
            console.error('\nCheck logs with: npm run dev:ml-models');
            connection.close();
            process.exit(1);
        }, TEST_TIMEOUT);

        // Listen for result
        console.log(`Listening on ${OUTPUT_QUEUE}...`);
        const startTime = Date.now();

        channel.consume(OUTPUT_QUEUE, (msg) => {
            if (msg) {
                clearTimeout(timeoutId);
                const duration = ((Date.now() - startTime) / 1000).toFixed(2);

                const content = JSON.parse(msg.content.toString());
                console.log('\n' + '='.repeat(60));
                console.log('Received Result from Waitlist Queue');
                console.log('='.repeat(60));
                console.log(JSON.stringify(content, null, 2));
                console.log('\n' + '='.repeat(60));
                console.log('Validation Results');
                console.log('='.repeat(60));

                let success = true;

                // Validate ML processing flag
                if (content.ml_processed === true) {
                    console.log('✅ ml_processed: true');
                } else {
                    console.log('❌ ml_processed: false or missing');
                    success = false;
                }

                // Validate priority
                if (content.priority && typeof content.priority === 'number') {
                    console.log(`✅ priority: ${content.priority} (${content.urgency_level || 'N/A'})`);
                } else {
                    console.log('❌ priority: missing or invalid');
                    success = false;
                }

                // Validate ML confidence
                if (content.ml_confidence !== undefined && typeof content.ml_confidence === 'number') {
                    console.log(`✅ ml_confidence: ${(content.ml_confidence * 100).toFixed(2)}%`);
                } else {
                    console.log('❌ ml_confidence: missing or invalid');
                    success = false;
                }

                // Validate patient data
                if (content.patient_id && content.patient_name) {
                    console.log(`✅ patient_id: ${content.patient_id}`);
                    console.log(`✅ patient_name: ${content.patient_name}`);
                } else {
                    console.log('❌ patient data: missing or invalid');
                    success = false;
                }

                console.log('\n' + '='.repeat(60));
                console.log(`Processing Time: ${duration}s`);
                console.log('='.repeat(60));

                if (success) {
                    console.log('\n✅ TEST PASSED: ML integration working correctly!');
                } else {
                    console.log('\n⚠️ TEST FAILED: Some validations failed (see above)');
                }

                channel.ack(msg);
                setTimeout(() => {
                    connection.close();
                    process.exit(success ? 0 : 1);
                }, 1000);
            }
        });

        // Send test message
        const testPatient = {
            patient_id: `TEST-${uuidv4().substring(0, 8)}`,
            patient_name: 'Test Patient',
            age: 45,
            symptoms: ['chest pain', 'shortness of breath', 'dizziness'],
            vitals: {
                heartRate: 110,
                bloodPressure: '140/90',
                temperature: 37.5,
                oxygenSaturation: 94
            }
        };

        console.log('Sending Test Patient to Priority Queue:');
        console.log(JSON.stringify(testPatient, null, 2));
        console.log('\nWaiting for ML processing...');

        channel.sendToQueue(INPUT_QUEUE, Buffer.from(JSON.stringify(testPatient)), {
            persistent: true
        });

    } catch (error) {
        if (timeoutId) clearTimeout(timeoutId);
        console.error('\n❌ Test failed with error:');
        console.error(error.message);
        console.error('\nStack trace:');
        console.error(error.stack);
        if (connection) {
            await connection.close();
        }
        process.exit(1);
    }
}

testIntegration();

const amqp = require('amqplib');

async function sendTestMessage() {
    try {
        console.log('Connecting to RabbitMQ...');
        const connection = await amqp.connect('amqp://guest:guest@localhost:5672/');
        const channel = await connection.createChannel();

        const patientId = `TEST-${Math.random().toString(36).substr(2, 8).toUpperCase()}`;
        const testData = {
            patient_id: patientId,
            patient_name: "Simple Test Patient",
            age: 45,
            gender: "Male",
            symptoms: ["chest pain"],
            vitals: {
                heartRate: 110,
                bloodPressure: "140/90"
            },
            timestamp: new Date().toISOString()
        };

        console.log(`\nSending patient ${patientId} to priority_queue...`);
        channel.sendToQueue('priority_queue', Buffer.from(JSON.stringify(testData)), {
            persistent: true
        });

        console.log('✓ Message sent!');
        console.log('\nCheck the ml-services terminal for processing logs.');
        console.log(`Look for patient ID: ${patientId}`);

        setTimeout(() => {
            connection.close();
            process.exit(0);
        }, 1000);

    } catch (error) {
        console.error('Error:', error);
        process.exit(1);
    }
}

sendTestMessage();

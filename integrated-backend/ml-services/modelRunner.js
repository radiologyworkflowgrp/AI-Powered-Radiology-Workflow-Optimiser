require('dotenv').config({ path: '../.env' });
const rabbitmq = require('../rabbitmq');
const { MLReport } = require('../models');
const { initDatabase } = require('../postgres');
const mongoose = require('mongoose');
const Patient = require('../mongoSchemas/Patient'); // Need MongoDB patient to get name if needed

async function startWorker() {
    console.log('🤖 ML Model Worker Starting...');

    // Connect to PostgreSQL
    try {
        await initDatabase();
        console.log('✅ Connected to PostgreSQL');
    } catch (err) {
        console.error('❌ Postgres connection failed:', err);
        process.exit(1);
    }

    // Connect to MongoDB (for patient name lookup)
    try {
        await mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/radiology_hospital');
        console.log('✅ Connected to MongoDB');
    } catch (err) {
        console.error('❌ MongoDB connection failed:', err);
    }

    // Connect to RabbitMQ
    try {
        const connected = await rabbitmq.connect();
        if (!connected) {
            console.error('❌ RabbitMQ connection failed');
            process.exit(1);
        }
        console.log('✅ Connected to RabbitMQ. Waiting for tasks...');
    } catch (err) {
        console.error('❌ RabbitMQ init failed:', err);
        process.exit(1);
    }

    // Consume Report Queue
    rabbitmq.consume('report_queue', async (data, msg) => {
        console.log(`\n📦 [Received Task] Scan: ${data.scanId} | Patient: ${data.patientId}`);

        try {
            // 1. Fetch Patient Name from Mongo
            let patientName = "Unknown";
            try {
                const patient = await Patient.findById(data.patientId);
                if (patient) patientName = patient.name;
            } catch (pErr) {
                console.warn('⚠️ Could not fetch patient name:', pErr.message);
            }

            // 2. Create Initial Report in Postgres
            let report = await MLReport.create({
                patient_id: data.patientId, // String (Mongo ID)
                doctor_id: null,
                patient_name: patientName,
                report_type: data.modality || "General",
                ml_model: "DuoFormer-v1",
                report_status: "processing",
                confidence_score: 0,
                created_at: new Date(),
                updated_at: new Date()
            });
            console.log(`📝 Report created with ID: ${report.id} (Status: processing)`);

            // 3. Simulate ML Processing
            console.log('⚙️  Running ML Analysis (Simulation)...');
            await new Promise(r => setTimeout(r, 5000));

            // 4. Generate Result
            const findings = `Automated analysis for ${data.modality || 'Medical'} scan.\n\nOBSERVATIONS:\n- No acute fracture or dislocation.\n- Soft tissues appear normal.\n- No pneumothorax or pleural effusion.\n\n(Note: This is an automated preliminary report)`;
            const impression = "Normal study. No significant abnormalities detected.";

            // 5. Update Report
            report.findings = findings;
            report.impression = impression;
            report.confidence_score = 0.95;
            report.report_status = "completed";
            await report.save();

            console.log(`✅ Analysis complete for ${data.scanId}`);
            console.log(`📊 Report updated: ${report.id} (Status: completed)`);

        } catch (error) {
            console.error('❌ Processing failed:', error);
        }
    });

    // Also consume Priority Queue if needed
    // rabbitmq.consume('priority_queue', async (data, msg) => {
    //     console.log(`\n📦 [Priority Task] Patient: ${data.patient_name} (${data.patient_id})`);
    //     // Prioritization logic simulation
    //     console.log('✅ Prioritization Check: Normal Priority');
    // });
}

startWorker();

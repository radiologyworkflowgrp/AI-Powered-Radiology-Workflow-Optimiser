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

    // X-Ray Analysis Queue - Handled by Python Worker (start_xray_worker.py)
    // The Python worker loads best.pt DuoFormer model and processes X-ray scans
    // DO NOT consume here - let Python worker handle it for real AI predictions

    /*
    rabbitmq.consume('xray_analysis_queue', async (data, msg) => {
        console.log(`\n📦 [Received X-Ray Task] Scan: ${data.scan_id} | Patient: ${data.patient_id}`);
        // This is now handled by Python worker with actual DuoFormer model
    });
    */

    console.log('✅ ML Workers ready. X-Ray and MRI processing delegated to Python workers.');

    // Consume X-Ray Results from Python Worker
    rabbitmq.consume('xray_results_queue', async (data, msg) => {
        console.log(`\n📥 [Received X-Ray Result from Python] Patient: ${data.patient_id}`);

        try {
            const { Patient: PatientPG } = require('../models');

            // Get patient name
            let patientName = data.patient_name || "Unknown";
            try {
                const patient = await PatientPG.findByPk(data.patient_id);
                if (patient) patientName = patient.name;
            } catch (e) { }

            // Format findings from Python worker
            const findingsArray = data.findings || [];
            let findingsText = `Automated chest X-ray analysis using ${data.ml_model}.\n\nFINDINGS:\n`;

            findingsArray.forEach(finding => {
                findingsText += `- Location: ${finding.location}\n`;
                findingsText += `  Finding: ${finding.finding}\n`;
                findingsText += `  Severity: ${finding.severity}\n`;
                findingsText += `  Confidence: ${(finding.confidence * 100).toFixed(1)}%\n\n`;
            });

            findingsText += `\nPREDICTIONS (CheXpert Labels):\n`;
            const predictions = data.predictions || [];
            predictions.forEach(pred => {
                if (pred.positive) {
                    findingsText += `✓ ${pred.label}: ${(pred.probability * 100).toFixed(1)}%\n`;
                }
            });

            findingsText += `\nTECHNICAL NOTE:\n`;
            findingsText += `- Scan ID: ${data.scan_id}\n`;
            findingsText += `- Model: ${data.ml_model}\n`;
            findingsText += `- Overall Confidence: ${(data.confidence_score * 100).toFixed(1)}%\n`;

            const impression = `Chest X-ray analysis completed with ${(data.confidence_score * 100).toFixed(1)}% confidence.`;
            const recommendation = findingsArray.length > 1
                ? "Abnormalities detected. Clinical correlation and specialist consultation recommended."
                : "No significant abnormalities detected. Follow up as clinically indicated.";

            // Create report in database
            const report = await MLReport.create({
                patient_id: data.patient_id,
                doctor_id: data.doctor_id || null,
                patient_name: patientName,
                report_type: "Chest X-Ray",
                ml_model: data.ml_model,
                report_status: "completed",
                confidence_score: data.confidence_score,
                findings: findingsText,
                impression: impression,
                recommendation: recommendation,
                created_at: new Date(),
                updated_at: new Date(),
                report_data: {
                    scan_id: data.scan_id,
                    predictions: predictions
                }
            });

            console.log(`✅ X-Ray Report saved to database: ID ${report.id}`);
            console.log(`📊 Confidence: ${(data.confidence_score * 100).toFixed(1)}%`);

        } catch (error) {
            console.error('❌ Error saving X-Ray result:', error);
            console.error(error.stack);
        }
    });


    // MRI Analysis Queue - Handled by Python Worker (start_mri_worker.py)
    // The Python worker loads best_mri_model.pth and processes MRI scans
    // DO NOT consume here - let Python worker handle it for real AI predictions

    /*
    rabbitmq.consume('mri_analysis_queue', async (data, msg) => {
        console.log(`\n📦 [Received MRI Task] Scan: ${data.scan_id} | Patient: ${data.patient_id}`);
        // This is now handled by Python worker with actual model
    });
    */

    console.log('✅ X-Ray worker ready. MRI processing delegated to Python worker.');

    // Consume MRI Results from Python Worker
    rabbitmq.consume('mri_results_queue', async (data, msg) => {
        console.log(`\n📥 [Received MRI Result from Python] Patient: ${data.patient_id}`);

        try {
            const { Patient: PatientPG } = require('../models');

            // Get patient name
            let patientName = data.patient_name || "Unknown";
            try {
                const patient = await PatientPG.findByPk(data.patient_id);
                if (patient) patientName = patient.name;
            } catch (e) { }

            // Format findings from Python worker
            const findingsArray = data.findings || [];
            let findingsText = `Automated brain MRI analysis using ${data.ml_model}.\n\nFINDINGS:\n`;

            findingsArray.forEach(finding => {
                findingsText += `- Location: ${finding.location}\n`;
                findingsText += `  Finding: ${finding.finding}\n`;
                findingsText += `  Severity: ${finding.severity}\n`;
                findingsText += `  Confidence: ${(finding.confidence * 100).toFixed(1)}%\n\n`;
            });

            findingsText += `\nPREDICTIONS:\n`;
            const predictions = data.predictions || [];
            predictions.forEach(pred => {
                const marker = pred.positive ? '✓' : ' ';
                findingsText += `[${marker}] ${pred.label}: ${(pred.probability * 100).toFixed(1)}%\n`;
            });

            findingsText += `\nTECHNICAL NOTE:\n`;
            findingsText += `- Scan ID: ${data.scan_id}\n`;
            findingsText += `- Model: ${data.ml_model}\n`;
            findingsText += `- Predicted Class: ${data.predicted_class}\n`;
            findingsText += `- Overall Confidence: ${(data.confidence_score * 100).toFixed(1)}%\n`;

            const impression = `${data.predicted_class} detected with ${(data.confidence_score * 100).toFixed(1)}% confidence.`;
            const recommendation = data.predicted_class === 'Notumor'
                ? "No tumor detected. Follow up as clinically indicated."
                : "Tumor detected. Immediate clinical correlation and specialist consultation recommended.";

            // Create report in database
            const report = await MLReport.create({
                patient_id: data.patient_id,
                doctor_id: data.doctor_id || null,
                patient_name: patientName,
                report_type: "Brain MRI",
                ml_model: data.ml_model,
                report_status: "completed",
                confidence_score: data.confidence_score,
                findings: findingsText,
                impression: impression,
                recommendation: recommendation,
                created_at: new Date(),
                updated_at: new Date(),
                report_data: {
                    scan_id: data.scan_id,
                    predicted_class: data.predicted_class,
                    predictions: predictions
                }
            });

            console.log(`✅ MRI Report saved to database: ID ${report.id}`);
            console.log(`📊 Prediction: ${data.predicted_class} (${(data.confidence_score * 100).toFixed(1)}%)`);

        } catch (error) {
            console.error('❌ Error saving MRI result:', error);
            console.error(error.stack);
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

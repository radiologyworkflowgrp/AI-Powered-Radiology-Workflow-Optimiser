const { RadiologyResult, Patient } = require('./models');
require('dotenv').config();

const createSampleResults = async () => {
    try {
        // Get the patient we created
        const patient = await Patient.findOne({ where: { email: 'patient@hospital.com' } });

        if (!patient) {
            console.log('❌ Patient not found. Please run create-patient.js first.');
            process.exit(1);
        }

        console.log(`✅ Found patient: ${patient.name} (ID: ${patient.id})`);

        // Create sample radiology results
        const sampleResults = [
            {
                patientId: patient.id,
                patientName: patient.name,
                testType: 'Chest X-Ray',
                result: 'Normal chest X-ray. No acute cardiopulmonary abnormality.',
                notes: 'Patient presented with mild cough. Imaging shows clear lung fields.'
            },
            {
                patientId: patient.id,
                patientName: patient.name,
                testType: 'Brain MRI',
                result: 'No evidence of acute intracranial abnormality. Normal brain MRI.',
                notes: 'Routine follow-up scan. No significant changes from previous study.'
            },
            {
                patientId: patient.id,
                patientName: patient.name,
                testType: 'Abdominal CT Scan',
                result: 'Unremarkable abdominal CT. No acute findings.',
                notes: 'Scan performed for abdominal pain evaluation. All organs appear normal.'
            }
        ];

        for (const result of sampleResults) {
            await RadiologyResult.create(result);
            console.log(`✅ Created ${result.testType} result`);
        }

        console.log('\n🎉 Sample radiology results created successfully!');
        console.log('📊 Total results created: 3');
        console.log('\nYou can now view these results in the Radiology Results section.');

        process.exit(0);
    } catch (error) {
        console.error('❌ Error creating sample results:', error);
        process.exit(1);
    }
};

createSampleResults();

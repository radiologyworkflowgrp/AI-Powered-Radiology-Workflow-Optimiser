const { Prescription } = require('./models');
require('dotenv').config();

const checkPrescriptions = async () => {
    try {
        console.log('🔍 Checking all prescriptions...\n');

        const prescriptions = await Prescription.findAll({
            order: [['datePrescribed', 'DESC']],
            limit: 10
        });

        if (prescriptions.length === 0) {
            console.log('❌ No prescriptions found in database');
        } else {
            console.log(`✅ Found ${prescriptions.length} prescriptions:\n`);
            prescriptions.forEach((rx, index) => {
                console.log(`Prescription ${index + 1}:`);
                console.log(`  ID: ${rx.id}`);
                console.log(`  Patient: ${rx.patientName}`);
                console.log(`  Medication: ${rx.medication}`);
                console.log(`  Dosage: ${rx.dosage}`);
                console.log(`  Prescribed By: ${rx.prescribedBy}`);
                console.log(`  Date: ${rx.datePrescribed}`);
                console.log(`  Status: ${rx.status}`);
                console.log('');
            });
        }

        process.exit(0);
    } catch (error) {
        console.error('❌ Error:', error);
        process.exit(1);
    }
};

checkPrescriptions();

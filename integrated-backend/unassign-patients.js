const { Patient } = require('./models');
require('dotenv').config();

const unassignPatients = async () => {
    try {
        console.log('🔄 Removing all patient-doctor assignments...');

        // Set assignedDoctor to null for all patients
        const result = await Patient.update(
            { assignedDoctor: null },
            { where: {} }
        );

        console.log(`✅ Unassigned ${result[0]} patients from doctors`);
        console.log('✅ All patients are now unassigned');

        process.exit(0);
    } catch (error) {
        console.error('❌ Error unassigning patients:', error);
        process.exit(1);
    }
};

unassignPatients();

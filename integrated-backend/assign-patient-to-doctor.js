const { Patient, Doctor } = require('./models');
require('dotenv').config();

const assignPatientToDoctor = async () => {
    try {
        // Get the doctor
        const doctor = await Doctor.findOne({ where: { email: 'doctor@hospital.com' } });
        if (!doctor) {
            console.log('❌ Doctor not found');
            process.exit(1);
        }

        // Get the patient
        const patient = await Patient.findOne({ where: { email: 'patient@hospital.com' } });
        if (!patient) {
            console.log('❌ Patient not found');
            process.exit(1);
        }

        // Assign patient to doctor
        await patient.update({
            assignedDoctor: {
                id: doctor.id,
                name: doctor.name
            }
        });

        console.log(`✅ Assigned patient "${patient.name}" to doctor "${doctor.name}"`);
        console.log(`\nNow when you log in as doctor@hospital.com, you will see:`);
        console.log(`- Patient: ${patient.name}`);
        console.log(`- All radiology results for this patient`);

        process.exit(0);
    } catch (error) {
        console.error('❌ Error:', error);
        process.exit(1);
    }
};

assignPatientToDoctor();

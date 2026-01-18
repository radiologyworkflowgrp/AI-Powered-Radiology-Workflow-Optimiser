const { MLReport } = require('./models');
require('dotenv').config();

const checkDoctorIds = async () => {
    try {
        console.log('🔍 Checking doctor IDs in system...\n');

        const { Doctor } = require('./models');
        const doctors = await Doctor.findAll();

        console.log('Doctors in database:');
        doctors.forEach(doc => {
            console.log(`  - ${doc.name}: ID = ${doc.id}`);
        });

        console.log('\n📋 Recent ML Reports:');
        const reports = await MLReport.findAll({
            order: [['created_at', 'DESC']],
            limit: 3
        });

        reports.forEach(r => {
            console.log(`  Report ${r.id}: doctor_id = ${r.doctor_id || 'NULL'}, patient = ${r.patient_name}`);
        });

        process.exit(0);
    } catch (error) {
        console.error('❌ Error:', error);
        process.exit(1);
    }
};

checkDoctorIds();

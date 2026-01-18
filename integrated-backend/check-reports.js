const { MLReport } = require('./models');
require('dotenv').config();

const checkReports = async () => {
    try {
        console.log('🔍 Checking all ML reports...\n');

        const reports = await MLReport.findAll({
            order: [['created_at', 'DESC']],
            limit: 10
        });

        if (reports.length === 0) {
            console.log('❌ No ML reports found in database');
            console.log('   Upload a DICOM scan to create reports');
        } else {
            console.log(`✅ Found ${reports.length} reports:\n`);
            reports.forEach((report, index) => {
                console.log(`Report ${index + 1}:`);
                console.log(`  ID: ${report.id}`);
                console.log(`  Patient: ${report.patient_name}`);
                console.log(`  Type: ${report.report_type}`);
                console.log(`  Doctor ID: ${report.doctor_id || 'NULL ❌'}`);
                console.log(`  Status: ${report.report_status}`);
                console.log(`  Created: ${report.created_at}`);
                console.log('');
            });
        }

        process.exit(0);
    } catch (error) {
        console.error('❌ Error:', error);
        process.exit(1);
    }
};

checkReports();

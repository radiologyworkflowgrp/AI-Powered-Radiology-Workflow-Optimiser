const { MLReport, RadiologyResult } = require('./models');
require('dotenv').config();

const deleteAllReports = async () => {
    try {
        console.log('🗑️  Deleting all reports...');

        // Delete ML Reports
        const mlDeleted = await MLReport.destroy({ where: {}, truncate: true });
        console.log(`✅ Deleted ${mlDeleted} ML reports`);

        // Delete Radiology Results
        const radDeleted = await RadiologyResult.destroy({ where: {}, truncate: true });
        console.log(`✅ Deleted ${radDeleted} radiology results`);

        console.log('✅ All reports deleted successfully!');
        process.exit(0);
    } catch (error) {
        console.error('❌ Error deleting reports:', error);
        process.exit(1);
    }
};

deleteAllReports();

const { sequelize } = require('./postgres');

async function fixSchema() {
    try {
        await sequelize.authenticate();
        console.log('✅ Connected to PostgreSQL');

        // Drop FK constraint on dicom_images
        try {
            await sequelize.query('ALTER TABLE "dicom_images" DROP CONSTRAINT IF EXISTS "dicom_images_patient_id_fkey"');
            console.log('✅ Dropped FK dicom_images_patient_id_fkey');
        } catch (e) {
            console.log('ℹ️ Error dropping FK dicom_images_patient_id_fkey:', e.message);
        }

        // Drop FK constraint on ml_reports
        try {
            await sequelize.query('ALTER TABLE "ml_reports" DROP CONSTRAINT IF EXISTS "ml_reports_patient_id_fkey"');
            console.log('✅ Dropped FK ml_reports_patient_id_fkey');
        } catch (e) {
            console.log('ℹ️ Error dropping FK ml_reports_patient_id_fkey:', e.message);
        }

        try {
            await sequelize.query('ALTER TABLE "ml_reports" DROP CONSTRAINT IF EXISTS "ml_reports_doctor_id_fkey"');
            console.log('✅ Dropped FK ml_reports_doctor_id_fkey');
        } catch (e) {
            console.log('ℹ️ Error dropping FK ml_reports_doctor_id_fkey:', e.message);
        }

        // Change column types to VARCHAR
        try {
            await sequelize.query('ALTER TABLE "dicom_images" ALTER COLUMN "patient_id" TYPE VARCHAR(255) USING patient_id::varchar');
            console.log('✅ Changed dicom_images.patient_id to VARCHAR');
        } catch (e) {
            console.log('ℹ️ Error altering dicom_images.patient_id:', e.message);
        }

        try {
            await sequelize.query('ALTER TABLE "ml_reports" ALTER COLUMN "patient_id" TYPE VARCHAR(255) USING patient_id::varchar');
            console.log('✅ Changed ml_reports.patient_id to VARCHAR');
        } catch (e) {
            console.log('ℹ️ Error altering ml_reports.patient_id:', e.message);
        }

        try {
            await sequelize.query('ALTER TABLE "ml_reports" ALTER COLUMN "doctor_id" TYPE VARCHAR(255) USING doctor_id::varchar');
            console.log('✅ Changed ml_reports.doctor_id to VARCHAR');
        } catch (e) {
            console.log('ℹ️ Error altering ml_reports.doctor_id:', e.message);
        }

        console.log('schema fix complete');
        process.exit(0);

    } catch (error) {
        console.error('❌ Schema fix failed:', error);
        process.exit(1);
    }
}

fixSchema();

/**
 * Migration script to transfer data from MySQL to PostgreSQL
 * Run this after setting up PostgreSQL and before shutting down MySQL
 */

require('dotenv').config();
const mysql = require('mysql2/promise');
const { sequelize, MLReport } = require('../models');

async function migrateMySQLToPostgres() {
    console.log('🔄 Starting MySQL to PostgreSQL migration...\n');

    let mysqlConnection;

    try {
        // Connect to MySQL
        console.log('📡 Connecting to MySQL...');
        mysqlConnection = await mysql.createConnection({
            host: process.env.MYSQL_HOST || 'localhost',
            user: process.env.MYSQL_USER || 'appuser',
            password: process.env.MYSQL_PASSWORD || 'AppUser123!',
            database: process.env.MYSQL_DATABASE || 'radiology_hospital'
        });
        console.log('✅ Connected to MySQL\n');

        // Connect to PostgreSQL
        console.log('📡 Connecting to PostgreSQL...');
        await sequelize.authenticate();
        console.log('✅ Connected to PostgreSQL\n');

        // Fetch all ML reports from MySQL
        console.log('📊 Fetching ML reports from MySQL...');
        const [rows] = await mysqlConnection.query('SELECT * FROM ml_reports');
        console.log(`✅ Found ${rows.length} ML reports\n`);

        if (rows.length === 0) {
            console.log('ℹ️  No data to migrate from MySQL');
            return;
        }

        // Migrate each report
        console.log('🔄 Migrating ML reports to PostgreSQL...');
        let successCount = 0;
        let errorCount = 0;

        for (const row of rows) {
            try {
                await MLReport.create({
                    id: row.id,
                    patient_id: row.patient_id,
                    patient_name: row.patient_name,
                    report_type: row.report_type,
                    ml_model: row.ml_model || 'Unknown',
                    confidence_score: row.confidence_score,
                    findings: row.findings,
                    impression: row.impression,
                    recommendation: row.recommendation,
                    image_url: row.image_url,
                    report_status: row.report_status || 'pending',
                    report_data: row.report_data ? JSON.parse(row.report_data) : null,
                    pdf_path: row.pdf_path,
                    doctor_id: row.doctor_id,
                    reviewed_by: row.reviewed_by,
                    created_at: row.created_at,
                    updated_at: row.updated_at
                });
                successCount++;
                if (successCount % 10 === 0) {
                    console.log(`   Migrated ${successCount}/${rows.length} reports...`);
                }
            } catch (error) {
                console.error(`   ❌ Error migrating report ID ${row.id}:`, error.message);
                errorCount++;
            }
        }

        console.log(`\n✅ Migration complete!`);
        console.log(`   Success: ${successCount} reports`);
        console.log(`   Errors: ${errorCount} reports`);

        // Verify counts match
        const postgresCount = await MLReport.count();
        console.log(`\n📊 Verification:`);
        console.log(`   MySQL count: ${rows.length}`);
        console.log(`   PostgreSQL count: ${postgresCount}`);

        if (postgresCount === rows.length) {
            console.log(`   ✅ Counts match!`);
        } else {
            console.log(`   ⚠️  Warning: Counts don't match. Please investigate.`);
        }

    } catch (error) {
        console.error('\n❌ Migration failed:', error.message);
        console.error(error.stack);
        process.exit(1);
    } finally {
        // Close connections
        if (mysqlConnection) {
            await mysqlConnection.end();
            console.log('\n✅ MySQL connection closed');
        }
        await sequelize.close();
        console.log('✅ PostgreSQL connection closed');
    }
}

// Run migration
migrateMySQLToPostgres()
    .then(() => {
        console.log('\n🎉 MySQL to PostgreSQL migration completed successfully!');
        process.exit(0);
    })
    .catch((error) => {
        console.error('\n💥 Migration failed:', error);
        process.exit(1);
    });

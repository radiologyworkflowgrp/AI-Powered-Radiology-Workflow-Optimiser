# PostgreSQL Migration Guide

This guide explains how to migrate your radiology workflow system from MySQL + MongoDB to PostgreSQL with DICOM image storage.

## Prerequisites

- PostgreSQL 12 or higher installed
- Backup of existing MySQL and MongoDB databases
- Node.js dependencies installed (`npm install`)

## Step 1: Setup PostgreSQL

Run the setup script to create the database and user:

```bash
cd /home/lnoob777/psproject/AI-Powered-Radiology-Workflow-Optimiser/integrated-backend
chmod +x migrations/setup-postgres.sh
./migrations/setup-postgres.sh
```

This will:
- Create PostgreSQL user `radiology_user`
- Create database `radiology_hospital`
- Grant necessary privileges

## Step 2: Initialize Database Schema

The schema will be automatically created when you first run the application with PostgreSQL. To manually initialize:

```bash
node -e "require('./postgres').initDatabase().then(() => process.exit(0))"
```

This creates all tables with proper indexes and relationships.

## Step 3: Migrate Data from MySQL

Transfer ML reports from MySQL to PostgreSQL:

```bash
node migrations/migrate-mysql-to-postgres.js
```

This script will:
- Connect to both MySQL and PostgreSQL
- Copy all `ml_reports` data
- Verify row counts match
- Report any errors

## Step 4: Migrate Data from MongoDB

Transfer all collections from MongoDB to PostgreSQL:

```bash
node migrations/migrate-mongodb-to-postgres.js
```

This script migrates:
- Patients
- Doctors
- Admins
- Prescriptions
- Notes
- Radiology Results
- Activity Logs
- Login Activities
- Jobs

## Step 5: Verify Migration

Check that all data was migrated successfully:

```sql
-- Connect to PostgreSQL
psql -U radiology_user -d radiology_hospital

-- Check table counts
SELECT 'patients' as table_name, COUNT(*) as count FROM patients
UNION ALL
SELECT 'doctors', COUNT(*) FROM doctors
UNION ALL
SELECT 'ml_reports', COUNT(*) FROM ml_reports
UNION ALL
SELECT 'dicom_images', COUNT(*) FROM dicom_images;
```

## Step 6: Update Application Configuration

The `.env` file has already been updated with PostgreSQL configuration. Verify the settings:

```env
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=radiology_user
POSTGRES_PASSWORD=SecurePassword123!
POSTGRES_DATABASE=radiology_hospital
```

## Step 7: Test the Application

Start the server with PostgreSQL:

```bash
npm start
```

The application will now use PostgreSQL for all database operations.

## Step 8: Test DICOM Upload

Test uploading a DICOM file:

```bash
curl -X POST http://localhost:3002/api/dicom/upload \
  -F "dicomFile=@/path/to/sample.dcm" \
  -F "patientId=<patient-uuid>" \
  -F "scanType=CT"
```

## Step 9: Decommission Old Databases (Optional)

Once you've verified everything works:

1. **Backup** MySQL and MongoDB one final time
2. Stop MySQL and MongoDB services
3. Archive the data directories
4. Remove old database connection files:
   - `mysql.js` (replaced by `postgres.js`)
   - `db.js` (MongoDB connection)

## Troubleshooting

### Connection Errors

If you get connection errors:

```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Check connection
psql -U radiology_user -d radiology_hospital -c "SELECT 1"
```

### Migration Errors

If migration fails:
1. Check the error messages in the console
2. Verify source databases are accessible
3. Check PostgreSQL has enough disk space
4. Review PostgreSQL logs: `sudo tail -f /var/log/postgresql/postgresql-*.log`

### Schema Issues

If tables aren't created:

```bash
# Force schema sync (WARNING: This will drop existing tables)
node -e "require('./models').sequelize.sync({ force: true }).then(() => process.exit(0))"
```

## DICOM Storage

DICOM images are now stored directly in PostgreSQL:

- **Table**: `dicom_images`
- **Storage**: Binary data in `image_data` column (BLOB)
- **Metadata**: JSONB column for DICOM tags
- **Thumbnails**: Optional preview images

### Querying DICOM Images

```sql
-- List all DICOM images for a patient
SELECT id, scan_id, modality, study_date, 
       pg_size_pretty(length(image_data)) as size
FROM dicom_images
WHERE patient_id = '<patient-uuid>';

-- Get total storage used
SELECT pg_size_pretty(SUM(length(image_data))) as total_size
FROM dicom_images;
```

## Performance Considerations

For large DICOM datasets:

1. **Indexing**: Indexes are automatically created on frequently queried columns
2. **Connection Pooling**: Configured for up to 20 connections
3. **TOAST**: PostgreSQL automatically compresses large BLOB data
4. **Partitioning**: Consider table partitioning for >100GB of DICOM data

## Rollback Plan

If you need to rollback to MySQL/MongoDB:

1. Stop the new PostgreSQL-based application
2. Restore the `.env` file to use MySQL/MongoDB
3. Restart with the old configuration
4. Your backup data remains intact

## Next Steps

- Monitor PostgreSQL performance
- Set up automated backups
- Configure replication if needed
- Optimize queries based on usage patterns

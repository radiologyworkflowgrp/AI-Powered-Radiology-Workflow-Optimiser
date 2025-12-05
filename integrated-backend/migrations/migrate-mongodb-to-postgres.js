/**
 * Migration script to transfer data from MongoDB to PostgreSQL
 * Run this after setting up PostgreSQL and before shutting down MongoDB
 */

require('dotenv').config();
const mongoose = require('mongoose');
const {
    sequelize,
    Patient,
    Doctor,
    Admin,
    Prescription,
    Note,
    RadiologyResult,
    ActivityLog,
    LoginActivity,
    Job
} = require('../models');

// MongoDB connection
const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017/radiology_hospital';

// Define MongoDB schemas (simplified versions for migration)
const PatientSchema = new mongoose.Schema({}, { strict: false });
const DoctorSchema = new mongoose.Schema({}, { strict: false });
const AdminSchema = new mongoose.Schema({}, { strict: false });
const PrescriptionSchema = new mongoose.Schema({}, { strict: false });
const NoteSchema = new mongoose.Schema({}, { strict: false });
const RadiologyResultSchema = new mongoose.Schema({}, { strict: false });
const ActivityLogSchema = new mongoose.Schema({}, { strict: false });
const LoginActivitySchema = new mongoose.Schema({}, { strict: false });
const JobSchema = new mongoose.Schema({}, { strict: false });

async function migrateMongoDBToPostgres() {
    console.log('🔄 Starting MongoDB to PostgreSQL migration...\n');

    try {
        // Connect to MongoDB
        console.log('📡 Connecting to MongoDB...');
        await mongoose.connect(MONGODB_URI);
        console.log('✅ Connected to MongoDB\n');

        // Connect to PostgreSQL
        console.log('📡 Connecting to PostgreSQL...');
        await sequelize.authenticate();
        console.log('✅ Connected to PostgreSQL\n');

        // Get MongoDB models
        const MongoPatient = mongoose.model('Patient', PatientSchema);
        const MongoDoctor = mongoose.model('Doctor', DoctorSchema);
        const MongoAdmin = mongoose.model('Admin', AdminSchema);
        const MongoPrescription = mongoose.model('Prescription', PrescriptionSchema);
        const MongoNote = mongoose.model('Note', NoteSchema);
        const MongoRadiologyResult = mongoose.model('RadiologyResult', RadiologyResultSchema);
        const MongoActivityLog = mongoose.model('ActivityLog', ActivityLogSchema);
        const MongoLoginActivity = mongoose.model('LoginActivity', LoginActivitySchema);
        const MongoJob = mongoose.model('Job', JobSchema);

        // Migrate Patients
        console.log('👤 Migrating Patients...');
        const patients = await MongoPatient.find({});
        console.log(`   Found ${patients.length} patients`);

        for (const patient of patients) {
            try {
                await Patient.create({
                    id: patient._id.toString(),
                    name: patient.name,
                    email: patient.email,
                    password: patient.password, // Already hashed
                    age: patient.age,
                    gender: patient.gender,
                    contact: patient.contact,
                    address: patient.address,
                    medical_history: patient.medical_history,
                    symptoms: patient.symptoms || [],
                    vitals: patient.vitals || {},
                    priority: patient.priority || 'normal',
                    assignedDoctor: patient.assignedDoctor || null,
                    profileCompleted: patient.profileCompleted || false,
                    created_at: patient.createdAt || new Date(),
                    updated_at: patient.updatedAt || new Date()
                });
            } catch (error) {
                console.error(`   ❌ Error migrating patient ${patient._id}:`, error.message);
            }
        }
        console.log(`   ✅ Migrated ${patients.length} patients\n`);

        // Migrate Doctors
        console.log('👨‍⚕️ Migrating Doctors...');
        const doctors = await MongoDoctor.find({});
        console.log(`   Found ${doctors.length} doctors`);

        for (const doctor of doctors) {
            try {
                await Doctor.create({
                    id: doctor._id.toString(),
                    name: doctor.name,
                    email: doctor.email,
                    password: doctor.password, // Already hashed
                    specialty: doctor.specialty || 'General Medicine',
                    availability: doctor.availability || 'Available',
                    profileCompleted: doctor.profileCompleted || false,
                    created_at: doctor.createdAt || new Date(),
                    updated_at: doctor.updatedAt || new Date()
                });
            } catch (error) {
                console.error(`   ❌ Error migrating doctor ${doctor._id}:`, error.message);
            }
        }
        console.log(`   ✅ Migrated ${doctors.length} doctors\n`);

        // Migrate Admins
        console.log('👔 Migrating Admins...');
        const admins = await MongoAdmin.find({});
        console.log(`   Found ${admins.length} admins`);

        for (const admin of admins) {
            try {
                await Admin.create({
                    id: admin._id.toString(),
                    name: admin.name,
                    email: admin.email,
                    password: admin.password, // Already hashed
                    role: admin.role || 'admin',
                    profileCompleted: admin.profileCompleted || true,
                    created_at: admin.createdAt || new Date(),
                    updated_at: admin.updatedAt || new Date()
                });
            } catch (error) {
                console.error(`   ❌ Error migrating admin ${admin._id}:`, error.message);
            }
        }
        console.log(`   ✅ Migrated ${admins.length} admins\n`);

        // Migrate Prescriptions
        console.log('💊 Migrating Prescriptions...');
        const prescriptions = await MongoPrescription.find({});
        console.log(`   Found ${prescriptions.length} prescriptions`);

        for (const prescription of prescriptions) {
            try {
                await Prescription.create({
                    id: prescription._id.toString(),
                    patientName: prescription.patientName,
                    medication: prescription.medication,
                    dosage: prescription.dosage,
                    frequency: prescription.frequency,
                    duration: prescription.duration,
                    status: prescription.status || 'active',
                    refillCount: prescription.refillCount || 0,
                    created_at: prescription.createdAt || new Date(),
                    updated_at: prescription.updatedAt || new Date()
                });
            } catch (error) {
                console.error(`   ❌ Error migrating prescription ${prescription._id}:`, error.message);
            }
        }
        console.log(`   ✅ Migrated ${prescriptions.length} prescriptions\n`);

        // Migrate Notes
        console.log('📝 Migrating Notes...');
        const notes = await MongoNote.find({});
        console.log(`   Found ${notes.length} notes`);

        for (const note of notes) {
            try {
                await Note.create({
                    id: note._id.toString(),
                    title: note.title,
                    content: note.content,
                    category: note.category,
                    tags: note.tags || [],
                    created_at: note.createdAt || new Date(),
                    updated_at: note.updatedAt || new Date()
                });
            } catch (error) {
                console.error(`   ❌ Error migrating note ${note._id}:`, error.message);
            }
        }
        console.log(`   ✅ Migrated ${notes.length} notes\n`);

        // Migrate Radiology Results
        console.log('🔬 Migrating Radiology Results...');
        const radiologyResults = await MongoRadiologyResult.find({});
        console.log(`   Found ${radiologyResults.length} radiology results`);

        for (const result of radiologyResults) {
            try {
                await RadiologyResult.create({
                    id: result._id.toString(),
                    patientId: result.patientId,
                    patientName: result.patientName,
                    testType: result.testType,
                    result: result.result,
                    notes: result.notes,
                    imageUrl: result.imageUrl,
                    dicom_image_id: null, // Will be linked later if needed
                    created_at: result.createdAt || new Date(),
                    updated_at: result.updatedAt || new Date()
                });
            } catch (error) {
                console.error(`   ❌ Error migrating radiology result ${result._id}:`, error.message);
            }
        }
        console.log(`   ✅ Migrated ${radiologyResults.length} radiology results\n`);

        // Migrate Activity Logs
        console.log('📋 Migrating Activity Logs...');
        const activityLogs = await MongoActivityLog.find({});
        console.log(`   Found ${activityLogs.length} activity logs`);

        for (const log of activityLogs) {
            try {
                await ActivityLog.create({
                    id: log._id.toString(),
                    action: log.action,
                    description: log.description,
                    entityType: log.entityType,
                    entityId: log.entityId,
                    metadata: log.metadata || {},
                    created_at: log.createdAt || new Date(),
                    updated_at: log.updatedAt || new Date()
                });
            } catch (error) {
                console.error(`   ❌ Error migrating activity log ${log._id}:`, error.message);
            }
        }
        console.log(`   ✅ Migrated ${activityLogs.length} activity logs\n`);

        // Migrate Login Activities
        console.log('🔐 Migrating Login Activities...');
        const loginActivities = await MongoLoginActivity.find({});
        console.log(`   Found ${loginActivities.length} login activities`);

        for (const activity of loginActivities) {
            try {
                await LoginActivity.create({
                    id: activity._id.toString(),
                    userId: activity.userId,
                    role: activity.role,
                    email: activity.email,
                    ipAddress: activity.ipAddress,
                    userAgent: activity.userAgent,
                    created_at: activity.createdAt || new Date(),
                    updated_at: activity.updatedAt || new Date()
                });
            } catch (error) {
                console.error(`   ❌ Error migrating login activity ${activity._id}:`, error.message);
            }
        }
        console.log(`   ✅ Migrated ${loginActivities.length} login activities\n`);

        // Migrate Jobs
        console.log('⚙️  Migrating Jobs...');
        const jobs = await MongoJob.find({});
        console.log(`   Found ${jobs.length} jobs`);

        for (const job of jobs) {
            try {
                await Job.create({
                    id: job._id.toString(),
                    type: job.type,
                    status: job.status || 'pending',
                    data: job.data || {},
                    result: job.result || {},
                    created_at: job.createdAt || new Date(),
                    updated_at: job.updatedAt || new Date()
                });
            } catch (error) {
                console.error(`   ❌ Error migrating job ${job._id}:`, error.message);
            }
        }
        console.log(`   ✅ Migrated ${jobs.length} jobs\n`);

        // Summary
        console.log('📊 Migration Summary:');
        console.log(`   Patients: ${await Patient.count()}`);
        console.log(`   Doctors: ${await Doctor.count()}`);
        console.log(`   Admins: ${await Admin.count()}`);
        console.log(`   Prescriptions: ${await Prescription.count()}`);
        console.log(`   Notes: ${await Note.count()}`);
        console.log(`   Radiology Results: ${await RadiologyResult.count()}`);
        console.log(`   Activity Logs: ${await ActivityLog.count()}`);
        console.log(`   Login Activities: ${await LoginActivity.count()}`);
        console.log(`   Jobs: ${await Job.count()}`);

    } catch (error) {
        console.error('\n❌ Migration failed:', error.message);
        console.error(error.stack);
        process.exit(1);
    } finally {
        // Close connections
        await mongoose.connection.close();
        console.log('\n✅ MongoDB connection closed');
        await sequelize.close();
        console.log('✅ PostgreSQL connection closed');
    }
}

// Run migration
migrateMongoDBToPostgres()
    .then(() => {
        console.log('\n🎉 MongoDB to PostgreSQL migration completed successfully!');
        process.exit(0);
    })
    .catch((error) => {
        console.error('\n💥 Migration failed:', error);
        process.exit(1);
    });

const { sequelize } = require('../postgres');

// Import all model definitions
const PatientModel = require('./Patient');
const DoctorModel = require('./Doctor');
const AdminModel = require('./Admin');
const MLReportModel = require('./MLReport');
const DICOMImageModel = require('./DICOMImage');
const PrescriptionModel = require('./Prescription');
const NoteModel = require('./Note');
const RadiologyResultModel = require('./RadiologyResult');
const ActivityLogModel = require('./ActivityLog');
const LoginActivityModel = require('./LoginActivity');
const JobModel = require('./Job');

// Initialize models
const Patient = PatientModel(sequelize);
const Doctor = DoctorModel(sequelize);
const Admin = AdminModel(sequelize);
const MLReport = MLReportModel(sequelize);
const DICOMImage = DICOMImageModel(sequelize);
const Prescription = PrescriptionModel(sequelize);
const Note = NoteModel(sequelize);
const RadiologyResult = RadiologyResultModel(sequelize);
const ActivityLog = ActivityLogModel(sequelize);
const LoginActivity = LoginActivityModel(sequelize);
const Job = JobModel(sequelize);

// Define associations/relationships
// Associations removed as Core Entities are in MongoDB
// Patient.hasMany(MLReport ...
// MLReport.belongsTo(Patient ...
// ...
// Relationships are now handled via string IDs (manual referencing)

// Export all models and sequelize instance
module.exports = {
    sequelize,
    Patient,
    Doctor,
    Admin,
    MLReport,
    DICOMImage,
    Prescription,
    Note,
    RadiologyResult,
    ActivityLog,
    LoginActivity,
    Job
};

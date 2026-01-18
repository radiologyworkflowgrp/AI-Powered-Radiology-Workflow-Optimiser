// Load environment variables first
require('dotenv').config();

console.log("Integrated Hospital Management Server is running...");
const express = require("express");
const cors = require("cors");
const bodyParser = require("body-parser");
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const cookieParser = require("cookie-parser");
const connectDB = require("./db");
const redisClient = require("./redisClient");
const { initDatabase, testConnection } = require("./postgres");
const mongoose = require("mongoose");

// MongoDB Models - ONLY for authentication
const User = require("./mongoSchemas/User"); // Auth User model
const LoginActivityMongo = require("./mongoSchemas/LoginActivity"); // Login tracking

// PostgreSQL Models - All business data
const {
  MLReport: MLReportModel,
  DICOMImage,
  Patient,
  Doctor,
  Prescription,
  Note,
  Admin,
  Job,
  RadiologyResult,
  ActivityLog,
  LoginActivity
} = require("./models");

// Import RabbitMQ, Logger, and Metrics
const rabbitmq = require("./rabbitmq");
const logger = require("./logger");
const metrics = require("./metrics");

// Import utilities
const { generateCredentials } = require("./utils/credentialGenerator");
const { extractUserInfo, getAllowedPatientIds, checkPatientAccess, requireAuth, requireAdmin } = require("./middleware/accessControl");

// Import routes
const authRoutes = require("./routes/authRoutes");
const catalogRoutes = require("./routes/catalogRoutes");
const mlReportsRoutes = require("./routes/mlReportsRoutes");
const dicomRoutes = require("./routes/dicomRoutes-postgres");
const viewerRoutes = require("./routes/viewerRoutes");
const userRoutes = require("./routes/userRoutes");
const prescriptionRoutes = require("./routes/prescriptionRoutes");

const app = express();
const PORT = 3002;

// Connect to MongoDB
connectDB();

// Initialize PostgreSQL database
initDatabase().catch(err => {
  logger.warn('PostgreSQL initialization failed:', err.message);
  logger.warn('ML reports functionality will be limited');
});
testConnection().catch(err => {
  logger.warn('PostgreSQL connection test failed:', err.message);
});

// Initialize RabbitMQ connection
rabbitmq.connect().then(() => {
  logger.info('✓ RabbitMQ connected and ready');
  metrics.setDatabaseConnection('rabbitmq', true);
}).catch(err => {
  logger.warn('RabbitMQ connection failed:', err.message);
  logger.warn('ML queue functionality will be limited');
  metrics.setDatabaseConnection('rabbitmq', false);
});

// Add logging middleware
app.use(logger.requestMiddleware());

// Add metrics middleware
app.use(metrics.middleware());

app.use(cors({
  origin: ["http://localhost:8080", "http://localhost:8081"], // Allow both frontend ports
  credentials: true, // Allow cookies to be sent
}));
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(cookieParser());
app.use(express.static("public"));
app.use('/uploads', express.static('uploads')); // Serve uploaded files
// Also serve from absolute path to be safe
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// Multer storage configuration
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    const uploadDir = 'uploads/';
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir);
    }
    cb(null, uploadDir);
  },
  filename: function (req, file, cb) {
    // Basic protection against directory traversal
    const safeSuffix = path.basename(file.originalname);
    cb(null, Date.now() + '-' + safeSuffix);
  }
});

const upload = multer({ storage: storage });

app.use((req, res, next) => {
  console.log(`[DEBUG] Request received: ${req.method} ${req.url}`);
  next();
});

const apiRouter = express.Router();

// Apply access control middleware to all API routes
apiRouter.use(extractUserInfo);

apiRouter.use((req, res, next) => {
  console.log(`[DEBUG] API Router received: ${req.method} ${req.url}`);
  next();
});

// Radiology Results Routes (Updated to include ML reports and doctor filtering)
apiRouter.get("/radiology-results", async (req, res) => {
  try {
    const { patientId } = req.query;

    // Get user from token (if available)
    let userId = null;
    let userRole = null;

    try {
      const token = req.cookies?.authToken || req.headers.authorization?.replace('Bearer ', '');
      console.log('🔍 DEBUG - Token received:', token ? 'YES' : 'NO');
      console.log('🔍 DEBUG - Cookies:', req.cookies);

      if (token) {
        const jwt = require('jsonwebtoken');
        const decoded = jwt.verify(token, process.env.JWT_SECRET || "your-secret-key-change-in-production");
        userId = decoded.userId;
        userRole = decoded.role;
        console.log('🔍 DEBUG - User Role:', userRole);
        console.log('🔍 DEBUG - User ID:', userId);
      } else {
        console.log('🔍 DEBUG - No token found, will show all results');
      }
    } catch (authError) {
      console.log('❌ No valid auth token, showing all results:', authError.message);
    }

    // If user is a doctor, filter by their assigned patients
    let allowedPatientIds = null;
    // For doctors, we need to use the referenceId (PostgreSQL doctor ID) not userId (MongoDB _id)
    const doctorProfileId = req.user?.referenceId || (req.cookies?.authToken || req.headers.authorization?.replace('Bearer ', '') ?
      (() => {
        try {
          const jwt = require('jsonwebtoken');
          const token = req.cookies?.authToken || req.headers.authorization?.replace('Bearer ', '');
          const decoded = jwt.verify(token, process.env.JWT_SECRET || "your-secret-key-change-in-production");
          return decoded.referenceId;
        } catch (e) { return null; }
      })() : null);

    if (userRole === 'doctor' && doctorProfileId) {
      // Get all patients assigned to this doctor (using PostgreSQL JSONB query)
      const { Op } = require('sequelize');
      const assignedPatients = await Patient.findAll({
        where: {
          assignedDoctor: {
            [Op.contains]: { id: doctorProfileId }
          }
        }
      });
      allowedPatientIds = assignedPatients.map(p => p.id);

      console.log(`✓ Doctor ${doctorProfileId} has ${allowedPatientIds.length} assigned patients`);
      console.log('🔍 DEBUG - Allowed Patient IDs:', allowedPatientIds);
    } else {
      console.log('🔍 DEBUG - Not a doctor or no doctorProfileId, allowedPatientIds:', allowedPatientIds);
    }

    // Get radiology results from PostgreSQL
    const { Op } = require('sequelize');
    let radiologyResults = [];
    if (patientId) {
      // Check if doctor is allowed to see this patient
      if (allowedPatientIds && !allowedPatientIds.includes(patientId)) {
        return res.json({
          message: "No results found - patient not assigned to you",
          results: [],
          total: 0
        });
      }
      radiologyResults = await RadiologyResult.findAll({
        where: { patientId: patientId }
      });
    } else {
      // Filter by allowed patients if doctor
      if (allowedPatientIds !== null) {
        if (allowedPatientIds.length === 0) {
          radiologyResults = [];
        } else {
          radiologyResults = await RadiologyResult.findAll({
            where: { patientId: { [Op.in]: allowedPatientIds } }
          });
        }
      } else {
        // Not a doctor (admin or no auth) - show all
        radiologyResults = await RadiologyResult.findAll();
      }
    }

    // Get ML reports from PostgreSQL (with error handling)
    let mlReports = [];
    try {
      if (patientId) {
        // Check if doctor is allowed to see this patient
        if (allowedPatientIds && !allowedPatientIds.includes(patientId)) {
          // Already returned above, but keep for consistency
          mlReports = [];
        } else {
          const reports = await MLReportModel.findAll({
            where: { patient_id: patientId },
            order: [['created_at', 'DESC']]
          });
          mlReports = reports.map(r => r.toJSON());
        }
      } else {
        // Show all ML reports to doctors (no filtering)
        if (allowedPatientIds !== null) {
          const reports = await MLReportModel.findAll({
            order: [['created_at', 'DESC']]
          });
          mlReports = reports.map(r => r.toJSON());
          console.log(`🔍 DEBUG - Showing all ${mlReports.length} ML reports to doctor`);
        } else if (userRole === 'patient' && userId) {
          // Patient is logged in - only show their own reports
          // We need to use the POSTGRES patient ID (referenceId), not the MongoDB userId
          const patientProfileId = req.user?.referenceId || (req.cookies?.authToken || req.headers.authorization?.replace('Bearer ', '') ?
            (() => {
              try {
                const jwt = require('jsonwebtoken');
                const token = req.cookies?.authToken || req.headers.authorization?.replace('Bearer ', '');
                const decoded = jwt.verify(token, process.env.JWT_SECRET || "your-secret-key-change-in-production");
                return decoded.referenceId;
              } catch (e) { return null; }
            })() : null);

          if (patientProfileId) {
            const reports = await MLReportModel.findAll({
              where: { patient_id: patientProfileId },
              order: [['created_at', 'DESC']]
            });
            mlReports = reports.map(r => r.toJSON());
          } else {
            mlReports = [];
          }
        } else {
          // not a doctor AND not a patient (admin) - show all
          const reports = await MLReportModel.findAll({
            order: [['created_at', 'DESC']]
          });
          mlReports = reports.map(r => r.toJSON());
        }
      }
    } catch (pgError) {
      console.warn('Failed to fetch ML reports from PostgreSQL:', pgError.message);
      // Continue without ML reports if PostgreSQL is not available
    }

    // Combine results with enhanced status information
    const combinedResults = [
      ...radiologyResults.map(result => ({
        ...(result.toJSON ? result.toJSON() : result),
        type: 'traditional',
        source: 'postgresql',
        report_status: 'completed',
        status_display: 'Completed',
        status_description: 'Report is ready for viewing'
      })),
      ...mlReports.map(report => {
        let statusDisplay = '';
        let statusDescription = '';

        switch (report.report_status) {
          case 'pending':
            statusDisplay = 'Pending';
            statusDescription = 'Report is being processed by ML system';
            break;
          case 'processing':
            statusDisplay = 'Processing';
            statusDescription = 'ML model is analyzing the images';
            break;
          case 'completed':
            statusDisplay = 'Completed';
            statusDescription = 'Report is ready for viewing';
            break;
          case 'failed':
            statusDisplay = 'Failed';
            statusDescription = 'Report processing failed, please contact support';
            break;
          default:
            statusDisplay = 'Unknown';
            statusDescription = 'Report status is unknown';
        }

        return {
          ...report,
          type: 'ml_generated',
          source: 'postgresql',
          id: report.id.toString(),
          status_display: statusDisplay,
          status_description: statusDescription
        };
      })
    ].sort((a, b) => new Date(b.created_at || b.createdAt) - new Date(a.created_at || a.createdAt));

    if (combinedResults.length === 0) {
      return res.json({
        message: userRole === 'doctor' ? "No results found for your assigned patients" : "No results found in radiology-results",
        results: [],
        total: 0
      });
    }

    res.json({
      message: "Radiology results retrieved successfully",
      results: combinedResults,
      total: combinedResults.length,
      filteredByDoctor: userRole === 'doctor'
    });
  } catch (error) {
    console.error("Error fetching radiology results:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});

apiRouter.post("/radiology-results", async (req, res) => {
  try {
    const radiologyResult = await RadiologyResult.create(req.body);

    // Log Activity: Report Generated (Manual)
    try {
      await ActivityLog.create({
        action: 'report_generated',
        description: `Radiology report generated for patient ${req.body.patientId || 'Unknown'}: ${req.body.title || 'Untitled'}`,
        entityType: 'report',
        entityId: radiologyResult.id ? (radiologyResult.id.toString ? radiologyResult.id.toString() : radiologyResult.id) : 'N/A',
        metadata: {
          reportType: 'Manual',
          title: req.body.title,
          patientId: req.body.patientId
        }
      });
    } catch (logError) {
      console.error('Error logging report generation:', logError);
    }

    res.status(201).json({
      message: "Radiology result added successfully",
      result: radiologyResult
    });
  } catch (error) {
    console.error("Error saving radiology result:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});

// GET /api/radiology-results/patient/:patientId/recent - Get patient's most recent reports
apiRouter.get("/radiology-results/patient/:patientId/recent", async (req, res) => {
  try {
    const { patientId } = req.params;
    const { limit = 5 } = req.query;

    // Get radiology results from PostgreSQL
    const radiologyResults = await RadiologyResult.findAll({
      where: { patientId: patientId },
      order: [['createdAt', 'DESC']],
      limit: parseInt(limit)
    });

    // Get ML reports from PostgreSQL
    let mlReports = [];
    try {
      const reports = await MLReportModel.findAll({
        where: { patient_id: patientId },
        order: [['created_at', 'DESC']]
      });
      mlReports = reports.map(r => r.toJSON());
    } catch (pgError) {
      console.warn('Failed to fetch ML reports from PostgreSQL:', pgError.message);
    }

    // Combine and sort by creation date
    const combinedResults = [
      ...radiologyResults.map(result => ({
        ...(result.toJSON ? result.toJSON() : result),
        type: 'traditional',
        source: 'postgresql',
        report_status: 'completed',
        status_display: 'Completed',
        status_description: 'Report is ready for viewing'
      })),
      ...mlReports.map(report => {
        let statusDisplay = '';
        let statusDescription = '';

        switch (report.report_status) {
          case 'pending':
            statusDisplay = 'Pending';
            statusDescription = 'Report is being processed by ML system';
            break;
          case 'processing':
            statusDisplay = 'Processing';
            statusDescription = 'ML model is analyzing the images';
            break;
          case 'completed':
            statusDisplay = 'Completed';
            statusDescription = 'Report is ready for viewing';
            break;
          case 'failed':
            statusDisplay = 'Failed';
            statusDescription = 'Report processing failed, please contact support';
            break;
          default:
            statusDisplay = 'Unknown';
            statusDescription = 'Report status is unknown';
        }

        return {
          ...report,
          type: 'ml_generated',
          source: 'postgresql',
          id: report.id.toString(),
          status_display: statusDisplay,
          status_description: statusDescription
        };
      })
    ].sort((a, b) => new Date(b.created_at || b.createdAt) - new Date(a.created_at || a.createdAt))
      .slice(0, parseInt(limit));

    if (combinedResults.length === 0) {
      return res.json({
        message: "No results found for this patient",
        results: [],
        total: 0
      });
    }

    res.json({
      message: "Patient's recent reports retrieved successfully",
      results: combinedResults,
      total: combinedResults.length
    });
  } catch (error) {
    console.error("Error fetching patient's recent reports:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});

// GET /api/radiology-results/status/:patientId - Get patient report status summary
apiRouter.get("/radiology-results/status/:patientId", async (req, res) => {
  try {
    const { patientId } = req.params;

    // Get radiology results from PostgreSQL
    const radiologyResults = await RadiologyResult.findAll({
      where: { patientId }
    });

    // Get ML reports from PostgreSQL
    const reports = await MLReportModel.findAll({
      where: { patient_id: patientId },
      order: [['created_at', 'DESC']]
    });
    const mlReports = reports.map(r => r.toJSON());

    // Count reports by status
    const statusSummary = {
      total_reports: radiologyResults.length + mlReports.length,
      traditional_reports: radiologyResults.length,
      ml_reports: mlReports.length,
      pending: 0,
      processing: 0,
      completed: 0,
      failed: 0,
      reports: []
    };

    // Add traditional reports (always completed)
    radiologyResults.forEach(result => {
      statusSummary.completed++;
      const resultData = result.toJSON ? result.toJSON() : result;
      statusSummary.reports.push({
        id: resultData.id,
        type: 'traditional',
        report_type: 'Radiology Report',
        status_display: 'Completed',
        status_description: 'Report is ready for viewing',
        created_at: resultData.createdAt
      });
    });

    // Add ML reports with status
    mlReports.forEach(report => {
      statusSummary[report.report_status]++;

      let statusDisplay = '';
      let statusDescription = '';

      switch (report.report_status) {
        case 'pending':
          statusDisplay = 'Pending';
          statusDescription = 'Report is being processed by ML system';
          break;
        case 'processing':
          statusDisplay = 'Processing';
          statusDescription = 'ML model is analyzing the images';
          break;
        case 'completed':
          statusDisplay = 'Completed';
          statusDescription = 'Report is ready for viewing';
          break;
        case 'failed':
          statusDisplay = 'Failed';
          statusDescription = 'Report processing failed, please contact support';
          break;
      }

      statusSummary.reports.push({
        id: report.id,
        type: 'ml_generated',
        report_type: report.report_type,
        status_display: statusDisplay,
        status_description: statusDescription,
        created_at: report.created_at
      });
    });

    // Sort by creation date (newest first)
    statusSummary.reports.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));

    res.json({
      message: "Patient report status retrieved successfully",
      patient_id: patientId,
      ...statusSummary
    });
  } catch (error) {
    console.error("Error fetching patient report status:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});

// Patients Routes
apiRouter.get("/patients", async (req, res) => {
  try {
    const patients = await Patient.findAll();
    res.json(patients);
  } catch (error) {
    console.error("Error fetching patients:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});



apiRouter.post("/patients", upload.single('file'), async (req, res) => {
  try {
    const patientData = req.body;
    console.log("Received patient data:", patientData);

    if (req.file) {
      patientData.medical_history = req.file.path;
    }

    // Auto-generate credentials if not provided (admin creating patient)
    let generatedCredentials = null;
    if (!patientData.email || !patientData.password) {
      // Generate credentials
      const tempId = Date.now().toString(); // Use timestamp as temp ID
      generatedCredentials = generateCredentials(patientData.name, tempId);

      if (!patientData.email) {
        patientData.email = generatedCredentials.email;
      }
      if (!patientData.password) {
        patientData.password = generatedCredentials.password;
      }

      console.log(`✅ Auto-generated credentials for patient ${patientData.name}`);
      console.log(`   Email: ${generatedCredentials.email}`);
    }

    // Check if user already exists in User collection
    const existingUser = await User.findOne({ email: patientData.email });
    if (existingUser) {
      return res.status(400).json({ message: "Patient already exists with this email" });
    }

    // 1. Create Patient in PostgreSQL
    const newPatient = await Patient.create({
      name: patientData.name,
      email: patientData.email,
      age: patientData.age,
      gender: patientData.gender,
      contact: patientData.guardian_phone || patientData.contact, // Use guardian phone if contact not provided
      address: patientData.address,
      medical_history: patientData.medical_history,
      symptoms: patientData.symptoms ? (Array.isArray(patientData.symptoms) ? patientData.symptoms : [patientData.symptoms]) : [],
      blood_type: patientData.blood_type,
      height: patientData.height,
      weight: patientData.weight,
      allergies: patientData.allergies,
      vitals: patientData.vitals || {},
      status: patientData.status || 'Admitted',
      date: patientData.date,
      assignedDoctor: null, // Will be assigned by admin later
      profileCompleted: true // Created by admin
    });

    console.log(`✅ Patient profile created in PostgreSQL with ID: ${newPatient.id}`);

    // 2. Create User in MongoDB for Auth
    const newUser = new User({
      email: patientData.email,
      password: patientData.password,
      plainPassword: patientData.password, // Store explicitly for admin display (Demo only)
      role: 'patient',
      referenceId: newPatient.id
    });

    await newUser.save();
    console.log(`✅ User credentials created in MongoDB with ID: ${newUser._id}`);

    // 3. Smart doctor assignment based on priority
    try {
      const { Op } = require('sequelize');
      const priority = newPatient.priority || 2;
      let doctorQuery = {};

      if (priority === 1) {
        // High priority → Available doctors only
        doctorQuery = { where: { availability: 'Available' } };
        console.log('Priority 1 (High): Looking for Available doctors');
      } else if (priority === 2) {
        // Medium priority → On Call OR Available
        doctorQuery = {
          where: {
            availability: { [Op.in]: ['On Call', 'Available'] }
          }
        };
        console.log('Priority 2 (Medium): Looking for Available or On Call doctors');
      } else if (priority === 3) {
        // Low priority → Anyone except Busy
        doctorQuery = {
          where: {
            availability: { [Op.not]: 'Busy' }
          }
        };
        console.log('Priority 3 (Low): Looking for any doctor not Busy');
      }

      const availableDoctor = await Doctor.findOne(doctorQuery);

      if (availableDoctor) {
        await newPatient.update({
          assignedDoctor: {
            id: availableDoctor.id,
            name: availableDoctor.name
          }
        });
        console.log(`Assigned doctor ${availableDoctor.name} (${availableDoctor.availability}) to patient ${newPatient.name} (Priority ${priority})`);

        // Auto-update doctor's availability status based on patient count
        try {
          const statusUpdate = await availableDoctor.updateAvailabilityStatus();
          console.log(`Doctor ${availableDoctor.name} status updated: ${statusUpdate.status} (${statusUpdate.patientCount} patients)`);
        } catch (statusError) {
          console.error('Error updating doctor status:', statusError);
        }
      } else {
        console.log(`No suitable doctors found for priority ${priority} patient`);
      }
    } catch (docError) {
      console.error('Error assigning doctor:', docError);
      // Continue without assignment
    }

    // Send to ML prioritization queue if symptoms are provided
    if (patientData.symptoms) {
      try {
        const mlPayload = {
          patient_id: newPatient.id, // UUID from Postgres
          patient_name: newPatient.name,
          age: newPatient.age,
          gender: newPatient.gender,
          symptoms: Array.isArray(patientData.symptoms) ? patientData.symptoms : [patientData.symptoms],
          vitals: patientData.vitals || {},
          medical_history: patientData.medical_history || "",
          timestamp: Date.now()
        };

        // Send directly to priority_queue
        const channel = rabbitmq.channel;
        channel.sendToQueue('priority_queue', Buffer.from(JSON.stringify(mlPayload)), {
          persistent: true
        });
        logger.info(`Patient ${newPatient.id} sent to ML prioritization queue`);
      } catch (queueError) {
        logger.error('Failed to send patient to ML queue:', queueError);
        // Continue even if queue fails - patient is already saved
      }
    }

    // Log activity: Patient added
    try {
      await ActivityLog.create({
        action: 'patient_added',
        description: `${newPatient.name} | Patient added to the system`,
        entityType: 'patient',
        entityId: newPatient.id,
        metadata: {
          patientName: newPatient.name,
          doctorName: newPatient.assignedDoctor?.name || 'Unassigned',
          status: 'Patient added to the system',
          age: newPatient.age,
          gender: newPatient.gender,
        }
      });
    } catch (logError) {
      console.error('Error creating activity log:', logError);
      // Continue even if logging fails
    }

    const response = {
      message: "Patient added successfully",
      patient: newPatient,
    };

    // Include generated credentials in response if they were auto-generated
    if (generatedCredentials) {
      response.credentials = {
        email: generatedCredentials.email,
        password: generatedCredentials.password,
        message: generatedCredentials.message
      };
      response.message += " - Login credentials generated";
    }

    res.json(response);
  } catch (error) {
    console.error("Error saving patient:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});

// GET a single patient by ID
apiRouter.get("/patients/:id", async (req, res) => {
  try {
    const patient = await Patient.findByPk(req.params.id);
    if (!patient) {
      return res.status(404).json({ message: "Patient not found" });
    }
    res.json(patient);
  } catch (error) {
    console.error("Error fetching patient:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});

// Update patient profile (for profile completion)
apiRouter.put("/patients/profile/:id", async (req, res) => {
  try {
    const { id } = req.params;
    const updateData = req.body;

    // Set profileCompleted to true when profile is updated
    updateData.profileCompleted = true;

    const [updatedCount] = await Patient.update(updateData, {
      where: { id },
      returning: true
    });

    if (updatedCount === 0) {
      return res.status(404).json({ message: "Patient not found" });
    }

    const updatedPatient = await Patient.findByPk(id);
    res.json(updatedPatient);
  } catch (error) {
    console.error("Error updating patient profile:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});

apiRouter.put("/patients/:id", async (req, res) => {
  try {
    const { id } = req.params;
    const updateData = req.body;

    const [updatedCount] = await Patient.update(updateData, {
      where: { id },
      returning: true
    });

    if (updatedCount === 0) {
      return res.status(404).json({ message: "Patient not found" });
    }

    const updatedPatient = await Patient.findByPk(id);
    res.json(updatedPatient);
  } catch (error) {
    console.error("Error updating patient:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});

// Doctor check patient endpoint
apiRouter.put("/patients/:id/check", async (req, res) => {
  try {
    const { id } = req.params;
    const { checked } = req.body; // true or false

    const patient = await Patient.findByPk(id);
    if (!patient) {
      return res.status(404).json({ message: "Patient not found" });
    }

    // Update patient check status and status
    await patient.update({
      checkedByDoctor: checked,
      status: checked ? 'Completed' : 'Doctor Appointment'
    });

    console.log(`Patient ${patient.name} marked as ${checked ? 'checked' : 'unchecked'} by doctor`);

    // Log activity: Patient checked by doctor
    if (checked) {
      try {
        const doctorName = patient.assignedDoctor?.name || 'Unassigned';
        await ActivityLog.create({
          action: 'patient_checked',
          description: `${patient.name} | Patient checked by doctor success`,
          entityType: 'patient',
          entityId: patient.id,
          metadata: {
            patientName: patient.name,
            doctorName: doctorName,
            status: 'Patient checked by doctor success',
            checkedByDoctor: checked
          }
        });
      } catch (logError) {
        console.error('Error logging patient check:', logError);
      }
    }

    // Update doctor availability after check
    if (patient.assignedDoctor && patient.assignedDoctor.id) {
      try {
        const doctor = await Doctor.findByPk(patient.assignedDoctor.id);
        if (doctor) {
          const previousStatus = doctor.availability;
          const statusUpdate = await doctor.updateAvailabilityStatus();
          console.log(`Doctor ${doctor.name} availability updated: ${statusUpdate.status} (${statusUpdate.patientCount} patients)`);

          // Log activity: Doctor is available now (when status changes to Available)
          if (statusUpdate.status === 'Available' && previousStatus !== 'Available') {
            try {
              await ActivityLog.create({
                action: 'doctor_available',
                description: `${doctor.name} | Doctor is available now`,
                entityType: 'doctor',
                entityId: doctor.id,
                metadata: {
                  doctorName: doctor.name,
                  status: 'Doctor is available now',
                  previousStatus: previousStatus,
                  patientCount: statusUpdate.patientCount
                }
              });
            } catch (logError) {
              console.error('Error logging doctor availability:', logError);
            }
          }
        }
      } catch (docError) {
        console.error('Error updating doctor status:', docError);
      }
    }

    res.json({
      message: `Patient ${checked ? 'checked' : 'unchecked'} successfully`,
      patient
    });
  } catch (error) {
    console.error("Error checking patient:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});

// Admin Routes
apiRouter.get("/admins", requireAdmin, async (req, res) => {
  try {
    const admins = await Admin.findAll();
    res.json(admins);
  } catch (error) {
    console.error("Error fetching admins:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});

apiRouter.get("/admins/:id", requireAdmin, async (req, res) => {
  try {
    const admin = await Admin.findByPk(req.params.id);
    if (!admin) {
      return res.status(404).json({ message: "Admin not found" });
    }
    res.json(admin);
  } catch (error) {
    console.error("Error fetching admin:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});

// Doctors Routes
apiRouter.get("/doctors", async (req, res) => {
  try {
    const doctors = await Doctor.findAll();

    // Auto-refresh each doctor's availability status
    for (const doctor of doctors) {
      try {
        await doctor.updateAvailabilityStatus();
      } catch (statusError) {
        console.error(`Error updating status for doctor ${doctor.id}:`, statusError);
      }
    }

    // Re-fetch to get updated statuses
    const updatedDoctors = await Doctor.findAll();
    res.json(updatedDoctors);
  } catch (error) {
    console.error("Error fetching doctors:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});

apiRouter.get("/doctors/:id", async (req, res) => {
  try {
    const doctor = await Doctor.findByPk(req.params.id);
    if (!doctor) {
      return res.status(404).json({ message: "Doctor not found" });
    }
    res.json(doctor);
  } catch (error) {
    console.error("Error fetching doctor:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});

apiRouter.put("/doctors/:id", async (req, res) => {
  try {
    const { id } = req.params;
    const { availability } = req.body;

    const [updatedCount] = await Doctor.update({ availability }, {
      where: { id },
      returning: true
    });

    if (updatedCount === 0) {
      return res.status(404).json({ message: "Doctor not found" });
    }

    const updatedDoctor = await Doctor.findByPk(id);
    res.json(updatedDoctor);
  } catch (error) {
    console.error("Error updating doctor:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});

apiRouter.post("/doctors", async (req, res) => {
  try {
    const doctorData = req.body;
    const newDoctor = await Doctor.create(doctorData);
    res.json({
      message: "Doctor added successfully",
      doctor: newDoctor,
    });
  } catch (error) {
    console.error("Error saving doctor:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});

// Prescriptions Routes
apiRouter.get("/prescriptions", async (req, res) => {
  try {
    const { patientName } = req.query;
    const whereClause = patientName ? { patientName } : {};
    const prescriptions = await Prescription.findAll({ where: whereClause });
    res.json(prescriptions);
  } catch (error) {
    console.error("Error fetching prescriptions:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});

apiRouter.get("/prescriptions/pending", async (req, res) => {
  try {
    const pendingPrescriptions = await Prescription.findAll({ where: { status: 'pending' } });
    res.json(pendingPrescriptions);
  } catch (error) {
    console.error("Error fetching pending prescriptions:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});

apiRouter.post("/prescriptions", async (req, res) => {
  try {
    const prescriptionData = req.body;
    const newPrescription = await Prescription.create(prescriptionData);

    // Log Activity: Prescription Written
    try {
      await ActivityLog.create({
        action: 'prescription_written',
        description: `${prescriptionData.patientName} | Prescription written`,
        entityType: 'prescription',
        entityId: newPrescription.id ? (newPrescription.id.toString ? newPrescription.id.toString() : newPrescription.id) : 'N/A',
        metadata: {
          patientName: prescriptionData.patientName,
          doctorName: prescriptionData.prescribedBy || 'N/A',
          status: 'Prescription written',
          medication: prescriptionData.medication
        }
      });
    } catch (logError) {
      console.error('Error logging prescription:', logError);
    }

    res.json({
      message: "Prescription added successfully",
      prescription: newPrescription,
    });
  } catch (error) {
    console.error("Error saving prescription:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});

apiRouter.put("/prescriptions/:id", async (req, res) => {
  try {
    const { id } = req.params;
    const prescription = await Prescription.findByPk(id);
    if (!prescription) {
      return res.status(404).json({ message: "Prescription not found" });
    }

    await prescription.update(req.body);
    res.json({
      message: "Prescription updated successfully",
      prescription,
    });
  } catch (error) {
    console.error("Error updating prescription:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});

apiRouter.delete("/prescriptions/:id", async (req, res) => {
  try {
    const { id } = req.params;
    const prescription = await Prescription.findByPk(id);
    if (!prescription) {
      return res.status(404).json({ message: "Prescription not found" });
    }

    await prescription.destroy();
    res.json({ message: "Prescription deleted successfully" });
  } catch (error) {
    console.error("Error deleting prescription:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});

apiRouter.put("/prescriptions/:id/reject", async (req, res) => {
  try {
    const { id } = req.params;
    const prescription = await Prescription.findById(id);
    if (!prescription) {
      return res.status(404).json({ message: "Prescription not found" });
    }
    prescription.status = 'rejected';
    await prescription.save();
    res.json(prescription);
  } catch (error) {
    console.error("Error rejecting prescription:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});

// Notes Routes
apiRouter.get("/notes", async (req, res) => {
  try {
    const notes = await Note.findAll({
      order: [['createdAt', 'DESC']]
    });
    res.json(notes);
  } catch (error) {
    console.error("Error fetching notes:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});

apiRouter.post("/notes", async (req, res) => {
  try {
    const noteData = req.body;
    const newNote = await Note.create(noteData);
    res.json({
      message: "Note added successfully",
      note: newNote,
    });
  } catch (error) {
    console.error("Error saving note:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});

// Login endpoint (direct route for frontend compatibility)
apiRouter.post("/login", async (req, res) => {
  try {
    const { email, password, role } = req.body;

    // Validate input
    if (!email || !password) {
      return res.status(400).json({ message: "Please provide email and password" });
    }

    const userRole = role || "patient";
    let Model;

    switch (userRole) {
      case "doctor":
        Model = Doctor;
        break;
      case "admin":
        Model = Admin;
        break;
      default:
        Model = Patient;
    }

    // Find user in the specific collection
    console.log(`[LOGIN DEBUG] Attempting login for email: ${email}, role: ${userRole}`);
    const user = await Model.findOne({ email });
    if (!user) {
      console.log(`[LOGIN DEBUG] User not found in ${userRole} collection`);
      return res.status(401).json({ message: "Invalid credentials" });
    }
    console.log(`[LOGIN DEBUG] User found: ${user._id}`);

    // Check password
    const isMatch = await user.comparePassword(password);
    console.log(`[LOGIN DEBUG] Password match result: ${isMatch}`);

    if (!isMatch) {
      console.log(`[LOGIN DEBUG] Password mismatch`);
      return res.status(401).json({ message: "Invalid credentials" });
    }

    // Generate JWT token
    const jwt = require('jsonwebtoken');
    const token = jwt.sign(
      {
        userId: user._id,
        email: user.email,
        role: userRole,
        profileCompleted: user.profileCompleted || false
      },
      process.env.JWT_SECRET || "your-secret-key-change-in-production",
      { expiresIn: "7d" }
    );

    res.json({
      message: "Login successful",
      token,
      user: {
        id: user._id,
        email: user.email,
        name: user.name,
        role: userRole,
        profileCompleted: user.profileCompleted || false
      }
    });
  } catch (error) {
    console.error("Login error:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});

// Admin Routes
apiRouter.get("/admins", async (req, res) => {
  try {
    const admins = await Admin.find();
    res.json(admins);
  } catch (error) {
    console.error("Error fetching admins:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});

apiRouter.post("/admins", async (req, res) => {
  try {
    const adminData = req.body;
    const newAdmin = await Admin.create(adminData);
    res.json({
      message: "Admin added successfully",
      admin: newAdmin,
    });
  } catch (error) {
    console.error("Error saving admin:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});

// Additional Admin CRUD operations
apiRouter.get("/admins/:id", async (req, res) => {
  try {
    const admin = await Admin.findByPk(req.params.id);
    if (!admin) {
      return res.status(404).json({ message: "Admin not found" });
    }
    res.json(admin);
  } catch (error) {
    console.error("Error fetching admin:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});

apiRouter.put("/admins/:id", async (req, res) => {
  try {
    const { id } = req.params;
    const updateData = req.body;

    const [updatedCount] = await Admin.update(updateData, {
      where: { id },
      returning: true
    });

    if (updatedCount === 0) {
      return res.status(404).json({ message: "Admin not found" });
    }

    const updatedAdmin = await Admin.findByPk(id);
    res.json(updatedAdmin);
  } catch (error) {
    console.error("Error updating admin:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});

apiRouter.delete("/admins/:id", async (req, res) => {
  try {
    const { id } = req.params;
    const admin = await Admin.findByPk(id);
    if (!admin) {
      return res.status(404).json({ message: "Admin not found" });
    }

    await admin.destroy();
    res.json({ message: "Admin deleted successfully" });
  } catch (error) {
    console.error("Error deleting admin:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});

// Logs Routes - Get activity logs, login activities, and system logs
apiRouter.post("/logs", async (req, res) => {
  try {
    const { action, description, entityType, entityId, metadata } = req.body;

    // Validate required fields
    if (!action || !description) {
      return res.status(400).json({ message: "Action and description are required" });
    }

    const newLog = await ActivityLog.create({
      action,
      description,
      entityType: entityType || 'system',
      entityId: entityId || 'N/A',
      metadata: metadata || {}
    });

    res.status(201).json({
      message: "Log entry created successfully",
      log: newLog
    });
  } catch (error) {
    console.error("Error creating log entry:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});

apiRouter.get("/logs", async (req, res) => {
  try {
    const { limit = 50, type = 'all' } = req.query;
    const { Sequelize } = require('sequelize');

    let logs = [];

    // Get activity logs (patient additions, doctor assignments, etc.)
    if (type === 'all' || type === 'activity') {
      const activityLogs = await ActivityLog.findAll({
        order: [[Sequelize.literal('"created_at"'), 'DESC']],
        limit: parseInt(limit)
      });
      logs.push(...activityLogs.map(log => ({ ...log.toJSON(), logType: 'activity' })));
    }

    // Get login activities  
    if (type === 'all' || type === 'login') {
      const loginActivities = await LoginActivity.findAll({
        order: [[Sequelize.literal('"created_at"'), 'DESC']],
        limit: parseInt(limit)
      });
      logs.push(...loginActivities.map(log => ({ ...log.toJSON(), logType: 'login' })));
    }

    // Sort combined logs by timestamp
    logs.sort((a, b) => {
      const timeA = new Date(a.created_at || a.createdAt);
      const timeB = new Date(b.created_at || b.createdAt);
      return timeB - timeA;
    });

    res.json({
      success: true,
      message: "Logs retrieved successfully",
      logs: logs.slice(0, parseInt(limit)),
      total: logs.length
    });
  } catch (error) {
    console.error("Error fetching logs:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});




// Mount authentication routes
app.use("/api/auth", authRoutes);

// Mount API routes
app.use("/api", apiRouter);

// Mount Catalog routes
app.use("/api/catalog", catalogRoutes); // Kept for backward compatibility if needed, but not primary data source
app.use("/api/ml-reports", mlReportsRoutes);
app.use("/api/dicom", dicomRoutes);
app.use("/api/viewer", viewerRoutes);
app.use("/api/users", userRoutes);

// Admin debugging routes for MySQL
app.get("/admin/reports", async (req, res) => {
  try {
    const limit = parseInt(req.query.limit) || 10;
    const reports = await MLReportModel.findAll({
      order: [['created_at', 'DESC']],
      limit: limit
    });
    res.json({
      message: "Recent ML reports from PostgreSQL",
      reports: reports.map(r => r.toJSON()),
      total: reports.length
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get("/admin/stats", async (req, res) => {
  try {
    const { Op } = require('sequelize');

    // Get stats from PostgreSQL
    const totalReports = await MLReportModel.count();
    const pendingReports = await MLReportModel.count({ where: { report_status: 'pending' } });
    const processingReports = await MLReportModel.count({ where: { report_status: 'processing' } });
    const completedReports = await MLReportModel.count({ where: { report_status: 'completed' } });
    const failedReports = await MLReportModel.count({ where: { report_status: 'failed' } });

    const stats = {
      total: totalReports,
      pending: pendingReports,
      processing: processingReports,
      completed: completedReports,
      failed: failedReports
    };

    res.json({
      message: "ML report statistics from PostgreSQL",
      stats
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// X-ray Analysis Endpoint (DuoFormer Integration)
apiRouter.post("/xray-analysis", upload.single('image'), async (req, res) => {
  try {
    const { patient_id, patient_name } = req.body;

    // Validate inputs
    if (!patient_id || !patient_name) {
      return res.status(400).json({
        message: "Missing required fields: patient_id and patient_name"
      });
    }

    if (!req.file) {
      return res.status(400).json({
        message: "No image file uploaded"
      });
    }

    // Read image file and convert to base64
    const imageBuffer = fs.readFileSync(req.file.path);
    const imageBase64 = imageBuffer.toString('base64');

    // Generate scan ID
    const scan_id = `XRAY_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    // Send to DuoFormer queue
    const payload = {
      patient_id,
      patient_name,
      scan_id,
      image_base64: imageBase64,
      timestamp: new Date().toISOString()
    };

    const channel = rabbitmq.channel;
    if (!channel) {
      return res.status(503).json({
        message: "ML service temporarily unavailable"
      });
    }

    await channel.assertQueue('xray_request_queue', { durable: true });
    channel.sendToQueue('xray_request_queue', Buffer.from(JSON.stringify(payload)), {
      persistent: true
    });

    logger.info(`X-ray analysis requested for patient ${patient_id}, scan ${scan_id}`);

    res.json({
      message: "X-ray analysis request submitted successfully",
      scan_id,
      patient_id,
      status: "processing",
      note: "Results will be available in ML Reports once analysis is complete"
    });

  } catch (error) {
    logger.error("Error processing X-ray analysis request:", error);
    res.status(500).json({
      message: "Server error",
      error: error.message
    });
  }
});

// Root endpoint
app.get('/', (req, res) => {
  res.json({
    message: 'Integrated Hospital Management API',
    version: '1.0.0',
    status: 'running',
    endpoints: {
      auth: '/api/auth',
      patients: '/api/patients',
      doctors: '/api/doctors',
      prescriptions: '/api/prescriptions',
      notes: '/api/notes',
      admins: '/api/admins',
      radiology: '/api/radiology-results',
      ml_reports: '/api/ml-reports',
      catalog: '/catalog'
    }
  });
});

// Health check endpoint
app.get('/health', async (req, res) => {
  try {
    // Check MongoDB connection
    const mongoStatus = mongoose.connection.readyState === 1 ? 'connected' : 'disconnected';

    // Check Redis connection
    const redisStatus = redisClient.isOpen ? 'connected' : 'disconnected';

    res.json({
      status: 'healthy',
      database: 'radiology_hospital',
      mongodb: mongoStatus,
      redis: redisStatus,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(503).json({
      status: 'unhealthy',
      error: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    success: false,
    message: 'Route not found'
  });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Error:', err);
  res.status(err.status || 500).json({
    success: false,
    message: err.message || 'Internal server error',
    error: process.env.NODE_ENV === 'development' ? err : {}
  });
});

app.listen(PORT, () => {
  logger.info(`
╔════════════════════════════════════════════════════════╗
║   🏥 Integrated Hospital Management API Server       ║
║                                                        ║
║   🚀 Server running on: http://localhost:${PORT}       ║
║   📊 Database: radiology_hospital                     ║
║   📝 API Docs: http://localhost:${PORT}/               ║
║                                                        ║
║   Available Endpoints:                                 ║
║   - Auth: /api/auth/*                                  ║
║   - Patients: /api/patients/*                         ║
║   - Doctors: /api/doctors/*                            ║
║   - Prescriptions: /api/prescriptions/*                 ║
║   - Notes: /api/notes/*                                 ║
║   - Admins: /api/admins/*                              ║
║   - Radiology: /api/radiology-results/*                ║
║   - ML Reports: /api/ml-reports/*                      ║
║   - Catalog: /catalog/*                                  ║
║   - Metrics: /catalog/metrics                          ║
║   - Health: /health                                    ║
╚════════════════════════════════════════════════════════╝
  `);

  // Update database connection metrics
  metrics.setDatabaseConnection('mongodb', mongoose.connection.readyState === 1);
  metrics.setDatabaseConnection('redis', redisClient.isOpen);
});

// Graceful shutdown
process.on('SIGINT', async () => {
  logger.info('\nSIGINT received. Shutting down gracefully...');

  try {
    // Close RabbitMQ connection
    await rabbitmq.close();
    logger.info('✓ RabbitMQ connection closed');

    // Close MongoDB connection
    await mongoose.connection.close();
    logger.info('✓ MongoDB connection closed');

    // Close Redis connection
    await redisClient.quit();
    logger.info('✓ Redis connection closed');

    logger.info('✓ Server shutdown complete');
    process.exit(0);
  } catch (error) {
    logger.error('Error during shutdown:', error);
    process.exit(1);
  }
});

process.on('SIGTERM', async () => {
  logger.info('\nSIGTERM received. Shutting down gracefully...');

  try {
    await rabbitmq.close();
    await mongoose.connection.close();
    await redisClient.quit();
    logger.info('✓ Server shutdown complete');
    process.exit(0);
  } catch (error) {
    logger.error('Error during shutdown:', error);
    process.exit(1);
  }
});

module.exports = app;

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
const { MLReport: MLReportModel, DICOMImage, Patient: PatientPG, Doctor: DoctorPG } = require("./models"); // Import Postgres models
const mongoose = require("mongoose");
const User = require("./mongoSchemas/User"); // Auth User model

// Import RabbitMQ, Logger, and Metrics
const rabbitmq = require("./rabbitmq");
const logger = require("./logger");
const metrics = require("./metrics");
// MongoDB Models (core entities)
const Patient = require("./mongoSchemas/Patient");
const Doctor = require("./mongoSchemas/Doctor");
const Prescription = require("./mongoSchemas/Prescription");
const Note = require("./mongoSchemas/Note");
const Admin = require("./mongoSchemas/Admin");
const Job = require("./mongoSchemas/Job");
const RadiologyResult = require("./mongoSchemas/RadiologyResult");
const ActivityLog = require("./mongoSchemas/ActivityLog");
const LoginActivity = require("./mongoSchemas/LoginActivity");

// Import utilities
const { generateCredentials } = require("./utils/credentialGenerator");
const { extractUserInfo, getAllowedPatientIds, checkPatientAccess, requireAuth, requireAdmin } = require("./middleware/accessControl");

// Import routes
const authRoutes = require("./routes/authRoutes");
const catalogRoutes = require("./routes/catalogRoutes");
const mlReportsRoutes = require("./routes/mlReportsRoutes");
const dicomRoutes = require("./routes/dicomRoutes-postgres");
const viewerRoutes = require("./routes/viewerRoutes");

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
    if (userRole === 'doctor' && userId) {
      // Get all patients assigned to this doctor
      const assignedPatients = await Patient.find({ 'assignedDoctor.id': userId });
      allowedPatientIds = assignedPatients.map(p => p._id.toString());

      console.log(`✓ Doctor ${userId} has ${allowedPatientIds.length} assigned patients`);
      console.log('🔍 DEBUG - Allowed Patient IDs:', allowedPatientIds);
    } else {
      console.log('🔍 DEBUG - Not a doctor or no userId, allowedPatientIds:', allowedPatientIds);
    }

    // Get traditional radiology results from MongoDB
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
      radiologyResults = await RadiologyResult.find({ patientId: patientId });
    } else {
      // Filter by allowed patients if doctor
      if (allowedPatientIds !== null) {
        // Doctor is logged in - only show their assigned patients
        if (allowedPatientIds.length === 0) {
          // Doctor has no assigned patients - show nothing
          radiologyResults = [];
        } else {
          // Doctor has assigned patients - filter by them
          radiologyResults = await RadiologyResult.find({
            patientId: { $in: allowedPatientIds }
          });
        }
      } else {
        // Not a doctor (admin or no auth) - show all
        radiologyResults = await RadiologyResult.find();
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
        // Filter ML reports by allowed patients if doctor
        if (allowedPatientIds !== null) {
          // Doctor is logged in - only show their assigned patients
          if (allowedPatientIds.length === 0) {
            // Doctor has no assigned patients - show nothing
            mlReports = [];
          } else {
            // Doctor has assigned patients - get all and filter
            const reports = await MLReportModel.findAll({
              where: { patient_id: allowedPatientIds },
              order: [['created_at', 'DESC']]
            });
            mlReports = reports.map(r => r.toJSON());
          }
        } else {
          // Not a doctor (admin or no auth) - show all
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
        ...result.toObject(),
        type: 'traditional',
        source: 'mongodb',
        report_status: 'completed', // Traditional reports are always completed
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
          source: 'mysql',
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
    const radiologyResult = new RadiologyResult(req.body);
    await radiologyResult.save();
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
    const { limit = 5 } = req.query; // Default to 5 most recent reports

    // Get traditional radiology results from MongoDB
    const radiologyResults = await RadiologyResult.find({ patientId: patientId })
      .sort({ createdAt: -1 })
      .limit(parseInt(limit));

    // Get ML reports from PostgreSQL (with error handling)
    let mlReports = [];
    try {
      const reports = await MLReportModel.findAll({
        where: { patient_id: patientId },
        order: [['created_at', 'DESC']]
      });
      mlReports = reports.map(r => r.toJSON());
    } catch (pgError) {
      console.warn('Failed to fetch ML reports from PostgreSQL:', pgError.message);
      // Continue without ML reports if PostgreSQL is not available
    }

    // Combine and sort by creation date (newest first)
    const combinedResults = [
      ...radiologyResults.map(result => ({
        ...result.toObject(),
        type: 'traditional',
        source: 'mongodb',
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
          source: 'mysql',
          id: report.id.toString(),
          status_display: statusDisplay,
          status_description: statusDescription
        };
      })
    ].sort((a, b) => new Date(b.created_at || b.createdAt) - new Date(a.created_at || a.createdAt))
      .slice(0, parseInt(limit)); // Take only the most recent ones

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

    // Get traditional radiology results from MongoDB
    const radiologyResults = await RadiologyResult.find({ patientId });

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
      statusSummary.reports.push({
        id: result._id,
        type: 'traditional',
        report_type: 'Radiology Report',
        status_display: 'Completed',
        status_description: 'Report is ready for viewing',
        created_at: result.createdAt
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
    const patients = await Patient.find();
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
    const newPatient = await PatientPG.create({
      name: patientData.name,
      email: patientData.email,
      age: patientData.age,
      gender: patientData.gender,
      contact: patientData.contact,
      address: patientData.address,
      medical_history: patientData.medical_history,
      symptoms: Array.isArray(patientData.symptoms) ? patientData.symptoms : (patientData.symptoms ? [patientData.symptoms] : []),
      vitals: patientData.vitals || {},
      priority: 'normal', // Default
      profileCompleted: true // Created by admin
      // assignedDoctor will be handled separately or added here if ID known
    });

    console.log(`✅ Patient profile created in PostgreSQL with ID: ${newPatient.id}`);

    // 2. Create User in MongoDB for Auth
    const newUser = new User({
      email: patientData.email,
      password: patientData.password,
      role: 'patient',
      referenceId: newPatient.id
    });

    await newUser.save();
    console.log(`✅ User credentials created in MongoDB with ID: ${newUser._id}`);


    // Assign an available doctor
    // Note: This logic assumes Doctor model is also Postgres or we bridge IDs. 
    // Existing logic used Mongoose Doctor. Let's keep using Mongoose Doctor for availability check 
    // IF doctors are staying in Mongo, BUT the plan was to move data to Postgres.
    // Assuming Doctors are also migrated or we need to look up in Postgres.
    // Let's use the Postgres Doctor model if available, or fallback to the one imported.
    // The previous imports: const Doctor = require("./mongoSchemas/Doctor");
    // We should probably check Postgres Doctor model.

    const { Doctor: DoctorPG } = require("./models");

    try {
      // Find a doctor in Postgres
      const availableDoctor = await DoctorPG.findOne({ where: { availability: 'Available' } });

      if (availableDoctor) {
        // Update patient with assigned doctor
        await newPatient.update({
          assignedDoctor: {
            id: availableDoctor.id, // UUID from Postgres
            name: availableDoctor.name
          }
        });
        console.log(`Assigned doctor ${availableDoctor.name} to patient ${newPatient.name}`);
      } else {
        console.log('No available doctors found for assignment in Postgres');
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
        description: `New patient added: ${newPatient.name}`,
        entityType: 'patient',
        entityId: newPatient.id,
        metadata: {
          patientName: newPatient.name,
          age: newPatient.age,
          gender: newPatient.gender,
          // priority: newPatient.priority, // removed from newPatient object if not explicitly set
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
    const patient = await Patient.findById(req.params.id);
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

    const updatedPatient = await Patient.findByIdAndUpdate(id, updateData, { new: true });
    if (!updatedPatient) {
      return res.status(404).json({ message: "Patient not found" });
    }
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

    // Update patient with all provided fields
    const updatedPatient = await Patient.findByIdAndUpdate(id, updateData, { new: true });
    if (!updatedPatient) {
      return res.status(404).json({ message: "Patient not found" });
    }
    res.json(updatedPatient);
  } catch (error) {
    console.error("Error updating patient:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});

// Doctors Routes
apiRouter.get("/doctors", async (req, res) => {
  try {
    const doctors = await Doctor.find();
    res.json(doctors);
  } catch (error) {
    console.error("Error fetching doctors:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});

apiRouter.get("/doctors/:id", async (req, res) => {
  try {
    const doctor = await Doctor.findById(req.params.id);
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
    const updatedDoctor = await Doctor.findByIdAndUpdate(id, { availability }, { new: true });
    if (!updatedDoctor) {
      return res.status(404).json({ message: "Doctor not found" });
    }
    res.json(updatedDoctor);
  } catch (error) {
    console.error("Error updating doctor:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});

apiRouter.post("/doctors", async (req, res) => {
  try {
    const doctorData = req.body;
    const newDoctor = new Doctor(doctorData);
    await newDoctor.save();
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
    const query = patientName ? { patientName } : {};
    const prescriptions = await Prescription.find(query);
    res.json(prescriptions);
  } catch (error) {
    console.error("Error fetching prescriptions:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});

apiRouter.get("/prescriptions/pending", async (req, res) => {
  try {
    const pendingPrescriptions = await Prescription.find({ status: 'pending' });
    res.json(pendingPrescriptions);
  } catch (error) {
    console.error("Error fetching pending prescriptions:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});

apiRouter.post("/prescriptions", async (req, res) => {
  try {
    const prescriptionData = req.body;
    const newPrescription = new Prescription(prescriptionData);
    await newPrescription.save();
    res.json(newPrescription);
  } catch (error) {
    console.error("Error creating prescription:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});

apiRouter.put("/prescriptions/:id/refill", async (req, res) => {
  try {
    const { id } = req.params;
    const prescription = await Prescription.findById(id);
    if (!prescription) {
      return res.status(404).json({ message: "Prescription not found" });
    }
    prescription.status = 'pending';
    await prescription.save();
    res.json(prescription);
  } catch (error) {
    console.error("Error requesting refill:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});

apiRouter.put("/prescriptions/:id/approve", async (req, res) => {
  try {
    const { id } = req.params;
    const prescription = await Prescription.findById(id);
    if (!prescription) {
      return res.status(404).json({ message: "Prescription not found" });
    }
    prescription.status = 'approved';
    prescription.refillCount += 1;
    await prescription.save();
    res.json(prescription);
  } catch (error) {
    console.error("Error approving prescription:", error);
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
    const notes = await Note.find().sort({ createdAt: -1 });
    res.json(notes);
  } catch (error) {
    console.error("Error fetching notes:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});

apiRouter.post("/notes", async (req, res) => {
  try {
    const noteData = req.body;
    const newNote = new Note(noteData);
    await newNote.save();
    res.json(newNote);
  } catch (error) {
    console.error("Error creating note:", error);
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
    const newAdmin = new Admin(adminData);
    await newAdmin.save();
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
    const admin = await Admin.findById(req.params.id);
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
    const updatedAdmin = await Admin.findByIdAndUpdate(id, updateData, { new: true });
    if (!updatedAdmin) {
      return res.status(404).json({ message: "Admin not found" });
    }
    res.json(updatedAdmin);
  } catch (error) {
    console.error("Error updating admin:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});

apiRouter.delete("/admins/:id", async (req, res) => {
  try {
    const { id } = req.params;
    const deletedAdmin = await Admin.findByIdAndDelete(id);
    if (!deletedAdmin) {
      return res.status(404).json({ message: "Admin not found" });
    }
    res.json({ message: "Admin deleted successfully" });
  } catch (error) {
    console.error("Error deleting admin:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});

// Logs Routes - Get activity logs, login activities, and system logs
apiRouter.get("/logs", async (req, res) => {
  try {
    const { limit = 50, type = 'all' } = req.query;
    const LoginActivity = require("./models/LoginActivity");

    let logs = [];

    // Get activity logs (patient additions, doctor assignments, etc.)
    if (type === 'all' || type === 'activity') {
      const activityLogs = await ActivityLog.find()
        .sort({ createdAt: -1 })
        .limit(parseInt(limit))
        .lean();

      logs = logs.concat(activityLogs.map(activity => ({
        _id: activity._id,
        type: 'activity',
        action: activity.action,
        message: activity.description,
        entityType: activity.entityType,
        entityId: activity.entityId,
        metadata: activity.metadata,
        timestamp: activity.createdAt,
        createdAt: activity.createdAt
      })));
    }

    // Get login activities from MongoDB
    if (type === 'all' || type === 'login') {
      const loginActivities = await LoginActivity.find()
        .sort({ createdAt: -1 })
        .limit(parseInt(limit))
        .lean();

      logs = logs.concat(loginActivities.map(activity => ({
        _id: activity._id,
        type: 'login',
        message: `${activity.role.toUpperCase()} login: ${activity.email}`,
        email: activity.email,
        role: activity.role,
        ipAddress: activity.ipAddress,
        userAgent: activity.userAgent,
        timestamp: activity.createdAt,
        createdAt: activity.createdAt
      })));
    }

    // Sort all logs by timestamp
    logs.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

    // Limit total results
    logs = logs.slice(0, parseInt(limit));

    res.json({
      success: true,
      logs,
      total: logs.length
    });
  } catch (error) {
    console.error("Error fetching logs:", error);
    res.status(500).json({
      success: false,
      message: "Server error",
      error: error.message
    });
  }
});




// Mount authentication routes
app.use("/api/auth", authRoutes);

// Mount API routes
app.use("/api", apiRouter);

// Mount Catalog routes (includes all queue APIs)
app.use("/catalog", catalogRoutes);

// Mount ML Reports routes
app.use("/api/ml-reports", mlReportsRoutes);

// Mount DICOM routes
app.use("/api/dicom", dicomRoutes);

// Mount Viewer routes (DICOM & PDF viewing)
app.use("/api/viewer", viewerRoutes);

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

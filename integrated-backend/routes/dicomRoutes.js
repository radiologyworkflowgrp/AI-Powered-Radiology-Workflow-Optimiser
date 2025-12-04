const express = require('express');
const router = express.Router();
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const DicomScan = require('../models/DicomScan');
const rabbitmq = require('../rabbitmq');
const logger = require('../logger');
const {
    parseDicomFile,
    validateDicomFile,
    generateScanId,
    sanitizeFilename
} = require('../utils/dicomUtils');

// Configure multer for DICOM file uploads
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        const uploadDir = 'uploads/dicom/';
        if (!fs.existsSync(uploadDir)) {
            fs.mkdirSync(uploadDir, { recursive: true });
        }
        cb(null, uploadDir);
    },
    filename: function (req, file, cb) {
        const safeName = sanitizeFilename(file.originalname);
        const timestamp = Date.now();
        cb(null, `${timestamp}-${safeName}`);
    }
});

// Accept all files - DICOM files often don't have standard extensions
// Validation will be done by checking file content
const upload = multer({
    storage: storage,
    limits: {
        fileSize: 100 * 1024 * 1024 // 100MB max file size
    }
});

/**
 * POST /api/dicom/upload
 * Upload DICOM file and process through DuoFormer
 */
router.post('/upload', upload.single('dicomFile'), async (req, res) => {
    console.log('=== DICOM UPLOAD ROUTE CALLED ===');
    console.log('File:', req.file ? req.file.filename : 'NO FILE');
    console.log('Body:', req.body);

    try {
        if (!req.file) {
            console.log('ERROR: No file uploaded');
            return res.status(400).json({
                success: false,
                message: 'No DICOM file uploaded'
            });
        }

        const { patientId, patientName, patientEmail, scanType, notes, deviceId, deviceModel } = req.body;

        // Validate required fields
        if (!patientId || !patientName) {
            // Clean up uploaded file
            fs.unlinkSync(req.file.path);
            return res.status(400).json({
                success: false,
                message: 'Patient ID and name are required'
            });
        }

        // Validate DICOM file
        const fileBuffer = fs.readFileSync(req.file.path);
        if (!validateDicomFile(fileBuffer)) {
            // Clean up invalid file
            fs.unlinkSync(req.file.path);
            return res.status(400).json({
                success: false,
                message: 'Invalid DICOM file format'
            });
        }

        // Parse DICOM metadata
        let dicomMetadata = {};
        try {
            dicomMetadata = await parseDicomFile(req.file.path);
            logger.info('DICOM metadata extracted successfully', { scanId: generateScanId() });
        } catch (error) {
            logger.warn('Failed to extract DICOM metadata, continuing with basic info', { error: error.message });
        }

        // Generate unique scan ID
        const scanId = generateScanId();

        // Convert DICOM to image for DuoFormer processing
        const { spawn } = require('child_process');
        let imageBase64 = null;
        let conversionMetadata = {};

        try {
            const pythonScript = path.join(__dirname, '../prioritization-ml/dicom_converter.py');
            const conversionResult = await new Promise((resolve, reject) => {
                const python = spawn('python', [
                    '-c',
                    `
import sys
sys.path.append('${path.join(__dirname, '../prioritization-ml').replace(/\\/g, '\\\\')}')
from dicom_converter import convert_dicom_to_image
import json
import base64

try:
    img_bytes, metadata = convert_dicom_to_image('${req.file.path.replace(/\\/g, '\\\\')}')
    result = {
        'success': True,
        'image_base64': base64.b64encode(img_bytes).decode('utf-8'),
        'metadata': metadata
    }
    print(json.dumps(result))
except Exception as e:
    result = {'success': False, 'error': str(e)}
    print(json.dumps(result))
`
                ]);

                let output = '';
                let errorOutput = '';

                python.stdout.on('data', (data) => {
                    output += data.toString();
                });

                python.stderr.on('data', (data) => {
                    errorOutput += data.toString();
                });

                python.on('close', (code) => {
                    if (code !== 0) {
                        reject(new Error(`DICOM conversion failed: ${errorOutput}`));
                    } else {
                        try {
                            const result = JSON.parse(output.trim());
                            if (result.success) {
                                resolve(result);
                            } else {
                                reject(new Error(result.error));
                            }
                        } catch (e) {
                            reject(new Error(`Failed to parse conversion output: ${e.message}`));
                        }
                    }
                });
            });

            imageBase64 = conversionResult.image_base64;
            conversionMetadata = conversionResult.metadata;
            logger.info('DICOM converted to image successfully', { scanId });

        } catch (conversionError) {
            logger.error('DICOM conversion error', { error: conversionError.message, scanId });
            fs.unlinkSync(req.file.path);
            return res.status(500).json({
                success: false,
                message: 'Failed to convert DICOM to image',
                error: conversionError.message
            });
        }

        // Create DICOM scan record
        const dicomScan = new DicomScan({
            patientId,
            patientName,
            patientEmail: patientEmail || '',
            scanId,
            scanType: scanType || dicomMetadata.modality || conversionMetadata.modality || 'CT',
            scanDescription: notes || dicomMetadata.studyDescription || '',
            dicomFilePath: req.file.path,
            dicomFileName: req.file.filename,
            fileSize: req.file.size,
            dicomMetadata: {
                ...dicomMetadata,
                ...conversionMetadata,
                // Remove NaN values
                rows: undefined,
                columns: undefined
            },
            uploadDevice: {
                deviceId: deviceId || 'doctor-portal',
                deviceModel: deviceModel || 'Web Browser',
                deviceType: req.body.deviceType || 'mobile', // Use 'mobile' as default (valid enum)
                uploadIP: req.ip || req.connection.remoteAddress,
                userAgent: req.get('user-agent')
            },
            notes: notes || '',
            uploadedBy: req.body.uploadedBy || 'doctor'
        });

        await dicomScan.save();
        logger.info('DICOM scan saved to database', { scanId, patientId });

        // Send to appropriate ML worker for analysis based on scan type
        try {
            const queueData = {
                patient_id: patientId,
                patient_name: patientName,
                scan_id: scanId,
                image_name: req.file.originalname,
                image_base64: imageBase64,
                scan_type: dicomScan.scanType,
                dicom_metadata: conversionMetadata,
                timestamp: new Date().toISOString(),
                source: 'dicom_upload'
            };

            // Determine which queue to use based on scan type
            let queueName;
            if (scanType === 'MRI') {
                queueName = 'mri_analysis_queue';
                logger.info('Routing to MRI worker for brain tumor analysis');
            } else if (scanType === 'X-RAY') {
                queueName = 'xray_analysis_queue';
                logger.info('Routing to DuoFormer worker for chest X-ray analysis');
            } else {
                // Default to X-ray queue for other types
                queueName = 'xray_analysis_queue';
                logger.info(`Routing ${scanType} to default X-ray queue`);
            }

            // Send to appropriate queue
            const amqp = require('amqplib');
            const connection = await amqp.connect(process.env.RABBITMQ_URL || 'amqp://localhost:5672');
            const channel = await connection.createChannel();

            await channel.assertQueue(queueName, { durable: true });
            channel.sendToQueue(queueName, Buffer.from(JSON.stringify(queueData)), {
                persistent: true
            });

            await channel.close();
            await connection.close();

            // Update scan status
            await dicomScan.markAsQueued();

            logger.info(`DICOM scan sent to ${queueName}`, { scanId });
        } catch (queueError) {
            logger.error('Failed to queue DICOM scan for analysis', { error: queueError.message, scanId });
            // Don't fail the upload, just log the error
        }

        res.status(201).json({
            success: true,
            message: 'DICOM file uploaded and sent for analysis',
            scan: {
                scanId: dicomScan.scanId,
                patientId: dicomScan.patientId,
                patientName: dicomScan.patientName,
                scanType: dicomScan.scanType,
                status: dicomScan.status,
                uploadedAt: dicomScan.createdAt,
                fileSize: dicomScan.fileSize,
                metadata: dicomScan.dicomMetadata
            }
        });

    } catch (error) {
        logger.error('DICOM upload error', {
            error: error.message,
            stack: error.stack,
            file: req.file ? req.file.filename : 'no file'
        });

        // Clean up file if it exists
        if (req.file && fs.existsSync(req.file.path)) {
            try {
                fs.unlinkSync(req.file.path);
            } catch (cleanupError) {
                logger.error('File cleanup error', { error: cleanupError.message });
            }
        }

        // Send detailed error response
        res.status(500).json({
            success: false,
            message: 'Failed to upload DICOM file',
            error: error.message,
            details: process.env.NODE_ENV === 'development' ? error.stack : undefined
        });
    }
});

/**
 * GET /api/dicom/:scanId
 * Get DICOM scan metadata
 */
router.get('/:scanId', async (req, res) => {
    try {
        const { scanId } = req.params;

        const scan = await DicomScan.findOne({ scanId });
        if (!scan) {
            return res.status(404).json({
                success: false,
                message: 'DICOM scan not found'
            });
        }

        res.json({
            success: true,
            scan
        });
    } catch (error) {
        logger.error('Error fetching DICOM scan', { error: error.message });
        res.status(500).json({
            success: false,
            message: 'Failed to fetch DICOM scan',
            error: error.message
        });
    }
});

/**
 * GET /api/dicom/:scanId/file
 * Serve DICOM file for viewing
 */
router.get('/:scanId/file', async (req, res) => {
    try {
        const { scanId } = req.params;

        const scan = await DicomScan.findOne({ scanId });
        if (!scan) {
            return res.status(404).json({
                success: false,
                message: 'DICOM scan not found'
            });
        }

        // Check if file exists
        if (!fs.existsSync(scan.dicomFilePath)) {
            return res.status(404).json({
                success: false,
                message: 'DICOM file not found on server'
            });
        }

        // Set appropriate headers for DICOM file
        res.setHeader('Content-Type', 'application/dicom');
        res.setHeader('Content-Disposition', `inline; filename="${scan.dicomFileName}"`);

        // Stream the file
        const fileStream = fs.createReadStream(scan.dicomFilePath);
        fileStream.pipe(res);

    } catch (error) {
        logger.error('Error serving DICOM file', { error: error.message });
        res.status(500).json({
            success: false,
            message: 'Failed to serve DICOM file',
            error: error.message
        });
    }
});

/**
 * GET /api/dicom/patient/:patientId
 * Get all DICOM scans for a patient
 */
router.get('/patient/:patientId', async (req, res) => {
    try {
        const { patientId } = req.params;

        const scans = await DicomScan.getByPatientId(patientId);

        res.json({
            success: true,
            patientId,
            total: scans.length,
            scans
        });
    } catch (error) {
        logger.error('Error fetching patient DICOM scans', { error: error.message });
        res.status(500).json({
            success: false,
            message: 'Failed to fetch patient scans',
            error: error.message
        });
    }
});

/**
 * GET /api/dicom/list/recent
 * Get recent DICOM scans
 */
router.get('/list/recent', async (req, res) => {
    try {
        const limit = parseInt(req.query.limit) || 10;
        const scans = await DicomScan.getRecentScans(limit);

        res.json({
            success: true,
            total: scans.length,
            scans
        });
    } catch (error) {
        logger.error('Error fetching recent DICOM scans', { error: error.message });
        res.status(500).json({
            success: false,
            message: 'Failed to fetch recent scans',
            error: error.message
        });
    }
});

/**
 * GET /api/dicom/list/pending
 * Get pending DICOM scans
 */
router.get('/list/pending', async (req, res) => {
    try {
        const scans = await DicomScan.getPendingScans();

        res.json({
            success: true,
            total: scans.length,
            scans
        });
    } catch (error) {
        logger.error('Error fetching pending DICOM scans', { error: error.message });
        res.status(500).json({
            success: false,
            message: 'Failed to fetch pending scans',
            error: error.message
        });
    }
});

/**
 * PUT /api/dicom/:scanId/status
 * Update DICOM scan status
 */
router.put('/:scanId/status', async (req, res) => {
    try {
        const { scanId } = req.params;
        const { status, errorMessage, mlReportId } = req.body;

        const scan = await DicomScan.findOne({ scanId });
        if (!scan) {
            return res.status(404).json({
                success: false,
                message: 'DICOM scan not found'
            });
        }

        if (status === 'completed') {
            await scan.markAsProcessed(mlReportId);
        } else {
            await scan.updateStatus(status, errorMessage);
        }

        res.json({
            success: true,
            message: 'Scan status updated',
            scan
        });
    } catch (error) {
        logger.error('Error updating DICOM scan status', { error: error.message });
        res.status(500).json({
            success: false,
            message: 'Failed to update scan status',
            error: error.message
        });
    }
});

module.exports = router;

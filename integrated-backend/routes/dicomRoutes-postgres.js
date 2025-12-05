const express = require('express');
const router = express.Router();
const multer = require('multer');
const fs = require('fs');
const { DICOMImage } = require('../models');
const Patient = require('../mongoSchemas/Patient');
const { extractDICOMMetadata, isValidDICOM, formatFileSize } = require('../utils/dicomStorage');
const logger = require('../logger');
const { v4: uuidv4 } = require('uuid');
const rabbitmq = require('../rabbitmq');

// Configure multer for memory storage (we'll store in DB, not filesystem)
const upload = multer({
    storage: multer.memoryStorage(),
    limits: {
        fileSize: 100 * 1024 * 1024 // 100MB max
    }
});

/**
 * POST /api/dicom/upload
 * Upload DICOM file and store in database
 */
router.post('/upload', upload.single('dicomFile'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({
                success: false,
                message: 'No DICOM file uploaded'
            });
        }

        const { patientId, scanType, notes } = req.body;

        // Validate required fields
        if (!patientId) {
            return res.status(400).json({
                success: false,
                message: 'Patient ID is required'
            });
        }

        // Verify patient exists
        const patient = await Patient.findById(patientId);
        if (!patient) {
            return res.status(404).json({
                success: false,
                message: 'Patient not found'
            });
        }

        // Validate DICOM file
        if (!isValidDICOM(req.file.buffer)) {
            return res.status(400).json({
                success: false,
                message: 'Invalid DICOM file format'
            });
        }

        // Extract DICOM metadata
        const metadata = extractDICOMMetadata(req.file.buffer);
        logger.info('DICOM metadata extracted', { patientId, metadataKeys: Object.keys(metadata) });

        // Generate unique scan ID
        const scanId = `DICOM_${Date.now()}_${uuidv4().substring(0, 8)}`;

        // Create DICOM image record
        const dicomImage = await DICOMImage.create({
            patient_id: patientId,
            scan_id: scanId,
            modality: scanType || metadata.modality || 'CT',
            study_date: metadata.studyDate ? new Date(metadata.studyDate) : new Date(),
            series_description: metadata.seriesDescription || notes || '',
            image_data: req.file.buffer,
            metadata: metadata,
            file_size: req.file.size,
            thumbnail: null // TODO: Generate thumbnail
        });

        logger.info('DICOM image stored in database', {
            scanId,
            patientId,
            size: formatFileSize(req.file.size)
        });

        // Publish to RabbitMQ for ML analysis
        try {
            const scanType = (dicomImage.modality || 'CT').toUpperCase();
            let queueName = 'xray_analysis_queue'; // Default

            if (scanType === 'MRI') {
                queueName = 'mri_analysis_queue';
            } else if (scanType === 'X-RAY' || scanType === 'DX' || scanType === 'CR') {
                queueName = 'xray_analysis_queue';
            }

            // Construct payload compatible with ML workers
            const queueData = {
                patient_id: dicomImage.patient_id, // Postgres UUID
                scan_id: dicomImage.scan_id,
                image_name: req.file.originalname,
                // We might need to send base64 or a path provided the worker can access it
                // For now, let's assume the worker can download from API or we send base64 if small enough.
                // The previous route sent base64. Postgres route stored buffer in DB.
                // It's expensive to send buffer in queue. 
                // Better: generic worker fetches by ID. 
                // BUT current ML worker expects 'image_base64'.

                // Let's send a flag or URL, but if worker expects base64, we might need to send it.
                // Converting buffer to base64 for queue:
                image_base64: req.file.buffer.toString('base64'),
                scan_type: scanType,
                timestamp: new Date().toISOString(),
                source: 'dicom_upload'
            };

            const channel = rabbitmq.channel;
            await channel.assertQueue(queueName, { durable: true });
            channel.sendToQueue(queueName, Buffer.from(JSON.stringify(queueData)), {
                persistent: true
            });
            logger.info(`Published to ${queueName}`, { scanId });
        } catch (mqError) {
            logger.error('Failed to publish to RabbitMQ', { error: mqError.message });
            // Don't fail the upload just because MQ failed, but user should know status might be delayed
        }

        res.status(201).json({
            success: true,
            message: 'DICOM file uploaded and stored successfully. Analysis scheduled.',
            scan: {
                id: dicomImage.id,
                scanId: dicomImage.scan_id,
                patientId: dicomImage.patient_id,
                modality: dicomImage.modality,
                studyDate: dicomImage.study_date,
                fileSize: formatFileSize(dicomImage.file_size),
                uploadedAt: dicomImage.created_at
            }
        });

    } catch (error) {
        logger.error('DICOM upload error', {
            error: error.message,
            stack: error.stack
        });

        res.status(500).json({
            success: false,
            message: 'Failed to upload DICOM file',
            error: error.message
        });
    }
});

/**
 * GET /api/dicom/:id
 * Retrieve DICOM image binary data
 */
router.get('/:id', async (req, res) => {
    try {
        const { id } = req.params;

        const dicomImage = await DICOMImage.findByPk(id);
        if (!dicomImage) {
            return res.status(404).json({
                success: false,
                message: 'DICOM image not found'
            });
        }

        // Set appropriate headers
        res.setHeader('Content-Type', 'application/dicom');
        res.setHeader('Content-Length', dicomImage.image_data.length);
        res.setHeader('Content-Disposition', `inline; filename="${dicomImage.scan_id}.dcm"`);

        // Send binary data
        res.send(dicomImage.image_data);

    } catch (error) {
        logger.error('Error retrieving DICOM image', { error: error.message });
        res.status(500).json({
            success: false,
            message: 'Failed to retrieve DICOM image',
            error: error.message
        });
    }
});

/**
 * GET /api/dicom/:id/metadata
 * Get DICOM metadata without binary data
 */
router.get('/:id/metadata', async (req, res) => {
    try {
        const { id } = req.params;

        const dicomImage = await DICOMImage.findByPk(id, {
            attributes: { exclude: ['image_data', 'thumbnail'] },
            include: [{
                model: Patient,
                as: 'patient',
                attributes: ['id', 'name', 'email']
            }]
        });

        if (!dicomImage) {
            return res.status(404).json({
                success: false,
                message: 'DICOM image not found'
            });
        }

        res.json({
            success: true,
            dicom: {
                ...dicomImage.toJSON(),
                file_size: formatFileSize(dicomImage.file_size)
            }
        });

    } catch (error) {
        logger.error('Error fetching DICOM metadata', { error: error.message });
        res.status(500).json({
            success: false,
            message: 'Failed to fetch DICOM metadata',
            error: error.message
        });
    }
});

/**
 * GET /api/dicom/patient/:patientId
 * List all DICOM images for a patient
 */
router.get('/patient/:patientId', async (req, res) => {
    try {
        const { patientId } = req.params;

        const dicomImages = await DICOMImage.findAll({
            where: { patient_id: patientId },
            attributes: { exclude: ['image_data', 'thumbnail'] },
            order: [['created_at', 'DESC']]
        });

        res.json({
            success: true,
            patientId,
            total: dicomImages.length,
            scans: dicomImages.map(img => ({
                ...img.toJSON(),
                file_size: formatFileSize(img.file_size)
            }))
        });

    } catch (error) {
        logger.error('Error fetching patient DICOM images', { error: error.message });
        res.status(500).json({
            success: false,
            message: 'Failed to fetch patient DICOM images',
            error: error.message
        });
    }
});

/**
 * GET /api/dicom/:id/thumbnail
 * Get thumbnail preview (if available)
 */
router.get('/:id/thumbnail', async (req, res) => {
    try {
        const { id } = req.params;

        const dicomImage = await DICOMImage.findByPk(id, {
            attributes: ['id', 'scan_id', 'thumbnail']
        });

        if (!dicomImage) {
            return res.status(404).json({
                success: false,
                message: 'DICOM image not found'
            });
        }

        if (!dicomImage.thumbnail) {
            return res.status(404).json({
                success: false,
                message: 'Thumbnail not available for this DICOM image'
            });
        }

        res.setHeader('Content-Type', 'image/png');
        res.send(dicomImage.thumbnail);

    } catch (error) {
        logger.error('Error retrieving DICOM thumbnail', { error: error.message });
        res.status(500).json({
            success: false,
            message: 'Failed to retrieve thumbnail',
            error: error.message
        });
    }
});

/**
 * DELETE /api/dicom/:id
 * Delete DICOM image
 */
router.delete('/:id', async (req, res) => {
    try {
        const { id } = req.params;

        const dicomImage = await DICOMImage.findByPk(id);
        if (!dicomImage) {
            return res.status(404).json({
                success: false,
                message: 'DICOM image not found'
            });
        }

        await dicomImage.destroy();
        logger.info('DICOM image deleted', { id, scanId: dicomImage.scan_id });

        res.json({
            success: true,
            message: 'DICOM image deleted successfully'
        });

    } catch (error) {
        logger.error('Error deleting DICOM image', { error: error.message });
        res.status(500).json({
            success: false,
            message: 'Failed to delete DICOM image',
            error: error.message
        });
    }
});

module.exports = router;

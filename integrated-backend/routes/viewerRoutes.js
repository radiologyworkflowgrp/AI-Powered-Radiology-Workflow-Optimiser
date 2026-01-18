const express = require('express');
const router = express.Router();
const { MLReport, DICOMImage } = require('../models');
const { checkPatientAccess } = require('../middleware/accessControl');
const fs = require('fs');
const path = require('path');

/**
 * GET /api/viewer/dicom/:id
 * Serve DICOM file for OHIF viewer
 */
router.get('/dicom/:id', checkPatientAccess, async (req, res) => {
    try {
        const { id } = req.params;

        // Fetch DICOM image from PostgreSQL
        const dicomImage = await DICOMImage.findByPk(id);

        if (!dicomImage) {
            return res.status(404).json({ message: 'DICOM image not found' });
        }

        // Check if user has access to this patient's data
        // (checkPatientAccess middleware already handles this via patient_id)

        // Serve DICOM file
        res.set({
            'Content-Type': 'application/dicom',
            'Content-Disposition': `inline; filename="${dicomImage.scan_id}.dcm"`,
            'Content-Length': dicomImage.file_size
        });

        res.send(dicomImage.image_data);
    } catch (error) {
        console.error('Error serving DICOM:', error);
        res.status(500).json({ message: 'Error serving DICOM file', error: error.message });
    }
});

/**
 * GET /api/viewer/dicom/metadata/:id
 * Get DICOM metadata for display
 */
router.get('/dicom/metadata/:id', async (req, res) => {
    try {
        const { id } = req.params;

        const dicomImage = await DICOMImage.findByPk(id, {
            attributes: ['id', 'patient_id', 'scan_id', 'modality', 'study_date', 'series_description', 'metadata', 'file_size', 'created_at']
        });

        if (!dicomImage) {
            return res.status(404).json({ message: 'DICOM image not found' });
        }

        res.json({
            message: 'DICOM metadata retrieved',
            dicom: dicomImage.toJSON()
        });
    } catch (error) {
        console.error('Error fetching DICOM metadata:', error);
        res.status(500).json({ message: 'Error fetching metadata', error: error.message });
    }
});

/**
 * GET /api/viewer/dicom/patient/:patientId
 * Get all DICOM images for a patient
 */
router.get('/dicom/patient/:patientId', checkPatientAccess, async (req, res) => {
    try {
        const { patientId } = req.params;

        const dicomImages = await DICOMImage.findAll({
            where: { patient_id: patientId },
            attributes: ['id', 'scan_id', 'modality', 'study_date', 'series_description', 'file_size', 'created_at'],
            order: [['created_at', 'DESC']]
        });

        res.json({
            message: 'DICOM images retrieved',
            dicomImages: dicomImages.map(d => d.toJSON()),
            total: dicomImages.length
        });
    } catch (error) {
        console.error('Error fetching patient DICOM images:', error);
        res.status(500).json({ message: 'Error fetching DICOM images', error: error.message });
    }
});

/**
 * GET /api/viewer/report/:id/pdf
 * Serve PDF report file
 */
router.get('/report/:id/pdf', async (req, res) => {
    try {
        const { id } = req.params;

        // Fetch ML report from PostgreSQL
        const report = await MLReport.findByPk(id);

        if (!report) {
            return res.status(404).json({ message: 'Report not found' });
        }

        if (!report.pdf_path) {
            return res.status(404).json({ message: 'PDF not available for this report' });
        }

        // Check if PDF file exists
        const pdfPath = path.resolve(report.pdf_path);
        if (!fs.existsSync(pdfPath)) {
            return res.status(404).json({ message: 'PDF file not found on server' });
        }

        // Serve PDF file
        res.set({
            'Content-Type': 'application/pdf',
            'Content-Disposition': `inline; filename="report-${report.id}.pdf"`,
        });

        const fileStream = fs.createReadStream(pdfPath);
        fileStream.pipe(res);
    } catch (error) {
        console.error('Error serving PDF:', error);
        res.status(500).json({ message: 'Error serving PDF', error: error.message });
    }
});

/**
 * GET /api/viewer/report/patient/:patientId
 * Get all reports for a patient
 */
router.get('/report/patient/:patientId', checkPatientAccess, async (req, res) => {
    try {
        const { patientId } = req.params;

        const reports = await MLReport.findAll({
            where: { patient_id: patientId },
            attributes: ['id', 'patient_name', 'report_type', 'report_status', 'confidence_score', 'pdf_path', 'created_at'],
            order: [['created_at', 'DESC']]
        });

        res.json({
            message: 'Reports retrieved',
            reports: reports.map(r => {
                const reportJson = r.toJSON();
                reportJson.has_pdf = !!reportJson.pdf_path;
                return reportJson;
            }),
            total: reports.length
        });
    } catch (error) {
        console.error('Error fetching patient reports:', error);
        res.status(500).json({ message: 'Error fetching reports', error: error.message });
    }
});

/**
 * GET /api/viewer/ohif/:patientId
 * Get OHIF viewer URL for patient's DICOM studies
 */
router.get('/ohif/:patientId', async (req, res) => {
    try {
        const { patientId } = req.params;

        // Get all DICOM studies for patient
        const dicomImages = await DICOMImage.findAll({
            where: { patient_id: patientId },
            attributes: ['id', 'scan_id', 'modality', 'study_date'],
            order: [['study_date', 'DESC']]
        });

        if (dicomImages.length === 0) {
            return res.status(404).json({ message: 'No DICOM studies found for this patient' });
        }

        // Return OHIF viewer configuration
        res.json({
            message: 'OHIF viewer data',
            studies: dicomImages.map(d => ({
                id: d.id,
                scanId: d.scan_id,
                modality: d.modality,
                studyDate: d.study_date,
                viewerUrl: `/api/viewer/dicom/${d.id}`
            })),
            total: dicomImages.length
        });
    } catch (error) {
        console.error('Error fetching OHIF data:', error);
        res.status(500).json({ message: 'Error fetching OHIF data', error: error.message });
    }
});

module.exports = router;

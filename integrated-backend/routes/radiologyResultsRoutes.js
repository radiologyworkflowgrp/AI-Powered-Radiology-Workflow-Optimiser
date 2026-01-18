const express = require('express');
const router = express.Router();
const RadiologyResult = require('../models/RadiologyResult');

// GET /api/radiology-results - Get all radiology results
router.get('/', async (req, res) => {
    try {
        const results = await RadiologyResult.find()
            .sort({ processedAt: -1 })
            .limit(100);

        res.json({
            message: 'Radiology results retrieved successfully',
            results,
            total: results.length
        });
    } catch (error) {
        console.error('Error fetching radiology results:', error);
        res.status(500).json({
            message: 'Failed to fetch radiology results',
            error: error.message
        });
    }
});

// GET /api/radiology-results/patient/:patientId - Get results by patient ID
router.get('/patient/:patientId', async (req, res) => {
    try {
        const results = await RadiologyResult.find({ patientId: req.params.patientId })
            .sort({ processedAt: -1 });

        res.json({
            message: 'Patient radiology results retrieved successfully',
            results,
            total: results.length
        });
    } catch (error) {
        console.error('Error fetching patient radiology results:', error);
        res.status(500).json({
            message: 'Failed to fetch patient radiology results',
            error: error.message
        });
    }
});

// GET /api/radiology-results/scan/:scanId - Get results by scan ID
router.get('/scan/:scanId', async (req, res) => {
    try {
        const results = await RadiologyResult.find({ scanId: req.params.scanId })
            .sort({ processedAt: -1 });

        res.json({
            message: 'Scan radiology results retrieved successfully',
            results,
            total: results.length
        });
    } catch (error) {
        console.error('Error fetching scan radiology results:', error);
        res.status(500).json({
            message: 'Failed to fetch scan radiology results',
            error: error.message
        });
    }
});

// GET /api/radiology-results/:id - Get result by ID
router.get('/:id', async (req, res) => {
    try {
        const result = await RadiologyResult.findById(req.params.id);

        if (!result) {
            return res.status(404).json({
                message: 'Radiology result not found'
            });
        }

        res.json({
            message: 'Radiology result retrieved successfully',
            result
        });
    } catch (error) {
        console.error('Error fetching radiology result:', error);
        res.status(500).json({
            message: 'Failed to fetch radiology result',
            error: error.message
        });
    }
});

module.exports = router;

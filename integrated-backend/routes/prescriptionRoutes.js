const express = require('express');
const router = express.Router();
const { Prescription } = require('../models');

// GET all prescriptions (with optional filter by patientName)
router.get('/', async (req, res) => {
    try {
        const { patientName } = req.query;
        const whereClause = {};
        if (patientName) {
            whereClause.patientName = patientName;
        }

        const prescriptions = await Prescription.findAll({
            where: whereClause,
            order: [['datePrescribed', 'DESC']]
        });
        res.json(prescriptions);
    } catch (error) {
        console.error('Error fetching prescriptions:', error);
        res.status(500).json({ message: 'Error fetching prescriptions', error: error.message });
    }
});

// POST create a new prescription
router.post('/', async (req, res) => {
    try {
        const { patientName, medication, dosage, frequency, duration, prescribedBy } = req.body;

        const newPrescription = await Prescription.create({
            patientName,
            medication,
            dosage,
            frequency,
            duration,
            prescribedBy, // Assuming model handles this or we need to add it
            status: 'approved', // Auto-approve doctor creations
            datePrescribed: new Date()
        });

        res.status(201).json(newPrescription);
    } catch (error) {
        console.error('Error creating prescription:', error);
        res.status(500).json({ message: 'Error creating prescription', error: error.message });
    }
});

// POST request refill
router.post('/:id/refill', async (req, res) => {
    try {
        const { id } = req.params;
        const prescription = await Prescription.findByPk(id);

        if (!prescription) {
            return res.status(404).json({ message: 'Prescription not found' });
        }

        // Logic for refill: update status to pending?
        // For now, let's just increment refill count or set status
        prescription.status = 'pending';
        await prescription.save();

        res.json({ message: 'Refill requested', prescription });
    } catch (error) {
        console.error('Error requesting refill:', error);
        res.status(500).json({ message: 'Error requesting refill', error: error.message });
    }
});

module.exports = router;

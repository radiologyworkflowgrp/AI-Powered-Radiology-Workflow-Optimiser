const mongoose = require('mongoose');

const PrescriptionSchema = new mongoose.Schema({
    patientName: {
        type: String,
        required: true
    },
    medication: {
        type: String,
        required: true
    },
    dosage: {
        type: String,
        required: true
    },
    frequency: {
        type: String,
        required: true
    },
    duration: {
        type: String,
        required: true
    },
    status: {
        type: String,
        default: 'active',
        enum: ['active', 'pending', 'approved', 'rejected', 'expired']
    },
    refillCount: {
        type: Number,
        default: 0
    }
}, {
    timestamps: true
});

module.exports = mongoose.model('Prescription', PrescriptionSchema);

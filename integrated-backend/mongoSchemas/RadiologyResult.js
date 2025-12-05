const mongoose = require('mongoose');

const RadiologyResultSchema = new mongoose.Schema({
    patientId: {
        type: String,
        required: true
    },
    patientName: {
        type: String,
        required: true
    },
    testType: {
        type: String,
        required: true
    },
    result: {
        type: String,
        required: true
    },
    notes: {
        type: String
    },
    imageUrl: {
        type: String
    },
    dicom_image_id: {
        type: String,
        default: null
    }
}, {
    timestamps: true
});

module.exports = mongoose.model('RadiologyResult', RadiologyResultSchema);

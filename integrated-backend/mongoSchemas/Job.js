const mongoose = require('mongoose');

const JobSchema = new mongoose.Schema({
    type: {
        type: String,
        required: true
    },
    status: {
        type: String,
        default: 'pending',
        enum: ['pending', 'processing', 'completed', 'failed']
    },
    data: {
        type: mongoose.Schema.Types.Mixed,
        default: {}
    },
    result: {
        type: mongoose.Schema.Types.Mixed,
        default: {}
    }
}, {
    timestamps: true
});

module.exports = mongoose.model('Job', JobSchema);

const mongoose = require('mongoose');

const ActivityLogSchema = new mongoose.Schema({
    action: {
        type: String,
        required: true
    },
    description: {
        type: String,
        required: true
    },
    entityType: {
        type: String
    },
    entityId: {
        type: String
    },
    metadata: {
        type: mongoose.Schema.Types.Mixed,
        default: {}
    }
}, {
    timestamps: true
});

module.exports = mongoose.model('ActivityLog', ActivityLogSchema);

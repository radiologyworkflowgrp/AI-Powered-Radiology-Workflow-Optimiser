const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');

const DoctorSchema = new mongoose.Schema({
    name: {
        type: String,
        required: true
    },
    email: {
        type: String,
        required: true,
        unique: true,
        lowercase: true
    },
    password: {
        type: String,
        required: true
    },
    specialty: {
        type: String,
        default: 'General Medicine'
    },
    availability: {
        type: String,
        default: 'Available',
        enum: ['Available', 'Busy', 'Off Duty']
    },
    profileCompleted: {
        type: Boolean,
        default: false
    }
}, {
    timestamps: true
});

// Hash password before saving
DoctorSchema.pre('save', async function () {
    if (!this.isModified('password')) {
        return;
    }

    try {
        const salt = await bcrypt.genSalt(10);
        this.password = await bcrypt.hash(this.password, salt);
    } catch (error) {
        throw error;
    }
});

// Method to compare password
DoctorSchema.methods.comparePassword = async function (candidatePassword) {
    return await bcrypt.compare(candidatePassword, this.password);
};

module.exports = mongoose.model('Doctor', DoctorSchema);

const mongoose = require('mongoose');
const User = require('./mongoSchemas/User');
const { Patient } = require('./models');
require('dotenv').config();

const createPatient = async () => {
    try {
        // Connect to MongoDB
        await mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/radiology_hospital');
        console.log('✅ Connected to MongoDB');

        const email = 'patient2@hospital.com';
        const password = 'patient123';
        const name = 'Jane Smith';

        // Create Patient in PostgreSQL
        let patientProfile = await Patient.findOne({ where: { email } });
        if (!patientProfile) {
            patientProfile = await Patient.create({
                name,
                email,
                password: 'placeholder',
                age: 28,
                gender: 'Female',
                phone: '+1-555-0103',
                address: '456 Oak Avenue, Medical District',
                medicalHistory: 'No significant medical history',
                status: 'Admitted',
                assignedDoctor: null
            });
            console.log('✅ Patient profile created in PostgreSQL');
        } else {
            console.log('ℹ️  Patient profile already exists');
        }

        // Create User in MongoDB
        let patientUser = await User.findOne({ email });
        if (!patientUser) {
            patientUser = new User({
                email,
                password,
                plainPassword: password,
                role: 'patient',
                referenceId: patientProfile.id.toString()
            });
            await patientUser.save();
            console.log('✅ Patient user created in MongoDB');
        } else {
            console.log('ℹ️  Patient user already exists in MongoDB');
        }

        console.log('\n🎉 Patient created successfully!');
        console.log('📧 Email:', email);
        console.log('🔑 Password:', password);
        console.log('👤 Name:', name);

        process.exit(0);
    } catch (error) {
        console.error('❌ Error creating patient:', error);
        process.exit(1);
    }
};

createPatient();

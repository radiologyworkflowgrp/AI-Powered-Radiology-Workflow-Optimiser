const mongoose = require('mongoose');
const User = require('./mongoSchemas/User');
const { Patient } = require('./models');
require('dotenv').config();

const createPatient = async () => {
    try {
        // Connect to MongoDB
        await mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/radiology_hospital');
        console.log('✅ Connected to MongoDB');

        // Create Patient in PostgreSQL if not exists
        let patientProfile = await Patient.findOne({ where: { email: 'patient@hospital.com' } });
        if (!patientProfile) {
            patientProfile = await Patient.create({
                name: 'John Doe',
                email: 'patient@hospital.com',
                password: 'placeholder', // Not used for auth
                age: 35,
                gender: 'Male',
                phone: '1234567890',
                address: '123 Main St',
                status: 'Admitted'
            });
            console.log('✅ Patient profile created in PostgreSQL');
        } else {
            console.log('ℹ️  Patient profile already exists in PostgreSQL');
        }

        // Create User for Patient in MongoDB
        let patientUser = await User.findOne({ email: 'patient@hospital.com' });
        if (!patientUser) {
            patientUser = new User({
                email: 'patient@hospital.com',
                password: 'patient123',
                plainPassword: 'patient123',
                role: 'patient',
                referenceId: patientProfile.id.toString()
            });
            await patientUser.save();
            console.log('✅ Patient user created in MongoDB: patient@hospital.com / patient123');
        } else {
            console.log('ℹ️  Patient user already exists in MongoDB');
        }

        console.log('\n🎉 Patient creation complete!');
        console.log('\n📝 Login Credentials:');
        console.log('   Patient: patient@hospital.com / patient123');

        process.exit(0);
    } catch (error) {
        console.error('❌ Error creating patient:', error);
        process.exit(1);
    }
};

createPatient();

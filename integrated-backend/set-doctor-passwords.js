require('dotenv').config();
const mongoose = require('mongoose');
const Doctor = require('./models/Doctor');

const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017/radiology_hospital';

async function setDoctorPasswords() {
    try {
        await mongoose.connect(MONGODB_URI);
        console.log('✓ Connected to MongoDB\n');

        const doctors = await Doctor.find();
        console.log(`Found ${doctors.length} doctors\n`);

        for (const doctor of doctors) {
            // Set password to 'doctor123'
            doctor.password = 'doctor123';
            await doctor.save();
            console.log(`✓ Set password for ${doctor.name} (${doctor.email})`);
        }

        console.log('\n' + '='.repeat(60));
        console.log('✓ ALL DOCTOR PASSWORDS SET TO: doctor123');
        console.log('='.repeat(60));
        console.log('\nYou can now login with:');
        console.log('');

        for (const doctor of doctors) {
            console.log(`${doctor.name}:`);
            console.log(`  Email: ${doctor.email}`);
            console.log(`  Password: doctor123`);
            console.log('');
        }

        process.exit(0);
    } catch (error) {
        console.error('\n❌ Error:', error.message);
        process.exit(1);
    }
}

setDoctorPasswords();

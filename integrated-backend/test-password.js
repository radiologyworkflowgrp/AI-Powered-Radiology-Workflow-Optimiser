require('dotenv').config();
const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');
const Admin = require('./models/Admin');

const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017/hospital';

async function testPassword() {
    try {
        await mongoose.connect(MONGODB_URI);
        console.log('✓ Connected to MongoDB\n');

        const admin = await Admin.findOne({ email: 'admin@hospital.com' });

        if (!admin) {
            console.log('❌ Admin not found!');
            process.exit(1);
        }

        console.log('Admin found:');
        console.log('Email:', admin.email);
        console.log('Password hash:', admin.password);
        console.log('');

        // Test password comparison
        const testPassword = 'admin123';
        console.log(`Testing password: "${testPassword}"`);

        const isMatch = await admin.comparePassword(testPassword);
        console.log('Password match:', isMatch);

        // Also test with bcrypt directly
        const directMatch = await bcrypt.compare(testPassword, admin.password);
        console.log('Direct bcrypt match:', directMatch);

        process.exit(0);
    } catch (error) {
        console.error('Error:', error.message);
        process.exit(1);
    }
}

testPassword();

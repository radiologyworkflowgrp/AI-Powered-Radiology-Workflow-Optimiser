const mongoose = require('mongoose');
const User = require('./mongoSchemas/User');
const { Admin, Doctor, Patient, MLReport, RadiologyResult } = require('./models');
require('dotenv').config();

const setupProductionData = async () => {
    try {
        // Connect to MongoDB
        await mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/radiology_hospital');
        console.log('✅ Connected to MongoDB\n');

        // 1. Delete all existing data (order matters due to foreign keys)
        console.log('🗑️  Deleting all test data...');
        await User.deleteMany({});
        await MLReport.destroy({ where: {}, truncate: true, cascade: true });
        await RadiologyResult.destroy({ where: {}, truncate: true, cascade: true });
        await Patient.destroy({ where: {}, truncate: true, cascade: true });
        await Doctor.destroy({ where: {}, truncate: true, cascade: true });
        await Admin.destroy({ where: {}, truncate: true, cascade: true });
        console.log('✅ All test data deleted\n');

        // 2. Create Admin
        console.log('👨‍💼 Creating Admin...');
        const adminProfile = await Admin.create({
            name: 'System Administrator',
            email: 'admin@hospital.com',
            password: 'placeholder'
        });

        const adminUser = new User({
            email: 'admin@hospital.com',
            password: 'admin123',
            role: 'admin',
            referenceId: adminProfile.id.toString()
        });
        await adminUser.save();
        console.log('✅ Admin created: admin@hospital.com / admin123\n');

        // 3. Create 4 Radiologist Doctors
        console.log('👨‍⚕️ Creating 4 Radiologist Doctors...');

        const doctors = [
            { name: 'Dr. Emily Chen', email: 'emily.chen@hospital.com' },
            { name: 'Dr. Michael Rodriguez', email: 'michael.rodriguez@hospital.com' },
            { name: 'Dr. Sarah Patel', email: 'sarah.patel@hospital.com' },
            { name: 'Dr. James Wilson', email: 'james.wilson@hospital.com' }
        ];

        for (const doc of doctors) {
            const doctorProfile = await Doctor.create({
                name: doc.name,
                email: doc.email,
                password: 'placeholder',
                specialty: 'Radiology',
                availability: 'Available'
            });

            const doctorUser = new User({
                email: doc.email,
                password: 'doctor123',
                role: 'doctor',
                referenceId: doctorProfile.id.toString()
            });
            await doctorUser.save();

            console.log(`✅ ${doc.name}: ${doc.email} / doctor123`);
        }

        console.log('\n🎉 Production data setup complete!\n');
        console.log('📝 Login Credentials:');
        console.log('   Admin: admin@hospital.com / admin123');
        console.log('   All Doctors: doctor123 (password)\n');
        console.log('Doctors:');
        doctors.forEach(d => console.log(`   - ${d.email}`));

        process.exit(0);
    } catch (error) {
        console.error('❌ Error:', error);
        process.exit(1);
    }
};

setupProductionData();

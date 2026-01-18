const mongoose = require('mongoose');
const User = require('./mongoSchemas/User');
const { Admin, Doctor } = require('./models');
require('dotenv').config();

const createUsers = async () => {
    try {
        // Connect to MongoDB
        await mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/radiology_hospital');
        console.log('✅ Connected to MongoDB');

        // Create Admin in PostgreSQL if not exists
        let adminProfile = await Admin.findOne({ where: { email: 'admin@hospital.com' } });
        if (!adminProfile) {
            adminProfile = await Admin.create({
                name: 'System Admin',
                email: 'admin@hospital.com',
                password: 'placeholder', // Not used for auth, stored in MongoDB User
            });
            console.log('✅ Admin profile created in PostgreSQL');
        } else {
            console.log('ℹ️  Admin profile already exists in PostgreSQL');
        }

        // Create User for Admin in MongoDB
        let adminUser = await User.findOne({ email: 'admin@hospital.com' });
        if (!adminUser) {
            adminUser = new User({
                email: 'admin@hospital.com',
                password: 'admin123',
                plainPassword: 'admin123',
                role: 'admin',
                referenceId: adminProfile.id.toString()
            });
            await adminUser.save();
            console.log('✅ Admin user created in MongoDB: admin@hospital.com / admin123');
        } else {
            console.log('ℹ️  Admin user already exists in MongoDB');
        }

        // Create Doctor in PostgreSQL if not exists
        let doctorProfile = await Doctor.findOne({ where: { email: 'doctor@hospital.com' } });
        if (!doctorProfile) {
            doctorProfile = await Doctor.create({
                name: 'Dr. Gregory House',
                email: 'doctor@hospital.com',
                password: 'placeholder', // Not used for auth, stored in MongoDB User
                specialty: 'Diagnostic Medicine',
                availability: 'Available'
            });
            console.log('✅ Doctor profile created in PostgreSQL');
        } else {
            console.log('ℹ️  Doctor profile already exists in PostgreSQL');
        }

        // Create User for Doctor in MongoDB
        let doctorUser = await User.findOne({ email: 'doctor@hospital.com' });
        if (!doctorUser) {
            doctorUser = new User({
                email: 'doctor@hospital.com',
                password: 'doctor123',
                plainPassword: 'doctor123',
                role: 'doctor',
                referenceId: doctorProfile.id.toString()
            });
            await doctorUser.save();
            console.log('✅ Doctor user created in MongoDB: doctor@hospital.com / doctor123');
        } else {
            console.log('ℹ️  Doctor user already exists in MongoDB');
        }

        // Create Doctor 2 in PostgreSQL if not exists
        let doctor2Profile = await Doctor.findOne({ where: { email: 'doctor2@hospital.com' } });
        if (!doctor2Profile) {
            doctor2Profile = await Doctor.create({
                name: 'Dr. Sarah Johnson',
                email: 'doctor2@hospital.com',
                password: 'placeholder',
                specialty: 'Radiology',
                availability: 'Available'
            });
            console.log('✅ Doctor 2 profile created in PostgreSQL');
        } else {
            console.log('ℹ️  Doctor 2 profile already exists in PostgreSQL');
        }

        // Create User for Doctor 2 in MongoDB
        let doctor2User = await User.findOne({ email: 'doctor2@hospital.com' });
        if (!doctor2User) {
            doctor2User = new User({
                email: 'doctor2@hospital.com',
                password: 'doctor123',
                plainPassword: 'doctor123',
                role: 'doctor',
                referenceId: doctor2Profile.id.toString()
            });
            await doctor2User.save();
            console.log('✅ Doctor 2 user created in MongoDB: doctor2@hospital.com / doctor123');
        } else {
            console.log('ℹ️  Doctor 2 user already exists in MongoDB');
        }

        console.log('\n🎉 User creation complete!');
        console.log('\n📝 Login Credentials:');
        console.log('   Admin:   admin@hospital.com / admin123');
        console.log('   Doctor:  doctor@hospital.com / doctor123');
        console.log('   Doctor2: doctor2@hospital.com / doctor123');

        process.exit(0);
    } catch (error) {
        console.error('❌ Error creating users:', error);
        process.exit(1);
    }
};

createUsers();

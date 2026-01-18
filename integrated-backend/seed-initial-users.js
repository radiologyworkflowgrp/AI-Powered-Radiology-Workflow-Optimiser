const mongoose = require('mongoose');
const Admin = require('./mongoSchemas/Admin');
const Doctor = require('./mongoSchemas/Doctor');
require('dotenv').config();

const seedData = async () => {
    try {
        await mongoose.connect(process.env.MONGODB_URI);
        console.log('✅ Connected to MongoDB');

        // Clear existing data (optional, but good for clean seed)
        // await Admin.deleteMany({});
        // await Doctor.deleteMany({});

        // check if admin exists
        const adminExists = await Admin.findOne({ email: 'admin@hospital.com' });
        if (!adminExists) {
            const admin = new Admin({
                name: 'System Admin',
                email: 'admin@hospital.com',
                password: 'admin123', // Will be hashed by pre-save hook
                role: 'admin',
                profileCompleted: true
            });
            await admin.save();
            console.log('✅ Admin user created: admin@hospital.com / admin123');
        } else {
            console.log('ℹ️ Admin user already exists');
        }

        // check if doctor exists
        const doctorExists = await Doctor.findOne({ email: 'doctor@hospital.com' });
        if (!doctorExists) {
            const doctor = new Doctor({
                name: 'Dr. Gregory House',
                email: 'doctor@hospital.com',
                password: 'doctor123', // Will be hashed by pre-save hook
                specialty: 'Diagnostic Medicine',
                availability: 'Available',
                profileCompleted: true
            });
            await doctor.save();
            console.log('✅ Doctor user created: doctor@hospital.com / doctor123');
        } else {
            console.log('ℹ️ Doctor user already exists');
        }

        process.exit(0);
    } catch (error) {
        console.error('❌ Seeding failed:', error);
        process.exit(1);
    }
};

seedData();

require('dotenv').config();
const mongoose = require('mongoose');
const { sequelize, Admin, Doctor } = require('./models');
const User = require('./mongoSchemas/User');

const seedUsers = async () => {
    try {
        // Connect to databases
        await mongoose.connect(process.env.MONGODB_URI);
        console.log('✅ Connected to MongoDB');

        await sequelize.authenticate();
        console.log('✅ Connected to PostgreSQL');

        // Sync PostgreSQL models
        await sequelize.sync({ alter: true });

        console.log('\n📝 Seeding Admins...\n');

        // =================
        // SEED 2 ADMINS
        // =================
        const admins = [
            {
                name: 'Dr. Sarah Johnson',
                email: 'admin@hospital.com',
                password: 'admin123',
                department: 'Administration',
                role: 'admin'
            },
            {
                name: 'Dr. Michael Chen',
                email: 'admin2@hospital.com',
                password: 'admin123',
                department: 'Administration',
                role: 'admin'
            }
        ];

        for (const adminData of admins) {
            // Check if admin exists in PostgreSQL
            const existingAdmin = await Admin.findOne({ where: { email: adminData.email } });

            if (!existingAdmin) {
                // Create in PostgreSQL (password will be hashed by Sequelize hooks)
                const newAdmin = await Admin.create({
                    name: adminData.name,
                    email: adminData.email,
                    password: adminData.password,
                    department: adminData.department
                });

                console.log(`✅ Admin created in PostgreSQL: ${newAdmin.name} (ID: ${newAdmin.id})`);

                // Check if user exists in MongoDB
                const existingUser = await User.findOne({ email: adminData.email });

                if (!existingUser) {
                    // Create in MongoDB for authentication
                    const newUser = new User({
                        email: adminData.email,
                        password: adminData.password,
                        role: 'admin',
                        referenceId: newAdmin.id
                    });
                    await newUser.save();

                    console.log(`   🔑 Credentials: ${adminData.email} / ${adminData.password}`);
                } else {
                    console.log(`   ℹ️  User already exists in MongoDB`);
                }
            } else {
                console.log(`ℹ️  Admin already exists: ${adminData.email}`);
            }
        }

        console.log('\n📝 Seeding Doctors...\n');

        // =================
        // SEED 4 DOCTORS
        // =================
        const doctors = [
            {
                name: 'Dr. Gregory House',
                email: 'doctor1@hospital.com',
                password: 'doctor123',
                specialty: 'Diagnostic Medicine',
                availability: 'Available',
                phone: '+1-555-0101'
            },
            {
                name: 'Dr. Emily Rodriguez',
                email: 'doctor2@hospital.com',
                password: 'doctor123',
                specialty: 'Radiology',
                availability: 'Available',
                phone: '+1-555-0102'
            },
            {
                name: 'Dr. James Wilson',
                email: 'doctor3@hospital.com',
                password: 'doctor123',
                specialty: 'Oncology',
                availability: 'Available',
                phone: '+1-555-0103'
            },
            {
                name: 'Dr. Lisa Cuddy',
                email: 'doctor4@hospital.com',
                password: 'doctor123',
                specialty: 'Endocrinology',
                availability: 'Available',
                phone: '+1-555-0104'
            }
        ];

        for (const doctorData of doctors) {
            // Check if doctor exists in PostgreSQL
            const existingDoctor = await Doctor.findOne({ where: { email: doctorData.email } });

            if (!existingDoctor) {
                // Create in PostgreSQL (password will be hashed by Sequelize hooks)
                const newDoctor = await Doctor.create({
                    name: doctorData.name,
                    email: doctorData.email,
                    password: doctorData.password,
                    specialty: doctorData.specialty,
                    availability: doctorData.availability,
                    phone: doctorData.phone
                });

                console.log(`✅ Doctor created in PostgreSQL: ${newDoctor.name} (${newDoctor.specialty})`);

                // Check if user exists in MongoDB
                const existingUser = await User.findOne({ email: doctorData.email });

                if (!existingUser) {
                    // Create in MongoDB for authentication
                    const newUser = new User({
                        email: doctorData.email,
                        password: doctorData.password,
                        role: 'doctor',
                        referenceId: newDoctor.id
                    });
                    await newUser.save();

                    console.log(`   🔑 Credentials: ${doctorData.email} / ${doctorData.password}`);
                } else {
                    console.log(`   ℹ️  User already exists in MongoDB`);
                }
            } else {
                console.log(`ℹ️  Doctor already exists: ${doctorData.email}`);
            }
        }

        console.log('\n\n✅ Seeding completed successfully!\n');
        console.log('===========================================');
        console.log('ADMIN CREDENTIALS:');
        console.log('===========================================');
        console.log('1. admin@hospital.com / admin123');
        console.log('2. admin2@hospital.com / admin123');
        console.log('\n===========================================');
        console.log('DOCTOR CREDENTIALS:');
        console.log('===========================================');
        console.log('1. doctor1@hospital.com / doctor123');
        console.log('2. doctor2@hospital.com / doctor123');
        console.log('3. doctor3@hospital.com / doctor123');
        console.log('4. doctor4@hospital.com / doctor123');
        console.log('===========================================\n');

        await mongoose.connection.close();
        await sequelize.close();
        process.exit(0);

    } catch (error) {
        console.error('❌ Seeding failed:', error);
        process.exit(1);
    }
};

seedUsers();

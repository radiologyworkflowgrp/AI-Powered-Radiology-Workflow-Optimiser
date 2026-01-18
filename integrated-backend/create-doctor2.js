const { Doctor } = require('./models');
const User = require('./mongoSchemas/User');
require('dotenv').config();

const createDoctor = async () => {
    try {
        console.log('👨‍⚕️ Creating new doctor account...');

        const email = 'doctor2@hospital.com';
        const password = 'doctor123';
        const name = 'Dr. Sarah Johnson';

        // 1. Create MongoDB User for authentication
        const existingUser = await User.findOne({ email });
        if (existingUser) {
            console.log('⚠️  User already exists in MongoDB');
        } else {
            const mongoUser = new User({
                email,
                password, // Will be hashed by pre-save hook
                role: 'doctor'
            });
            await mongoUser.save();
            console.log('✅ Created MongoDB user');
        }

        // 2. Create PostgreSQL Doctor profile
        const [doctor, created] = await Doctor.findOrCreate({
            where: { email },
            defaults: {
                name,
                email,
                password, // Required by model
                specialty: 'Radiology',
                phone: '+1-555-0102',
                availability: 'Available'
            }
        });

        if (created) {
            console.log('✅ Created PostgreSQL doctor profile');
        } else {
            console.log('⚠️  Doctor profile already exists');
        }

        console.log('\n✅ Doctor account created successfully!');
        console.log('📧 Email:', email);
        console.log('🔑 Password:', password);
        console.log('👤 Name:', name);

        process.exit(0);
    } catch (error) {
        console.error('❌ Error creating doctor:', error);
        process.exit(1);
    }
};

createDoctor();

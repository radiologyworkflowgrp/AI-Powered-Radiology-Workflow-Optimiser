const mongoose = require('mongoose');
const Doctor = require('./models/Doctor');
const connectDB = require('./db');

// Connect to MongoDB
connectDB();

const doctors = [
    {
        name: "Dr. Preetham",
        email: "preetham@hospital.com",
        password: "Preetham123!",
        specialty: "Radiology",
        availability: "Available"
    },
    {
        name: "Dr. Vamsi",
        email: "vamsi@hospital.com",
        password: "Vamsi123!",
        specialty: "Cardiology",
        availability: "Available"
    },
    {
        name: "Dr. SSC",
        email: "ssc@hospital.com",
        password: "SSC123!",
        specialty: "Neurology",
        availability: "On-call"
    },
    {
        name: "Dr. Dabush",
        email: "dabush@hospital.com",
        password: "Dabush123!",
        specialty: "Orthopedics",
        availability: "Available"
    },
    {
        name: "Dr. Tejesh",
        email: "tejesh@hospital.com",
        password: "Tejesh123!",
        specialty: "General Medicine",
        availability: "Unavailable"
    }
];

const seedDoctors = async () => {
    try {
        // Clear existing doctors
        await Doctor.deleteMany({});
        console.log('Cleared existing doctors');

        // Add new doctors
        for (const doctor of doctors) {
            const newDoctor = new Doctor(doctor);
            await newDoctor.save();
            console.log(`Added doctor: ${doctor.name}`);
        }

        console.log('Database seeded successfully');
        process.exit(0);
    } catch (error) {
        console.error('Error seeding database:', error);
        process.exit(1);
    }
};

// Wait for connection before seeding
mongoose.connection.once('open', () => {
    seedDoctors();
});

const mongoose = require('mongoose');
const Patient = require('./models/Patient');
const Doctor = require('./models/Doctor');
const connectDB = require('./db');

// Connect to MongoDB
connectDB();

const testAssignment = async () => {
    try {
        // 1. Ensure there is an available doctor
        let doctor = await Doctor.findOne({ availability: 'Available' });
        if (!doctor) {
            console.log('Creating a dummy available doctor...');
            doctor = new Doctor({
                name: "Dr. Test Assignment",
                email: "testassign@hospital.com",
                password: "password123",
                specialty: "General",
                availability: "Available"
            });
            await doctor.save();
        }
        console.log(`Using available doctor: ${doctor.name} (${doctor._id})`);

        // 2. Create a new patient via direct DB call (simulating API logic for now, or use axios to hit API)
        // Since I modified server.js, I should ideally hit the API. But for quick test of logic, I can replicate logic here or just run a script that hits the API.
        // Let's use axios to hit the running server to test the full flow.

        const axios = require('axios');
        const patientData = {
            name: "Test Patient Assignment",
            age: 30,
            gender: "Male",
            email: `testpatient${Date.now()}@example.com`,
            symptoms: "Headache"
        };

        console.log('Sending POST request to /api/patients...');
        try {
            const response = await axios.post('http://localhost:3002/api/patients', patientData);
            const createdPatient = response.data.patient;

            console.log('Patient created:', createdPatient._id);

            if (createdPatient.assignedDoctor && createdPatient.assignedDoctor.id === doctor._id.toString()) {
                console.log('SUCCESS: Patient assigned to the correct doctor.');
                console.log('Assigned Doctor:', createdPatient.assignedDoctor);
            } else {
                console.log('FAILURE: Patient NOT assigned to the correct doctor.');
                console.log('Expected:', doctor._id);
                console.log('Actual:', createdPatient.assignedDoctor);
            }

        } catch (apiError) {
            console.error('API Error:', apiError.message);
            if (apiError.response) {
                console.error('Response data:', apiError.response.data);
            }
        }

    } catch (error) {
        console.error('Test Error:', error);
    } finally {
        mongoose.connection.close();
    }
};

// Wait for connection
mongoose.connection.once('open', () => {
    testAssignment();
});

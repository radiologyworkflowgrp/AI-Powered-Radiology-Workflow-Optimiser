require('dotenv').config();
const mongoose = require('mongoose');
const Patient = require('./models/Patient');
const Doctor = require('./models/Doctor');
const RadiologyResult = require('./models/RadiologyResult');

const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017/radiology_hospital';

async function testDoctorLogin() {
    try {
        await mongoose.connect(MONGODB_URI);
        console.log('✓ Connected to MongoDB\n');

        // Test with Dr. Preetham
        const doctorEmail = 'preetham@hospital.com';
        const doctor = await Doctor.findOne({ email: doctorEmail });

        if (!doctor) {
            console.log(`❌ Doctor ${doctorEmail} not found`);
            process.exit(1);
        }

        console.log(`Testing for: ${doctor.name} (${doctor.email})`);
        console.log(`Doctor ID: ${doctor._id}\n`);

        // Check assigned patients
        const assignedPatients = await Patient.find({ 'assignedDoctor.id': doctor._id });
        console.log(`Assigned patients: ${assignedPatients.length}`);

        if (assignedPatients.length > 0) {
            assignedPatients.forEach(p => {
                console.log(`  - ${p.name} (ID: ${p._id})`);
            });
        } else {
            console.log('  (No patients assigned)');
        }

        // Get patient IDs
        const allowedPatientIds = assignedPatients.map(p => p._id.toString());
        console.log(`\nAllowed patient IDs: [${allowedPatientIds.join(', ')}]`);

        // Check radiology results
        console.log('\n' + '='.repeat(60));
        console.log('RADIOLOGY RESULTS:');
        console.log('='.repeat(60));

        const allResults = await RadiologyResult.find();
        console.log(`\nTotal results in database: ${allResults.length}`);

        if (allResults.length > 0) {
            allResults.forEach(r => {
                const isAllowed = allowedPatientIds.includes(r.patientId);
                console.log(`  - Patient ID: ${r.patientId} ${isAllowed ? '✓ ALLOWED' : '✗ NOT ALLOWED'}`);
            });
        }

        // Simulate filtering
        let filteredResults;
        if (allowedPatientIds.length === 0) {
            console.log('\n🔍 Doctor has 0 assigned patients → Should see 0 results');
            filteredResults = [];
        } else {
            console.log('\n🔍 Filtering by allowed patient IDs...');
            filteredResults = await RadiologyResult.find({
                patientId: { $in: allowedPatientIds }
            });
        }

        console.log(`\nFiltered results: ${filteredResults.length}`);
        if (filteredResults.length > 0) {
            filteredResults.forEach(r => {
                console.log(`  - ${r.mlModel} for patient ${r.patientId}`);
            });
        }

        console.log('\n' + '='.repeat(60));
        console.log('EXPECTED BEHAVIOR:');
        console.log('='.repeat(60));
        console.log(`Doctor ${doctor.name} should see: ${filteredResults.length} results`);

        process.exit(0);
    } catch (error) {
        console.error('\n❌ Error:', error.message);
        process.exit(1);
    }
}

testDoctorLogin();

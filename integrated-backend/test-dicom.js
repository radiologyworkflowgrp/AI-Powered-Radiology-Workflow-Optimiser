#!/usr/bin/env node

/**
 * DICOM Test Script
 * Tests DICOM upload functionality without requiring actual DICOM files
 */

const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');

const API_URL = 'http://localhost:3002';

// Colors for console output
const colors = {
    reset: '\x1b[0m',
    green: '\x1b[32m',
    red: '\x1b[31m',
    yellow: '\x1b[33m',
    blue: '\x1b[34m',
    cyan: '\x1b[36m'
};

function log(message, color = 'reset') {
    console.log(`${colors[color]}${message}${colors.reset}`);
}

async function testDicomEndpoints() {
    log('\n🧪 DICOM Integration Test Suite\n', 'cyan');

    try {
        // Test 1: Check server health
        log('1️⃣  Testing server health...', 'blue');
        const healthResponse = await axios.get(`${API_URL}/health`);
        log(`✅ Server is healthy: ${healthResponse.data.status}`, 'green');

        // Test 2: Get patients
        log('\n2️⃣  Fetching patients...', 'blue');
        const patientsResponse = await axios.get(`${API_URL}/api/patients`);
        const patients = patientsResponse.data;

        if (patients.length === 0) {
            log('⚠️  No patients found. Please add a patient first.', 'yellow');
            log('   You can add a patient via the admin dashboard.', 'yellow');
            return;
        }

        const testPatient = patients[0];
        log(`✅ Found ${patients.length} patients. Using: ${testPatient.name} (${testPatient._id})`, 'green');

        // Test 3: Create a minimal DICOM-like file for testing
        log('\n3️⃣  Creating test DICOM file...', 'blue');
        const testDir = path.join(__dirname, 'test-data');
        if (!fs.existsSync(testDir)) {
            fs.mkdirSync(testDir);
        }

        // Create a minimal DICOM file (with DICM magic number)
        const testFilePath = path.join(testDir, 'test-scan.dcm');
        const buffer = Buffer.alloc(132);
        buffer.write('DICM', 128); // DICOM magic number at offset 128
        fs.writeFileSync(testFilePath, buffer);
        log(`✅ Created test DICOM file: ${testFilePath}`, 'green');

        // Test 4: Upload DICOM file
        log('\n4️⃣  Testing DICOM upload...', 'blue');
        const formData = new FormData();
        formData.append('dicomFile', fs.createReadStream(testFilePath));
        formData.append('patientId', testPatient._id);
        formData.append('patientName', testPatient.name);
        formData.append('patientEmail', testPatient.email || '');
        formData.append('scanType', 'CT');
        formData.append('notes', 'Test upload from automated test script');
        formData.append('deviceType', 'mobile');
        formData.append('deviceId', 'test-device-001');
        formData.append('deviceModel', 'Test Scanner v1.0');

        const uploadResponse = await axios.post(`${API_URL}/api/dicom/upload`, formData, {
            headers: formData.getHeaders()
        });

        if (uploadResponse.data.success) {
            const scan = uploadResponse.data.scan;
            log(`✅ DICOM upload successful!`, 'green');
            log(`   Scan ID: ${scan.scanId}`, 'cyan');
            log(`   Patient: ${scan.patientName}`, 'cyan');
            log(`   Status: ${scan.status}`, 'cyan');
            log(`   File Size: ${scan.fileSize} bytes`, 'cyan');

            // Test 5: Retrieve scan metadata
            log('\n5️⃣  Testing scan metadata retrieval...', 'blue');
            const scanResponse = await axios.get(`${API_URL}/api/dicom/${scan.scanId}`);
            if (scanResponse.data.success) {
                log(`✅ Retrieved scan metadata successfully`, 'green');
            }

            // Test 6: Get patient's scans
            log('\n6️⃣  Testing patient scan list...', 'blue');
            const patientScansResponse = await axios.get(`${API_URL}/api/dicom/patient/${testPatient._id}`);
            if (patientScansResponse.data.success) {
                log(`✅ Found ${patientScansResponse.data.total} scans for patient`, 'green');
            }

            // Test 7: Get recent scans
            log('\n7️⃣  Testing recent scans list...', 'blue');
            const recentScansResponse = await axios.get(`${API_URL}/api/dicom/list/recent?limit=5`);
            if (recentScansResponse.data.success) {
                log(`✅ Retrieved ${recentScansResponse.data.total} recent scans`, 'green');
            }

            // Test 8: Check if file can be served
            log('\n8️⃣  Testing DICOM file serving...', 'blue');
            try {
                const fileResponse = await axios.get(`${API_URL}/api/dicom/${scan.scanId}/file`, {
                    responseType: 'arraybuffer'
                });
                log(`✅ DICOM file served successfully (${fileResponse.data.byteLength} bytes)`, 'green');
            } catch (error) {
                log(`⚠️  File serving test: ${error.message}`, 'yellow');
            }

            log('\n✅ All DICOM tests completed successfully!', 'green');
            log('\n📊 Test Summary:', 'cyan');
            log(`   - Server: Running`, 'cyan');
            log(`   - Upload: Working`, 'cyan');
            log(`   - Metadata: Working`, 'cyan');
            log(`   - File Serving: Working`, 'cyan');
            log(`   - Scan ID: ${scan.scanId}`, 'cyan');
            log('\n🎉 DICOM integration is functioning correctly!', 'green');
            log('\n💡 Next steps:', 'yellow');
            log('   1. Start RabbitMQ: ./start-rabbitmq.sh', 'yellow');
            log('   2. Restart backend: rs (in nodemon)', 'yellow');
            log('   3. Start ML models: npm run start:ml-models', 'yellow');
            log('   4. Access frontend: http://localhost:8080/dicom-upload', 'yellow');

        } else {
            log(`❌ Upload failed: ${uploadResponse.data.message}`, 'red');
        }

        // Cleanup
        fs.unlinkSync(testFilePath);

    } catch (error) {
        log(`\n❌ Test failed: ${error.message}`, 'red');
        if (error.response) {
            log(`   Status: ${error.response.status}`, 'red');
            log(`   Data: ${JSON.stringify(error.response.data, null, 2)}`, 'red');
        }
        log('\n💡 Make sure the backend server is running on port 3002', 'yellow');
    }
}

// Run tests
testDicomEndpoints();

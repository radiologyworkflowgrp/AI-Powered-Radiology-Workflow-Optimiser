// Test the actual API endpoint
const axios = require('axios');

async function testAPI() {
    try {
        console.log('Testing /api/radiology-results endpoint...\n');

        // Test 1: Without authentication (should show all)
        console.log('Test 1: No authentication');
        console.log('Expected: All 5 results (no filtering)');
        try {
            const response1 = await axios.get('http://localhost:3002/api/radiology-results');
            console.log(`✓ Got ${response1.data.total} results`);
            console.log(`  filteredByDoctor: ${response1.data.filteredByDoctor || 'false'}\n`);
        } catch (err) {
            console.log(`❌ Error: ${err.message}\n`);
        }

        // Test 2: Login as doctor and test
        console.log('Test 2: Login as Dr. Preetham');
        console.log('Expected: 0 results (doctor has no assigned patients)');

        try {
            // Login
            const loginResponse = await axios.post('http://localhost:3002/api/login', {
                email: 'preetham@hospital.com',
                password: 'doctor123',
                role: 'doctor'
            });

            const token = loginResponse.data.token;
            console.log(`✓ Logged in successfully`);
            console.log(`  Token: ${token.substring(0, 20)}...`);

            // Get results with auth
            const response2 = await axios.get('http://localhost:3002/api/radiology-results', {
                headers: {
                    'Cookie': `authToken=${token}`
                }
            });

            console.log(`✓ Got ${response2.data.total} results`);
            console.log(`  filteredByDoctor: ${response2.data.filteredByDoctor || 'false'}`);
            console.log(`  message: ${response2.data.message}\n`);

            if (response2.data.total === 0) {
                console.log('✅ SUCCESS: Doctor sees 0 results (correct!)');
            } else {
                console.log('❌ FAIL: Doctor should see 0 results but sees', response2.data.total);
            }

        } catch (err) {
            console.log(`❌ Error: ${err.response?.data?.message || err.message}\n`);
        }

    } catch (error) {
        console.error('Error:', error.message);
    }
}

testAPI();

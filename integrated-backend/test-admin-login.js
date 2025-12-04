const axios = require('axios');

async function testAdminLogin() {
    try {
        const response = await axios.post('http://localhost:3002/api/auth/login', {
            email: 'admin@hospital.com',
            password: 'admin123',
            role: 'admin'
        });

        console.log('✓ Login successful!');
        console.log('\nResponse:');
        console.log(JSON.stringify(response.data, null, 2));
    } catch (error) {
        console.error('❌ Login failed!');
        if (error.response) {
            console.error('Status:', error.response.status);
            console.error('Message:', error.response.data.message);
        } else {
            console.error('Error:', error.message);
        }
    }
}

testAdminLogin();

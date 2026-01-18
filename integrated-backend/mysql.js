const mysql = require('mysql2/promise');

const pool = mysql.createPool({
    host: process.env.MYSQL_HOST || 'localhost',
    user: process.env.MYSQL_USER || 'appuser',
    password: process.env.MYSQL_PASSWORD || 'AppUser123!',
    database: process.env.MYSQL_DATABASE || 'radiology_hospital',
    waitForConnections: true,
    connectionLimit: 10,
    queueLimit: 0
});

async function initDatabase() {
    try {
        const connection = await pool.getConnection();

        // Create ml_reports table if it doesn't exist
        await connection.query(`
      CREATE TABLE IF NOT EXISTS ml_reports (
        id INT AUTO_INCREMENT PRIMARY KEY,
        patient_id VARCHAR(255) NOT NULL,
        report_type VARCHAR(100) NOT NULL,
        report_status ENUM('pending', 'processing', 'completed', 'failed') DEFAULT 'pending',
        report_data JSON,
        pdf_path VARCHAR(500),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_patient_id (patient_id),
        INDEX idx_status (report_status),
        INDEX idx_created_at (created_at)
      )
    `);

        connection.release();
        console.log('✅ MySQL database initialized successfully');
        console.log('📊 Table: ml_reports created/verified');
    } catch (error) {
        console.error('❌ MySQL initialization error:', error.message);
        throw error;
    }
}

async function testConnection() {
    try {
        const connection = await pool.getConnection();
        const [rows] = await connection.query('SELECT 1 as test');
        console.log('✅ MySQL connection test successful');
        connection.release();
        return true;
    } catch (error) {
        console.error('❌ MySQL connection test failed:', error.message);
        console.error('💡 Make sure MySQL is running and credentials are correct');
        console.error('💡 Run: ./setup-mysql.sh to setup the database');
        throw error;
    }
}

module.exports = { pool, initDatabase, testConnection };

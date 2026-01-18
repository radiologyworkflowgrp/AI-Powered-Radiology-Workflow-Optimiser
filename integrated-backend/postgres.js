const { Sequelize } = require('sequelize');
require('dotenv').config();

// PostgreSQL connection configuration
const sequelize = new Sequelize(
    process.env.POSTGRES_DATABASE || 'radiology_hospital',
    process.env.POSTGRES_USER || 'radiology_user',
    process.env.POSTGRES_PASSWORD || 'SecurePassword123!',
    {
        host: process.env.POSTGRES_HOST || 'localhost',
        port: process.env.POSTGRES_PORT || 5432,
        dialect: 'postgres',
        logging: process.env.NODE_ENV === 'development' ? console.log : false,
        pool: {
            max: parseInt(process.env.POSTGRES_MAX_CONNECTIONS) || 20,
            min: 0,
            acquire: 30000,
            idle: 10000
        },
        define: {
            timestamps: true,
            underscored: false,
            createdAt: 'created_at',
            updatedAt: 'updated_at'
        }
    }
);

/**
 * Initialize database connection and sync models
 */
async function initDatabase() {
    try {
        await sequelize.authenticate();
        console.log('✅ PostgreSQL connection established successfully');
        console.log(`📊 Database: ${sequelize.config.database}`);

        // Sync models (create tables if they don't exist)
        // Use { alter: true } in development to update existing tables
        // Use { force: false } to prevent dropping tables
        await sequelize.sync({ alter: process.env.NODE_ENV === 'development' });
        console.log('✅ Database models synchronized');

        return true;
    } catch (error) {
        console.error('❌ PostgreSQL initialization error:', error.message);
        throw error;
    }
}

/**
 * Test database connection
 */
async function testConnection() {
    try {
        await sequelize.authenticate();
        console.log('✅ PostgreSQL connection test successful');
        return true;
    } catch (error) {
        console.error('❌ PostgreSQL connection test failed:', error.message);
        console.error('💡 Make sure PostgreSQL is running and credentials are correct');
        console.error('💡 Run: ./migrations/setup-postgres.sh to setup the database');
        throw error;
    }
}

/**
 * Close database connection
 */
async function closeConnection() {
    try {
        await sequelize.close();
        console.log('✅ PostgreSQL connection closed');
    } catch (error) {
        console.error('❌ Error closing PostgreSQL connection:', error.message);
        throw error;
    }
}

module.exports = {
    sequelize,
    initDatabase,
    testConnection,
    closeConnection
};

# AI-Powered Radiology Workflow Optimizer - Setup Guide

## 📋 Project Overview

This is a full-stack radiology workflow optimization system with:
- **Frontend**: React + Vite + TailwindCSS + shadcn/ui
- **Backend**: Node.js + Express
- **Databases**: MongoDB, MySQL, Redis
- **Message Queue**: RabbitMQ
- **ML Services**: Python (PyTorch, FastAPI)

---

## 🔧 Prerequisites

### Required Software

1. **Node.js & npm** (v18 or higher)
   ```bash
   # Check version
   node --version
   npm --version
   
   # Install if needed (using nvm)
   curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
   nvm install 18
   nvm use 18
   ```

2. **MongoDB** (v6.0 or higher)
   ```bash
   # Ubuntu/Pop!_OS
   sudo apt update
   sudo apt install -y mongodb-org
   sudo systemctl start mongod
   sudo systemctl enable mongod
   
   # Verify installation
   mongosh --version
   ```

3. **MySQL** (v8.0 or higher)
   ```bash
   # Ubuntu/Pop!_OS
   sudo apt update
   sudo apt install mysql-server
   sudo systemctl start mysql
   sudo systemctl enable mysql
   
   # Verify installation
   mysql --version
   ```

4. **Redis** (v6.0 or higher)
   ```bash
   # Ubuntu/Pop!_OS
   sudo apt update
   sudo apt install redis-server
   sudo systemctl start redis-server
   sudo systemctl enable redis-server
   
   # Verify installation
   redis-cli ping  # Should return "PONG"
   ```

5. **RabbitMQ** (v3.9 or higher)
   ```bash
   # Ubuntu/Pop!_OS
   sudo apt update
   sudo apt install rabbitmq-server
   sudo systemctl start rabbitmq-server
   sudo systemctl enable rabbitmq-server
   
   # Verify installation
   sudo rabbitmqctl status
   ```

6. **Python** (v3.8 or higher)
   ```bash
   # Check version
   python3 --version
   pip3 --version
   
   # Install if needed
   sudo apt update
   sudo apt install python3 python3-pip python3-venv
   ```

---

## 📦 Installation Steps

### 1. Clone the Repository (if not already done)
```bash
cd /home/lnoob777/psproject/AI-Powered-Radiology-Workflow-Optimiser
```

### 2. Setup Backend

```bash
cd integrated-backend

# Install Node.js dependencies
npm install

# Setup MySQL database
chmod +x setup-mysql.sh
./setup-mysql.sh

# Setup RabbitMQ
chmod +x start-rabbitmq.sh
./start-rabbitmq.sh
```

### 3. Setup Python ML Services

```bash
cd integrated-backend/prioritization-ml

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Install additional dependencies for DuoFormer (if needed)
pip install -r duoformer_requirements.txt
```

### 4. Setup Frontend

```bash
cd ../../RadiologyFrontend

# Install Node.js dependencies
npm install
```

### 5. Create Missing Configuration Files

#### Create `db.js` for MongoDB connection (Backend)
```bash
cd ../integrated-backend
```

Create file: `integrated-backend/db.js`
```javascript
const mongoose = require('mongoose');

const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017/radiology_hospital';

const connectDB = async () => {
  try {
    await mongoose.connect(MONGODB_URI, {
      useNewUrlParser: true,
      useUnifiedTopology: true,
    });
    console.log('✅ MongoDB connected successfully');
  } catch (error) {
    console.error('❌ MongoDB connection error:', error.message);
    process.exit(1);
  }
};

module.exports = connectDB;
```

#### Create `mysql.js` for MySQL connection (Backend)
Create file: `integrated-backend/mysql.js`
```javascript
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
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_patient_id (patient_id),
        INDEX idx_status (report_status)
      )
    `);
    
    connection.release();
    console.log('✅ MySQL database initialized successfully');
  } catch (error) {
    console.error('❌ MySQL initialization error:', error.message);
    throw error;
  }
}

async function testConnection() {
  try {
    const connection = await pool.getConnection();
    console.log('✅ MySQL connection test successful');
    connection.release();
  } catch (error) {
    console.error('❌ MySQL connection test failed:', error.message);
    throw error;
  }
}

module.exports = { pool, initDatabase, testConnection };
```

#### Create `.env` file (Backend)
Create file: `integrated-backend/.env`
```env
# Server Configuration
PORT=3002
NODE_ENV=development

# MongoDB Configuration
MONGODB_URI=mongodb://localhost:27017/radiology_hospital

# MySQL Configuration
MYSQL_HOST=localhost
MYSQL_USER=appuser
MYSQL_PASSWORD=AppUser123!
MYSQL_DATABASE=radiology_hospital

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379

# RabbitMQ Configuration
RABBITMQ_URL=amqp://localhost:5672

# JWT Secret
JWT_SECRET=your-secret-key-change-in-production

# CORS Origins
CORS_ORIGIN=http://localhost:8080,http://localhost:8081
```

#### Create Models Directory and Files (Backend)
```bash
mkdir -p integrated-backend/models
```

You'll need to create model files. Based on the server.js imports, you need:
- `Patient.js`
- `Doctor.js`
- `Prescription.js`
- `Note.js`
- `Admin.js`
- `Job.js`
- `RadiologyResult.js`
- `MLReport.js`
- `ActivityLog.js`

(These should already exist or need to be created based on your schema)

---

## 🚀 Starting the Application

### Terminal 1: Start MongoDB (if not running as service)
```bash
sudo systemctl status mongod
# If not running:
sudo systemctl start mongod
```

### Terminal 2: Start MySQL (if not running as service)
```bash
sudo systemctl status mysql
# If not running:
sudo systemctl start mysql
```

### Terminal 3: Start Redis (if not running as service)
```bash
sudo systemctl status redis-server
# If not running:
sudo systemctl start redis-server
```

### Terminal 4: Start RabbitMQ (if not running as service)
```bash
sudo systemctl status rabbitmq-server
# If not running:
sudo systemctl start rabbitmq-server
```

### Terminal 5: Start Backend Server
```bash
cd /home/lnoob777/psproject/AI-Powered-Radiology-Workflow-Optimiser/integrated-backend

# Start in development mode (with auto-reload)
npm run dev

# OR start in production mode
npm start
```

**Backend will run on:** `http://localhost:3002`

### Terminal 6: Start Python ML Services (Optional)
```bash
cd /home/lnoob777/psproject/AI-Powered-Radiology-Workflow-Optimiser/integrated-backend/prioritization-ml

# Activate virtual environment
source venv/bin/activate

# Start ML model runner
cd ..
npm run start:ml-models

# OR in development mode
npm run dev:ml-models
```

### Terminal 7: Start Frontend
```bash
cd /home/lnoob777/psproject/AI-Powered-Radiology-Workflow-Optimiser/RadiologyFrontend

# Start development server
npm run dev
```

**Frontend will run on:** `http://localhost:8080` or `http://localhost:8081`

---

## 🔍 Verification

### Check All Services
```bash
# MongoDB
mongosh --eval "db.adminCommand('ping')"

# MySQL
mysql -u appuser -pAppUser123! -e "SELECT 1;"

# Redis
redis-cli ping

# RabbitMQ
sudo rabbitmqctl status

# Backend API
curl http://localhost:3002/api/health

# Frontend
curl http://localhost:8080
```

### Run Backend Tests
```bash
cd integrated-backend

# Test ML integration
chmod +x verify-ml-integration.sh
./verify-ml-integration.sh

# Test DICOM functionality
node test-dicom.js

# Test full pipeline
node test-full-pipeline.js
```

---

## 📊 Access Points

- **Frontend**: http://localhost:8080 or http://localhost:8081
- **Backend API**: http://localhost:3002
- **RabbitMQ Management**: http://localhost:15672 (guest/guest)
- **MongoDB**: mongodb://localhost:27017
- **MySQL**: localhost:3306
- **Redis**: localhost:6379

---

## 🛠️ Useful Commands

### Backend
```bash
# Development mode with auto-reload
npm run dev

# Production mode
npm start

# Start ML models
npm run start:ml-models

# Seed doctors data
node seed-doctors.js

# Set doctor passwords
node set-doctor-passwords.js
```

### Frontend
```bash
# Development mode
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Lint code
npm run lint
```

### Database Management
```bash
# MongoDB shell
mongosh

# MySQL shell
mysql -u appuser -pAppUser123!

# Redis CLI
redis-cli

# RabbitMQ management
sudo rabbitmq-plugins enable rabbitmq_management
```

---

## 🐛 Troubleshooting

### MongoDB Connection Issues
```bash
# Check if MongoDB is running
sudo systemctl status mongod

# View MongoDB logs
sudo journalctl -u mongod -n 50

# Restart MongoDB
sudo systemctl restart mongod
```

### MySQL Connection Issues
```bash
# Check if MySQL is running
sudo systemctl status mysql

# Reset MySQL password
sudo mysql
ALTER USER 'appuser'@'localhost' IDENTIFIED BY 'AppUser123!';
FLUSH PRIVILEGES;
```

### RabbitMQ Issues
```bash
# Check status
sudo rabbitmqctl status

# Restart RabbitMQ
sudo systemctl restart rabbitmq-server

# View logs
sudo journalctl -u rabbitmq-server -n 50
```

### Port Already in Use
```bash
# Find process using port 3002 (backend)
lsof -i :3002
kill -9 <PID>

# Find process using port 8080 (frontend)
lsof -i :8080
kill -9 <PID>
```

---

## 📝 Notes

1. **Missing Files**: The project references `db.js` and `mysql.js` which need to be created (see above)
2. **Models**: Ensure all Mongoose models are properly defined in the `models/` directory
3. **Environment Variables**: Update `.env` with your actual credentials for production
4. **Python Environment**: Always activate the virtual environment before running ML services
5. **File Permissions**: Make sure shell scripts have execute permissions (`chmod +x`)

---

## 🔐 Default Credentials

### RabbitMQ Management Console
- Username: `guest`
- Password: `guest`

### MySQL Database
- User: `appuser`
- Password: `AppUser123!`
- Database: `radiology_hospital`

### MongoDB
- No authentication by default (local development)
- Database: `radiology_hospital`

---

## 📚 Additional Resources

- [MongoDB Documentation](https://docs.mongodb.com/)
- [MySQL Documentation](https://dev.mysql.com/doc/)
- [RabbitMQ Documentation](https://www.rabbitmq.com/documentation.html)
- [Redis Documentation](https://redis.io/documentation)
- [Express.js Documentation](https://expressjs.com/)
- [React Documentation](https://react.dev/)
- [Vite Documentation](https://vitejs.dev/)

---

## ✅ Quick Start Checklist

- [ ] Install all prerequisites (Node.js, MongoDB, MySQL, Redis, RabbitMQ, Python)
- [ ] Run `npm install` in `integrated-backend/`
- [ ] Run `npm install` in `RadiologyFrontend/`
- [ ] Create Python virtual environment and install dependencies
- [ ] Run `setup-mysql.sh` to setup MySQL database
- [ ] Run `start-rabbitmq.sh` to start RabbitMQ
- [ ] Create missing `db.js` and `mysql.js` files
- [ ] Create `.env` file with configuration
- [ ] Start all services (MongoDB, MySQL, Redis, RabbitMQ)
- [ ] Start backend server (`npm run dev`)
- [ ] Start frontend (`npm run dev`)
- [ ] Access application at http://localhost:8080

---

**Need Help?** Check the troubleshooting section or review the logs for each service.

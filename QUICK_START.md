# 🚀 Quick Start Commands - Radiology Workflow Optimizer

## ⚡ One-Command Installation

```bash
cd /home/lnoob777/psproject/AI-Powered-Radiology-Workflow-Optimiser
chmod +x install.sh
./install.sh
```

This will:
- Check all prerequisites
- Install Node.js (if missing)
- Install all npm dependencies (frontend + backend)
- Setup Python virtual environment
- Install Python ML dependencies

---

## 🎯 Start the Application (3 Simple Steps)

### Step 1: Start Backend
```bash
cd /home/lnoob777/psproject/AI-Powered-Radiology-Workflow-Optimiser/integrated-backend
npm run dev
```
**Backend runs on:** http://localhost:3002

### Step 2: Start Frontend (New Terminal)
```bash
cd /home/lnoob777/psproject/AI-Powered-Radiology-Workflow-Optimiser/RadiologyFrontend
npm run dev
```
**Frontend runs on:** http://localhost:8080

### Step 3: Open Browser
```
http://localhost:8080
```

---

## 🗄️ Database Setup (One-Time)

### Setup MySQL
```bash
cd /home/lnoob777/psproject/AI-Powered-Radiology-Workflow-Optimiser/integrated-backend
chmod +x setup-mysql.sh
./setup-mysql.sh
```

### Setup RabbitMQ
```bash
cd /home/lnoob777/psproject/AI-Powered-Radiology-Workflow-Optimiser/integrated-backend
chmod +x start-rabbitmq.sh
./start-rabbitmq.sh
```

### Start MongoDB (if not running)
```bash
sudo systemctl start mongod
sudo systemctl enable mongod  # Auto-start on boot
```

### Start Redis (if not running)
```bash
sudo systemctl start redis-server
sudo systemctl enable redis-server  # Auto-start on boot
```

---

## 🐍 Python ML Services (Optional)

```bash
cd /home/lnoob777/psproject/AI-Powered-Radiology-Workflow-Optimiser/integrated-backend

# Activate Python environment
source prioritization-ml/venv/bin/activate

# Start ML models
npm run start:ml-models
```

---

## 🔍 Check Service Status

```bash
# MongoDB
sudo systemctl status mongod

# MySQL
sudo systemctl status mysql

# Redis
sudo systemctl status redis-server

# RabbitMQ
sudo systemctl status rabbitmq-server
```

---

## 🛑 Stop Services

```bash
# Stop backend: Press Ctrl+C in terminal

# Stop frontend: Press Ctrl+C in terminal

# Stop databases (optional)
sudo systemctl stop mongod
sudo systemctl stop mysql
sudo systemctl stop redis-server
sudo systemctl stop rabbitmq-server
```

---

## 🔧 Troubleshooting

### Port Already in Use
```bash
# Kill process on port 3002 (backend)
lsof -i :3002
kill -9 <PID>

# Kill process on port 8080 (frontend)
lsof -i :8080
kill -9 <PID>
```

### MongoDB Not Running
```bash
sudo systemctl start mongod
sudo systemctl status mongod
```

### MySQL Connection Failed
```bash
# Check status
sudo systemctl status mysql

# Restart MySQL
sudo systemctl restart mysql

# Re-run setup
cd integrated-backend
./setup-mysql.sh
```

### Node.js Not Found
```bash
# Install Node.js using nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
source ~/.bashrc
nvm install 18
nvm use 18
```

---

## 📊 Access Points

| Service | URL | Credentials |
|---------|-----|-------------|
| Frontend | http://localhost:8080 | - |
| Backend API | http://localhost:3002 | - |
| RabbitMQ Management | http://localhost:15672 | guest / guest |
| MongoDB | mongodb://localhost:27017 | - |
| MySQL | localhost:3306 | appuser / AppUser123! |

---

## 📝 Useful Development Commands

### Backend
```bash
# Development mode (auto-reload)
npm run dev

# Production mode
npm start

# Run tests
node test-full-pipeline.js
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

### Database
```bash
# MongoDB shell
mongosh

# MySQL shell
mysql -u appuser -pAppUser123!

# Redis CLI
redis-cli

# Check RabbitMQ queues
sudo rabbitmqctl list_queues
```

---

## 📚 Documentation Files

- **SETUP_GUIDE.md** - Complete installation guide
- **PROJECT_SUMMARY.md** - Project overview and architecture
- **README.md** - Project description
- **QUICK_START.md** - This file

---

## ✅ Quick Checklist

- [ ] Node.js installed (`node --version`)
- [ ] npm installed (`npm --version`)
- [ ] MongoDB running (`sudo systemctl status mongod`)
- [ ] MySQL running (`sudo systemctl status mysql`)
- [ ] Redis running (`sudo systemctl status redis-server`)
- [ ] RabbitMQ running (`sudo systemctl status rabbitmq-server`)
- [ ] Backend dependencies installed (`cd integrated-backend && npm install`)
- [ ] Frontend dependencies installed (`cd RadiologyFrontend && npm install`)
- [ ] MySQL database setup (`./setup-mysql.sh`)
- [ ] Backend running (`npm run dev`)
- [ ] Frontend running (`npm run dev`)
- [ ] Application accessible (http://localhost:8080)

---

**Need more help?** See `SETUP_GUIDE.md` for detailed instructions.

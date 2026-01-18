# Quick Start Guide - AI-Powered Radiology Workflow Optimizer

## 🚀 One-Click Startup

### Option 1: Batch File (Recommended for Windows)
Double-click: `START_ALL_SERVICES.bat`

This will automatically start:
- ✅ Backend Server (Port 3002)
- ✅ ML Workers (Auto-processes DICOM uploads)
- ✅ Frontend (Port 8080)

### Option 2: PowerShell Script
Right-click `START_ALL_SERVICES.ps1` → Run with PowerShell

---

## 📋 Login Credentials

### Admin
- Email: `admin@hospital.com`
- Password: `admin123`

### Doctor
- Email: `doctor@hospital.com`
- Password: `doctor123`

### Patient
- Email: `patient@hospital.com`
- Password: `patient123`

---

## 🔧 Manual Startup (If needed)

### Terminal 1: Backend
```powershell
cd integrated-backend
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
npm run dev
```

### Terminal 2: ML Workers
```powershell
cd integrated-backend
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
.\prioritization-ml\venv\Scripts\Activate.ps1
npm run start:ml-models
```

### Terminal 3: Frontend
```powershell
cd RadiologyFrontend
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
npm run dev
```

---

## 🏥 How It Works

1. **Doctor uploads DICOM** → Stored in database
2. **ML Workers automatically process** → AI analyzes the scan
3. **Results appear** → Radiology Results page shows AI-generated reports

**Important:** ML workers MUST be running for automatic DICOM processing!

---

## 🌐 Access URLs

- **Frontend**: http://localhost:8080
- **Backend API**: http://localhost:3002
- **RabbitMQ Management**: http://localhost:15672 (guest/guest)

---

## ✨ Features

- ✅ Patient Management
- ✅ Doctor Assignment (Auto-assigned by priority)
- ✅ DICOM Upload & Viewing
- ✅ AI-Powered Radiology Analysis
- ✅ ML Report Generation
- ✅ PDF Export
- ✅ Real-time Processing Queue

---

## 🆘 Troubleshooting

**DICOM uploads not showing results?**
→ Make sure ML workers are running (Terminal 2)

**Can't login?**
→ Check backend is running (Terminal 1)

**Frontend not loading?**
→ Check frontend is running (Terminal 3)

---

Enjoy your AI-Powered Radiology System! 🎉

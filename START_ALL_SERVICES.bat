@echo off
echo ========================================
echo Starting AI-Powered Radiology System
echo ========================================
echo.

REM Start Backend Server
echo [1/5] Starting Backend Server...
start "Backend Server" cmd /k "cd /d %~dp0integrated-backend && npm run dev"
timeout /t 5 /nobreak >nul

REM Start ML Workers (Node.js)
echo [2/5] Starting ML Workers (Node.js)...
start "ML Workers" cmd /k "cd /d %~dp0integrated-backend && npm run start:ml-models"
timeout /t 5 /nobreak >nul

REM Start MRI Python Worker (Real AI Model)
echo [3/5] Starting MRI AI Worker (Python)...
start "MRI AI Worker" powershell -NoExit -Command "cd '%~dp0integrated-backend\prioritization-ml'; Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force; .\venv\Scripts\Activate.ps1; python start_mri_worker.py"
timeout /t 3 /nobreak >nul

REM Start X-Ray Python Worker (DuoFormer Model)
echo [4/5] Starting X-Ray AI Worker (Python)...
start "X-Ray AI Worker" powershell -NoExit -Command "cd '%~dp0integrated-backend\prioritization-ml'; Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force; .\venv\Scripts\Activate.ps1; python start_xray_worker.py"
timeout /t 3 /nobreak >nul

REM Start Frontend
echo [5/5] Starting Frontend...
start "Frontend Server" cmd /k "cd /d %~dp0RadiologyFrontend && npm run dev"

echo.
echo ========================================
echo All services started!
echo ========================================
echo.
echo Backend:  http://localhost:3002
echo Frontend: http://localhost:8080
echo.
echo ML Workers Running:
echo - X-Ray Analysis (DuoFormer - REAL AI MODEL)
echo - MRI Analysis (EfficientNet-B0 - REAL AI MODEL)
echo.
echo Press any key to exit this window...
pause >nul

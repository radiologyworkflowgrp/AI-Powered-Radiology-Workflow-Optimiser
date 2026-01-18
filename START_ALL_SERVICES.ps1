# AI-Powered Radiology Workflow Optimizer - Startup Script
# This PowerShell script starts all services including ML workers

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting AI-Powered Radiology System" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get the script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Start Backend Server
Write-Host "[1/3] Starting Backend Server..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$scriptDir\integrated-backend'; Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force; npm run dev" -WindowStyle Normal
Start-Sleep -Seconds 5

# Start ML Workers
Write-Host "[2/3] Starting ML Workers..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$scriptDir\integrated-backend'; Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force; .\prioritization-ml\venv\Scripts\Activate.ps1; npm run start:ml-models" -WindowStyle Normal
Start-Sleep -Seconds 5

# Start Frontend
Write-Host "[3/3] Starting Frontend..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$scriptDir\RadiologyFrontend'; Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force; npm run dev" -WindowStyle Normal

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "All services started!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Backend:  http://localhost:3002" -ForegroundColor White
Write-Host "Frontend: http://localhost:8080" -ForegroundColor White
Write-Host ""
Write-Host "ML Workers are running and will process DICOM uploads automatically!" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

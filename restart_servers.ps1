#!/usr/bin/env pwsh
# PowerShell script to restart both frontend and backend

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  AI Interview Assistant - Restart Script" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

$projectRoot = Split-Path -Parent $PSScriptRoot

# Test if backend is running
Write-Host "Testing backend connection..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/api/auth/dev-login" -Method Get -TimeoutSec 2 -ErrorAction Stop
    Write-Host "✓ Backend is running" -ForegroundColor Green
} catch {
    Write-Host "✗ Backend not responding" -ForegroundColor Red
    Write-Host "  Starting backend in new window..." -ForegroundColor Yellow
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$projectRoot\backend'; python server.py"
    Start-Sleep -Seconds 3
}

# Test if frontend is running
Write-Host "Testing frontend connection..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:3000" -Method Get -TimeoutSec 2 -ErrorAction Stop
    Write-Host "✓ Frontend is running" -ForegroundColor Green
} catch {
    Write-Host "✗ Frontend not responding" -ForegroundColor Red
    Write-Host "  Starting frontend in new window..." -ForegroundColor Yellow
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$projectRoot\frontend'; npm start"
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Setup Instructions:" -ForegroundColor White
Write-Host "1. Backend: http://localhost:8000" -ForegroundColor White
Write-Host "2. Frontend: http://localhost:3000" -ForegroundColor White
Write-Host "3. Run test: cd backend && python test_connection.py" -ForegroundColor White
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

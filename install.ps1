# One-command installer for Q-SFX (Windows PowerShell)
# 1. Sets up backend Python virtual environment (backend\.venv) and installs requirements.txt
# 2. Installs frontend Node.js packages and copies .env.local if not present
# 3. Builds the frontend Next.js application
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

Write-Host "=============================================="
Write-Host "          Q-SFX Installation Setup            "
Write-Host "=============================================="

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python 3 is required but was not found in PATH."
}
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Write-Error "Node.js (>= 20.9) is required but was not found in PATH."
}
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Error "npm is required but was not found in PATH."
}

Write-Host "`n[1/3] Setting up backend Python virtual environment and dependencies..."
& backend\scripts\setup_env.ps1

Write-Host "`n[2/3] Installing frontend dependencies (npm)..."
npm install

if (-not (Test-Path frontend\.env.local)) {
    Copy-Item frontend\.env.local.example frontend\.env.local
    Write-Host "Created frontend\.env.local from frontend\.env.local.example"
}

Write-Host "`n[3/3] Building frontend application..."
Set-Location frontend
npm run build
Set-Location ..

Write-Host "`n=============================================="
Write-Host "          Installation Complete!              "
Write-Host "=============================================="
Write-Host "To start both backend and frontend, run:"
Write-Host "  .\start-all.ps1"
Write-Host ""
Write-Host "Note: Make sure models and assets are placed"
Write-Host "in backend\ (see README.md section 4) before"
Write-Host "running the backend."
Write-Host "=============================================="

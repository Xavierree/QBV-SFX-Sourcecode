# One-command start (Windows PowerShell): backend (port 8500, or $env:SFXDEPLOY_PORT) in the background, then the frontend (port 3000).
# First run: also installs the Node packages and builds the frontend. Stop the backend with backend\scripts\stop_server.ps1.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

if (-not (Test-Path backend\.venv\Scripts\python.exe)) { Write-Error 'no backend\.venv yet: run backend\scripts\setup_env.ps1 first' }
$port = if ($env:SFXDEPLOY_PORT) { $env:SFXDEPLOY_PORT } else { '8500' }
if ($env:SFXDEPLOY_PORT -and -not $env:INFERENCE_URL) { $env:INFERENCE_URL = "http://127.0.0.1:$port" }   # frontend follows a non-default port
& backend\scripts\run_server.ps1
if (-not (Test-Path node_modules)) { npm install }
if (-not (Test-Path frontend\.env.local)) { Copy-Item frontend\.env.local.example frontend\.env.local; Write-Host 'created frontend\.env.local from the example' }
Set-Location frontend
if (-not (Test-Path .next)) { npm run build }
Write-Host 'frontend: http://localhost:3000   (backend health: http://127.0.0.1:' + $port + '/health)'
npm run start

# Idempotent launcher for the Q-SFX backend (localhost only; port 8500, or $env:SFXDEPLOY_PORT), Windows PowerShell.
# Does nothing if the server is already healthy; refuses to start if model/data assets are missing.
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')

$port = if ($env:SFXDEPLOY_PORT) { $env:SFXDEPLOY_PORT } else { '8500' }
$health = $null
try { $health = Invoke-RestMethod -TimeoutSec 2 "http://127.0.0.1:$port/health" } catch {}
if ($health) {
  if ($health.root -ieq (Get-Location).Path) { Write-Host "already running and healthy (port $port)"; exit 0 }
  Write-Host "port $port is already answered by a different backend (not $((Get-Location).Path))."
  Write-Host "Stop that one, or start this one on another port: SFXDEPLOY_PORT=8501 and INFERENCE_URL=http://127.0.0.1:8501 for the frontend."
  exit 1
}

python scripts\check_assets.py
if ($LASTEXITCODE -ne 0) { Write-Host 'backend not started: assets missing (see list above)'; exit 1 }

New-Item -ItemType Directory -Force logs | Out-Null
if (Test-Path logs\server.pid) {
  $old = Get-Process -Id ([int](Get-Content logs\server.pid)) -ErrorAction SilentlyContinue
  if ($old) { Write-Host "stale process $($old.Id) not responding to /health -- stopping it first"; Stop-Process -Id $old.Id -Force; Start-Sleep 2 }
}
$p = Start-Process -PassThru -WindowStyle Hidden -FilePath .venv\Scripts\python.exe -ArgumentList 'server\main.py' `
     -RedirectStandardOutput logs\server.log -RedirectStandardError logs\server.err.log
$p.Id | Set-Content logs\server.pid
Write-Host "started, pid $($p.Id) -- check logs\server.log and logs\server.err.log; health takes a few seconds to come up"

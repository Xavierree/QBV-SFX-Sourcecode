# Idempotent stopper for the Q-SFX backend (Windows PowerShell). Frees the GPU memory held by the models.
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
if (-not (Test-Path logs\server.pid)) { Write-Host 'no pidfile -- server does not appear to be running'; exit 0 }
$procId = [int](Get-Content logs\server.pid)
$p = Get-Process -Id $procId -ErrorAction SilentlyContinue
if (-not $p) { Write-Host "pidfile present ($procId) but process is not running -- removing stale pidfile" }
else { Write-Host "stopping pid $procId"; Stop-Process -Id $procId -Force; Write-Host 'stopped' }
Remove-Item logs\server.pid -Force

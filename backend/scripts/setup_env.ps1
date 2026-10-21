# Build backend\.venv and install ..\requirements.txt into it (Windows PowerShell).
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
if (-not (Test-Path .venv\Scripts\python.exe)) { python -m venv .venv }
.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.venv\Scripts\python.exe -m pip install -r ..\requirements.txt
.venv\Scripts\python.exe -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available())"

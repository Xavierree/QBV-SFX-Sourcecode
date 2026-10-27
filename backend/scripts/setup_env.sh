#!/usr/bin/env bash
# Build backend/.venv and install ../requirements.txt into it (the venv, the pip cache and temp files all stay
# inside backend/). If python3 has no ensurepip, the venv is created without pip and pip is installed from outside.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$PWD"; mkdir -p .cache/pip .cache/tmp
export PIP_CACHE_DIR="$ROOT/.cache/pip" TMPDIR="$ROOT/.cache/tmp"
PY_BIN="python3"
command -v python3 >/dev/null 2>&1 || PY_BIN="python"

if [ ! -x .venv/bin/python ]; then
  $PY_BIN -m venv .venv 2>/dev/null || { rm -rf .venv; $PY_BIN -m venv --without-pip .venv; $PY_BIN -m pip --python .venv/bin/python install -q pip setuptools wheel 2>/dev/null || curl -sS https://bootstrap.pypa.io/get-pip.py | .venv/bin/python; }
fi
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r "$ROOT/../requirements.txt"
echo "done: $(.venv/bin/python -c 'import torch, transformers; print("torch", torch.__version__, "| transformers", transformers.__version__, "| cuda", torch.cuda.is_available())')"

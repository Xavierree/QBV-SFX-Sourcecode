#!/usr/bin/env bash
# Build backend/.venv and install ../requirements.txt into it (the venv, the pip cache and temp files all stay
# inside backend/). If python3 has no ensurepip, the venv is created without pip and pip is installed from outside.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$PWD"; mkdir -p .cache/pip .cache/tmp
export PIP_CACHE_DIR="$ROOT/.cache/pip" TMPDIR="$ROOT/.cache/tmp"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv || { rm -rf .venv; python3 -m venv --without-pip .venv; python3 -m pip --python .venv/bin/python install -q pip setuptools wheel; }
fi
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r ../requirements.txt
echo "done: $(.venv/bin/python -c 'import torch, transformers; print("torch", torch.__version__, "| transformers", transformers.__version__, "| cuda", torch.cuda.is_available())')"

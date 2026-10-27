#!/usr/bin/env bash
# One-command installer for Q-SFX:
# 1. Sets up backend Python virtual environment (backend/.venv) and installs requirements.txt
# 2. Installs frontend Node.js packages and copies .env.local if not present
# 3. Builds the frontend Next.js application
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "=============================================="
echo "          Q-SFX Installation Setup            "
echo "=============================================="

# Check for Python
if command -v python3 >/dev/null 2>&1; then
  PY="python3"
elif command -v python >/dev/null 2>&1; then
  PY="python"
else
  echo "Error: Python 3 is required but was not found in PATH." >&2
  exit 1
fi

# Check for Node.js and npm
if ! command -v node >/dev/null 2>&1; then
  echo "Error: Node.js (>= 20.9) is required but was not found in PATH." >&2
  exit 1
fi
if ! command -v npm >/dev/null 2>&1; then
  echo "Error: npm is required but was not found in PATH." >&2
  exit 1
fi

echo ""
echo "[1/3] Setting up backend Python virtual environment and dependencies..."
bash backend/scripts/setup_env.sh

echo ""
echo "[2/3] Installing frontend dependencies (npm)..."
npm install

if [ ! -f frontend/.env.local ]; then
  cp frontend/.env.local.example frontend/.env.local
  echo "Created frontend/.env.local from frontend/.env.local.example"
fi

echo ""
echo "[3/3] Building frontend application..."
(cd frontend && npm run build)

echo ""
echo "=============================================="
echo "          Installation Complete!              "
echo "=============================================="
echo "To start both backend and frontend, run:"
echo "  ./start-all.sh"
echo ""
echo "Note: Make sure models and assets are placed"
echo "in backend/ (see README.md section 4) before"
echo "running the backend."
echo "=============================================="

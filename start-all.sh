#!/usr/bin/env bash
# One-command start: backend (port 8500, or $SFXDEPLOY_PORT) in the background, then the frontend (port 3000) in the foreground.
# First run: also installs the Node packages and builds the frontend. Ctrl+C stops the frontend only;
# stop the backend with backend/scripts/stop_server.sh.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

[ -x backend/.venv/bin/python ] || { echo "no backend/.venv yet: run backend/scripts/setup_env.sh first"; exit 1; }
PORT="${SFXDEPLOY_PORT:-8500}"
[ -n "${SFXDEPLOY_PORT:-}" ] && export INFERENCE_URL="${INFERENCE_URL:-http://127.0.0.1:$PORT}"   # frontend follows a non-default port
backend/scripts/run_server.sh                       # checks the assets first and refuses to start if any are missing
[ -d node_modules ] || npm install
[ -f frontend/.env.local ] || { cp frontend/.env.local.example frontend/.env.local; echo "created frontend/.env.local from the example"; }
[ -d frontend/.next ] || (cd frontend && npm run build)
echo "frontend: http://localhost:3000   (backend health: curl http://127.0.0.1:$PORT/health)"
cd frontend && exec npm run start

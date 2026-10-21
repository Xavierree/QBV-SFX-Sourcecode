#!/bin/bash
# Idempotent launcher for the Q-SFX inference backend (localhost only; port 8500, or $SFXDEPLOY_PORT). Safe to run repeatedly:
# does nothing if the server is already up and healthy, and cleans up a stale PID file if the process died.
# The frontend's "Start backend" button on /admin runs this same script.
set -euo pipefail
cd "$(dirname "$0")/.."   # backend/

PIDFILE=logs/server.pid
LOGFILE=logs/server.log
mkdir -p logs

PORT="${SFXDEPLOY_PORT:-8500}"
HEALTH="$(curl -s -m 2 "http://127.0.0.1:$PORT/health" || true)"
if [ -n "$HEALTH" ]; then
  if echo "$HEALTH" | grep -q "\"root\":\"$(pwd -P)\""; then
    echo "already running and healthy (port $PORT)"
    exit 0
  fi
  # Something else answers on this port (another copy of the backend, or an older build): do NOT treat it as ours.
  echo "port $PORT is already answered by a different backend (not $(pwd -P))." >&2
  echo "Stop that one, or start this one on another port: SFXDEPLOY_PORT=8501 and INFERENCE_URL=http://127.0.0.1:8501 for the frontend." >&2
  exit 1
fi

# refuse to start (with a readable list) if models/data are missing
python3 scripts/check_assets.py || { echo "backend not started: assets missing (see list above)"; exit 1; }

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "stale process $(cat "$PIDFILE") not responding to /health -- stopping it first"
  kill "$(cat "$PIDFILE")" 2>/dev/null || true
  sleep 2
fi

nohup .venv/bin/python server/main.py >> "$LOGFILE" 2>&1 &
echo $! > "$PIDFILE"
disown
echo "started, pid $(cat "$PIDFILE") -- check $LOGFILE, health takes a few seconds to come up"

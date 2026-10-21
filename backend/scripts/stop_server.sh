#!/bin/bash
# Idempotent stopper for the Q-SFX inference backend. Safe to run repeatedly: no-ops cleanly if the server is
# already down or the PID file is stale. Frees the GPU memory held by the models.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root (backend/)

PIDFILE=logs/server.pid

if [ ! -f "$PIDFILE" ]; then
  echo "no pidfile -- server does not appear to be running"
  exit 0
fi

PID="$(cat "$PIDFILE")"

if ! kill -0 "$PID" 2>/dev/null; then
  echo "pidfile present ($PID) but process is not running -- removing stale pidfile"
  rm -f "$PIDFILE"
  exit 0
fi

echo "stopping pid $PID (SIGTERM)"
kill "$PID" 2>/dev/null || true

for _ in $(seq 1 10); do
  if ! kill -0 "$PID" 2>/dev/null; then
    rm -f "$PIDFILE"
    echo "stopped"
    exit 0
  fi
  sleep 1
done

echo "still alive after 10s -- sending SIGKILL"
kill -9 "$PID" 2>/dev/null || true
rm -f "$PIDFILE"
echo "stopped (forced)"

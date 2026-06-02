#!/bin/bash
# Script to launch training progress monitor in background
# Usage: ./ml2/scripts/start_monitor.sh [interval_seconds] [pid]

set -e

# Go to workspace root
cd "$(dirname "$0")/../.."

# Path to python in virtualenv
VENV_PYTHON="venv_ml2/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
  echo "Virtual env python not found at $VENV_PYTHON. Falling back to system python3."
  VENV_PYTHON="python3"
fi

# Make monitor_training.py executable just in case
chmod +x ml2/scripts/monitor_training.py

INTERVAL=${1:-600} # default to 600s/10m
PID=$2

ARGS="--interval $INTERVAL"
if [ -n "$PID" ]; then
  ARGS="$ARGS --pid $PID"
fi

LOG_FILE="ml2/runs/monitor_training.log"
mkdir -p ml2/runs

echo "[ML2 Monitor] Starting monitor in background..."
echo "  - Python: $VENV_PYTHON"
echo "  - Command: $VENV_PYTHON ml2/scripts/monitor_training.py $ARGS"
echo "  - Logs: tail -f $LOG_FILE"

# Run in background using nohup to prevent it from terminating when session closes
nohup "$VENV_PYTHON" ml2/scripts/monitor_training.py $ARGS >/dev/null 2>&1 &

MONITOR_PID=$!

echo "[ML2 Monitor] Monitor started successfully with PID: $MONITOR_PID"
echo "To stop monitoring, run: kill $MONITOR_PID"

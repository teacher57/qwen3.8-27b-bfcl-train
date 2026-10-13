#!/bin/bash
# start|stop|status for fs_update.py (copies the fs_env run's JSONL + status JSON from the pod every 2 min; never touches the notebook)
cd "$(dirname "$0")" || exit 1
PID=fs_updater.pid; LOG=fs_updater.log
running() { [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; }
case "$1" in
  start) running && { echo "already running"; exit 0; }; nohup ../.venv/bin/python -u fs_update.py >> "$LOG" 2>&1 & echo $! > "$PID"; disown 2>/dev/null; echo "started (pid $(cat $PID))" ;;
  stop) running && kill "$(cat "$PID")"; rm -f "$PID"; echo stopped ;;
  status) running && echo running || echo "not running"; tail -n 2 "$LOG" 2>/dev/null ;;
  *) echo "usage: $0 start|stop|status" ;;
esac

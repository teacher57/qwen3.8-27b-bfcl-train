#!/bin/bash
# start|stop|status for grpo_update.py (copies the GRPO run's status/log/probe JSON from the pod every 2 min; never touches the notebook)
cd "$(dirname "$0")" || exit 1
PID=grpo_updater.pid; LOG=grpo_updater.log
running() { [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; }
case "$1" in
  start) running && { echo "already running"; exit 0; }; nohup ../.venv/bin/python -u grpo_update.py >> "$LOG" 2>&1 & echo $! > "$PID"; disown 2>/dev/null; echo "started (pid $(cat $PID))" ;;
  stop) running && kill "$(cat "$PID")"; rm -f "$PID"; echo stopped ;;
  status) running && echo running || echo "not running"; tail -n 2 "$LOG" 2>/dev/null ;;
  *) echo "usage: $0 start|stop|status" ;;
esac

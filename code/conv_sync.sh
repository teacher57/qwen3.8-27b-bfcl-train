#!/bin/bash
# start|stop|status for sync_conversations.py (copies the pod's conversations and scores every 5 min)
cd "$(dirname "$0")" || exit 1
PID=conv_sync.pid; LOG=conv_sync.log
running() { [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; }
case "$1" in
  start) running && { echo "already running"; exit 0; }; nohup ../.venv/bin/python -u sync_conversations.py >> "$LOG" 2>&1 & echo $! > "$PID"; disown 2>/dev/null; echo "started (pid $(cat $PID))" ;;
  stop) running && kill "$(cat "$PID")"; rm -f "$PID"; echo stopped ;;
  status) running && echo running || echo "not running"; tail -n 2 "$LOG" 2>/dev/null ;;
  *) echo "usage: $0 start|stop|status" ;;
esac

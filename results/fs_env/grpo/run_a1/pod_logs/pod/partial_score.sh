#!/bin/bash
# While BFCL generates: every 5 min, snapshot the complete result lines of each category and score them with
# `bfcl evaluate --partial-eval` into /root/bfcl-results/partial/ (the official result/ and score/ are never touched),
# so the notebook can show "passed so far". Exits after one last pass once run_multiturn.sh logs MULTITURN_DONE.
#   MODEL=qwen3.8-27b-grpo20-FC setsid nohup /root/partial_score.sh > /root/partial_score.out 2>&1 < /dev/null &
M=${MODEL:?}; R=/root/bfcl-results; P=$R/partial
export BFCL_PROJECT_ROOT=$R
cd /root/gorilla/berkeley-function-call-leaderboard || exit 1
while true; do
  # done = run_multiturn.sh logged MULTITURN_DONE for this model (checking for a running "bfcl generate" raced at startup:
  # started in the same second, the first check found nothing yet and the loop quit after one empty pass)
  grep -q "MULTITURN_DONE ${M}" /root/pipeline.log 2>/dev/null && alive=0 || alive=1
  for f in $R/result/$M/multi_turn/BFCL_v4_*_result.json; do
    [ -f "$f" ] || continue
    cat=$(basename $f | sed -E 's/BFCL_v4_(.*)_result.json/\1/'); mkdir -p $P/result/$M/multi_turn
    python3 - "$f" "$P/result/$M/multi_turn/$(basename $f)" <<'PY'
import json, sys
ok = []
for l in open(sys.argv[1], errors="replace"):
    try: json.loads(l); ok.append(l if l.endswith("\n") else l + "\n")
    except ValueError: pass
open(sys.argv[2], "w").writelines(ok)
PY
    /root/bfcl-venv/bin/bfcl evaluate --model $M --test-category $cat --partial-eval --result-dir $P/result --score-dir $P/score >> /root/partial_score.log 2>&1
  done
  echo "$(date -u +%T) partial scores: $(for s in $P/score/$M/multi_turn/*_score.json; do [ -f $s ] && echo -n "$(basename $s _score.json | sed s/BFCL_v4_multi_turn_//)=$(head -1 $s | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d["correct_count"],"/",d["total_count"],sep="")') "; done)"
  [ "$alive" = 0 ] && break
  sleep 300
done

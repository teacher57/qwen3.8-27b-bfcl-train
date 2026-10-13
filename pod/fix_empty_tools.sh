#!/bin/bash
# One-off repair, run AFTER run.sh has finished (RUN_DONE): the 4 live_irrelevance entries with no tools failed with
# HTTP 400 under the old handler (vLLM rejects `tools: []`). Install the fixed handler, drop those 4 results,
# regenerate just them (BFCL fills in missing ids), then rescore live_irrelevance and the overall summary.
set -eu
log() { echo "$(date -u +%FT%TZ) $*" >> /root/pipeline.log; }
pgrep -f "[b]fcl generate" > /dev/null && { echo "bfcl generate is still running; wait for RUN_DONE"; exit 1; }
export BFCL_PROJECT_ROOT=/root/bfcl-results QWEN38_BASE_URL=http://127.0.0.1:8000/v1 QWEN38_MAX_TOKENS=16384
cp /root/qwen38_vllm.py /root/gorilla/berkeley-function-call-leaderboard/bfcl_eval/model_handler/api_inference/qwen38_vllm.py
F=$(find $BFCL_PROJECT_ROOT/result/qwen3.8-27b-FC -name BFCL_v4_live_irrelevance_result.json)
cp "$F" "$F.before_empty_tools_fix"
/root/bfcl-venv/bin/python - "$F" <<'EOF'
import json, sys
p = sys.argv[1]; keep = []
for l in open(p):
    if not l.strip(): continue
    r = json.loads(l)
    if "tools` must not be an empty array" in str(r.get("result", "")): print("dropping", r["id"]); continue
    keep.append(l if l.endswith("\n") else l + "\n")
open(p, "w").writelines(keep)
EOF
cd /root/gorilla/berkeley-function-call-leaderboard
B=/root/bfcl-venv/bin/bfcl
log "fix: regenerating the 4 no-tool live_irrelevance entries"
$B generate --model qwen3.8-27b-FC --test-category live_irrelevance --num-threads 4 >> /root/bfcl_generate.log 2>&1
CATS=$(/root/bfcl-venv/bin/python -c "import re; s=open('/root/run.sh').read(); print(','.join(re.search(r'CATEGORIES:-\"(.*?)\"}', s, re.S).group(1).replace('\\\\','').split()))")
$B evaluate --model qwen3.8-27b-FC --test-category live_irrelevance >> /root/bfcl_evaluate.log 2>&1
$B evaluate --model qwen3.8-27b-FC --test-category "$CATS" >> /root/bfcl_evaluate.log 2>&1
log "fix: done, live_irrelevance inference errors left: $(grep -c "Error during inference" "$F")"

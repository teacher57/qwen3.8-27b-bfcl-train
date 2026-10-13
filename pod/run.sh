#!/bin/bash
# Full BFCL (minus web_search, which needs a paid SerpAPI key) on qwen3.8-27b-FC, thinking on, against serve.sh's vLLM.
# All categories go through ONE `bfcl generate`, so a slow entry only holds 1 of the 32 slots instead of stalling the
# GPU at the end of every category. eval_loop.py scores each category as soon as it is complete.
# Generation resumes where it stopped (BFCL skips ids that already have results), so rerunning this is safe.
#   setsid nohup /root/run.sh > /root/run.out 2>&1 < /dev/null &
set -u
log() { echo "$(date -u +%FT%TZ) $*" >> /root/pipeline.log; }
MODEL=qwen3.8-27b-FC
THREADS=${THREADS:-32}   # = vLLM --max-num-seqs
CATEGORIES=${CATEGORIES:-"simple_python simple_java simple_javascript parallel multiple parallel_multiple irrelevance \
live_simple live_multiple live_parallel live_parallel_multiple live_irrelevance live_relevance \
multi_turn_base multi_turn_miss_func multi_turn_miss_param multi_turn_long_context \
memory_kv memory_vector memory_rec_sum"}
CATS_COMMA=$(echo $CATEGORIES | tr " " ",")

export BFCL_PROJECT_ROOT=/root/bfcl-results   # where BFCL writes result/ and score/
export QWEN38_BASE_URL=http://127.0.0.1:8000/v1
export QWEN38_MAX_TOKENS=${QWEN38_MAX_TOKENS:-16384}
mkdir -p "$BFCL_PROJECT_ROOT"
cd /root/gorilla/berkeley-function-call-leaderboard
B=/root/bfcl-venv/bin/bfcl

log "run: start ($MODEL, threads $THREADS, max_tokens $QWEN38_MAX_TOKENS)"
$B generate --model $MODEL --test-category $CATS_COMMA --num-threads $THREADS >> /root/bfcl_generate.log 2>&1 &
sleep 30
/root/bfcl-venv/bin/python /root/eval_loop.py "$CATEGORIES" &
wait %1 && log "generate: finished" || log "generate FAILED (see /root/bfcl_generate.log)"
wait
# Final evaluation over everything, so the overall/summary CSVs include every category.
$B evaluate --model $MODEL --test-category $CATS_COMMA >> /root/bfcl_evaluate.log 2>&1 && log "evaluate: all" || log "evaluate FAILED: all"
log "RUN_DONE"

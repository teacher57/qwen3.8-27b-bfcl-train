#!/bin/bash
# BFCL multi-turn on a LoRA model (registered by install_handler.py NAME), same settings as the base-model baseline
# (temp 0.001 = BFCL default, thinking on, 16,384-token cap per call). Phase 1: miss_func + miss_param, scored right
# away; phase 2: base + long_context; then one evaluate over all four. Resumable (BFCL skips finished ids).
#   MODEL=qwen3.8-27b-grpo30-FC setsid nohup /root/run_multiturn.sh > /root/run_multiturn.out 2>&1 < /dev/null &
set -u
MODEL=${MODEL:?set MODEL to the BFCL registry name}
THREADS=${THREADS:-32}
log() { echo "$(date -u +%FT%TZ) $*" >> /root/pipeline.log; }
export BFCL_PROJECT_ROOT=/root/bfcl-results QWEN38_BASE_URL=http://127.0.0.1:8000/v1 QWEN38_MAX_TOKENS=${QWEN38_MAX_TOKENS:-16384}
mkdir -p "$BFCL_PROJECT_ROOT"; cd /root/gorilla/berkeley-function-call-leaderboard
B=/root/bfcl-venv/bin/bfcl
for phase in "multi_turn_miss_func,multi_turn_miss_param" "multi_turn_base,multi_turn_long_context"; do
  log "multiturn $MODEL: generate $phase (threads $THREADS)"
  $B generate --model $MODEL --test-category $phase --num-threads $THREADS >> /root/bfcl_generate.log 2>&1 || log "generate FAILED: $phase"
  $B evaluate --model $MODEL --test-category $phase >> /root/bfcl_evaluate.log 2>&1 && log "evaluate: $phase" || log "evaluate FAILED: $phase"
done
$B evaluate --model $MODEL --test-category multi_turn >> /root/bfcl_evaluate.log 2>&1 && log "evaluate: multi_turn (all four)" || log "evaluate FAILED: multi_turn"
log "MULTITURN_DONE $MODEL"

#!/bin/bash
# After GRPO run_v1 ends: held-out eval of the final GRPO adapter and the SFT v1 start, in parallel, same seed.
# 73 eval tasks x 4 trials, temperature 1, causes A3,D,E first. Log: /root/eval_v1/after.log
OUT=/root/eval_v1; mkdir -p $OUT; cd /root
log() { echo "$(date -u +%FT%TZ) $*" >> $OUT/after.log; }
log "waiting for GRPO to end"
while pgrep -f "[f]s_env.grpo_train" >/dev/null; do sleep 60; done
STAGE=$(python3 -c 'import json;print(json.load(open("/root/grpo/run_v1/status.json"))["stage"])')
log "GRPO ended: stage=$STAGE"
case "$STAGE" in finished*|stopped*) ;; *) log "GRPO did not finish cleanly; not running eval"; echo EVAL_SKIPPED > $OUT/DONE; exit 1;; esac
LAST=$(ls -d /root/grpo/run_v1/adapters/step_* | grep -v _vllm | sed 's/.*step_//' | sort -n | tail -1)
log "final adapter: step_$LAST"
curl -s -X POST localhost:8000/wake_up >/dev/null
U() { curl -s -X POST localhost:8000/v1/unload_lora_adapter -H 'Content-Type: application/json' -d "{\"lora_name\":\"$1\"}" >/dev/null; }
L() { curl -s -X POST localhost:8000/v1/load_lora_adapter -H 'Content-Type: application/json' -d "{\"lora_name\":\"$1\",\"lora_path\":\"$2\"}"; }
for m in $(curl -s localhost:8000/v1/models | python3 -c 'import sys,json;print(" ".join(m["id"] for m in json.load(sys.stdin)["data"] if m["id"]!="qwen3.8-27b"))'); do U $m; done
log "load grpo-final: $(L grpo-final /root/grpo/run_v1/adapters/step_${LAST}_vllm)"
test -d /root/sft/run_v1/final_adapter_vllm || /root/train-venv/bin/python -m fs_env.adapter_fix /root/sft/run_v1/final_adapter /root/sft/run_v1/final_adapter_vllm
log "load sft-v1: $(L sft-v1 /root/sft/run_v1/final_adapter_vllm)"
ORDER=A3,D,E,A4,B,C,F,G,H,A1,A2
for m in grpo-final sft-v1; do
  /root/train-venv/bin/python -u -m fs_env.run --model $m --split eval --tasks $ORDER --trials 4 --seed 13 --temperature 1 \
     --threads 16 --out $OUT/$m.jsonl > $OUT/$m.log 2>&1 &
done
wait
log "eval done: grpo-final $(tail -2 $OUT/grpo-final.log | tr '\n' ' ') | sft-v1 $(tail -2 $OUT/sft-v1.log | tr '\n' ' ')"
echo EVAL_DONE > $OUT/DONE

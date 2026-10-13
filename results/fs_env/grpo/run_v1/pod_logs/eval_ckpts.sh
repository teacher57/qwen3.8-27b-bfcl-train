#!/bin/bash
# Held-out fs_env eval of GRPO run_v1 steps 10 and 20, same settings as /root/after_grpo_eval.sh (seed 13, 4 trials, temp 1)
OUT=/root/eval_v1; cd /root
log() { echo "$(date -u +%FT%TZ) $*" >> $OUT/ckpts.log; }
U() { curl -s -X POST localhost:8000/v1/unload_lora_adapter -H "Content-Type: application/json" -d "{\"lora_name\":\"$1\"}" >/dev/null; }
L() { curl -s -X POST localhost:8000/v1/load_lora_adapter -H "Content-Type: application/json" -d "{\"lora_name\":\"$1\",\"lora_path\":\"$2\"}"; }
for m in $(curl -s localhost:8000/v1/models | python3 -c "import sys,json;print(\" \".join(m[\"id\"] for m in json.load(sys.stdin)[\"data\"] if m[\"id\"]!=\"qwen3.8-27b\"))"); do U $m; done
for s in 10 20; do
  /root/train-venv/bin/python -m fs_env.adapter_fix /root/grpo/run_v1/adapters/step_$s /dev/shm/step_${s}_vllm >> $OUT/ckpts.log 2>&1
  log "load grpo-s$s: $(L grpo-s$s /dev/shm/step_${s}_vllm)"
done
for s in 10 20; do
  /root/train-venv/bin/python -u -m fs_env.run --model grpo-s$s --split eval --tasks A3,D,E,A4,B,C,F,G,H,A1,A2 --trials 4 --seed 13 \
     --temperature 1 --threads 16 --out $OUT/grpo-s$s.jsonl > $OUT/grpo-s$s.log 2>&1 &
done
wait
log "done: s10 $(tail -1 $OUT/grpo-s10.log) | s20 $(tail -1 $OUT/grpo-s20.log)"
echo CKPTS_DONE > $OUT/CKPTS_DONE

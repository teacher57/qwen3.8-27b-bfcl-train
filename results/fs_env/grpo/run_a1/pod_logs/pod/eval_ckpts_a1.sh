#!/bin/bash
# Test GRPO run_a1 checkpoints 10/20/30/40 on the A1-I/II/III/VI test tasks (a1_brake.json), 4 trials, temp 1, seed 13,
# up to 20 model calls per turn: the same settings as the step-20 baseline. Stops on any failed check. Log: /root/pipeline.log
log() { echo "$(date -u +%FT%TZ) $*" >> /root/pipeline.log; }
fail() { log "CKPT_EVAL STOPPED: $*"; echo STOPPED > /root/a1_eval_run_a1/DONE; exit 1; }
O=/root/a1_eval_run_a1; mkdir -p $O; cd /root
for s in 10 20 30 40; do
  [ -d /root/grpo/run_a1/adapters/step_${s}_vllm ] || /root/train-venv/bin/python -m fs_env.adapter_fix /root/grpo/run_a1/adapters/step_$s /root/grpo/run_a1/adapters/step_${s}_vllm >> $O/adapter_fix.log 2>&1 || fail "adapter_fix step $s"
done
pkill -f "vllm serve"; for i in $(seq 1 30); do pgrep -f "vllm serve" >/dev/null || break; sleep 2; done
A=/root/grpo/run_a1/adapters
(setsid nohup env PATH=/root/venv/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 /root/venv/bin/vllm serve /dev/shm/q38 --served-model-name qwen3.8-27b \
   --host 127.0.0.1 --port 8000 --max-model-len 32768 --max-num-seqs 32 --gpu-memory-utilization 0.90 \
   --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --limit-mm-per-prompt '{"image":0,"video":0}' \
   --enable-lora --max-lora-rank 16 --max-loras 4 \
   --lora-modules run_a1-s10=$A/step_10_vllm run_a1-s20=$A/step_20_vllm run_a1-s30=$A/step_30_vllm run_a1-s40=$A/step_40_vllm \
   > /root/vllm_eval.log 2>&1 < /dev/null &)
for i in $(seq 1 120); do curl -s -m5 localhost:8000/v1/models | grep -q run_a1-s40 && break; grep -q Traceback /root/vllm_eval.log && fail "vLLM crashed"; sleep 10; done
curl -s -m5 localhost:8000/v1/models | grep -q run_a1-s40 || fail "vLLM not serving the checkpoints"
log "CKPT_EVAL vLLM up with 4 checkpoints"
for m in qwen3.8-27b run_a1-s10 run_a1-s20 run_a1-s30 run_a1-s40; do
  curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d "{\"model\":\"$m\",\"temperature\":0,\"max_tokens\":200,\"messages\":[{\"role\":\"user\",\"content\":\"Create a file called notes.txt in my home folder for my meeting notes.\"}],\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
    | /root/venv/bin/python -c 'import sys,json; d=json.load(sys.stdin); print(d["choices"][0]["message"]["content"][:300] if "choices" in d else "ERROR "+str(d))' > $O/sanity_$m.txt
done
grep -q "^ERROR" $O/sanity_*.txt && fail "a sanity request errored"
for s in 10 20 30 40; do cmp -s $O/sanity_qwen3.8-27b.txt $O/sanity_run_a1-s$s.txt && fail "LoRA run_a1-s$s not applied"; done
log "CKPT_EVAL sanity ok: every checkpoint changes the output"
for s in 10 20 30 40; do
  /root/venv/bin/python -u -m fs_env.run --model run_a1-s$s --task-set a1_brake.json --tasks A1-I,A1-II,A1-III,A1-VI --trials 4 --seed 13 \
     --temperature 1 --threads 8 --out $O/run_a1-s$s.jsonl > $O/run_a1-s$s.log 2>&1 &
done
wait
log "CKPT_EVAL_DONE $(for s in 10 20 30 40; do echo -n "s$s: $(tail -2 $O/run_a1-s$s.log | head -1) | "; done)"
echo DONE > $O/DONE

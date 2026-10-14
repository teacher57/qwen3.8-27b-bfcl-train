#!/bin/bash
# GRPO run_a1: from GRPO step 20, on a1_brake_train.json (A1-I/II/III/VI, one each per step) + 1 A3/D/E replay task per step.
log() { echo "$(date -u +%FT%TZ) $*" >> /root/pipeline.log; }
cd /root
(setsid nohup /root/pod/serve_grpo_h200.sh > /root/vllm.log 2>&1 < /dev/null &)
for i in $(seq 1 120); do curl -s -m5 localhost:8000/v1/models >/dev/null && break; grep -q Traceback /root/vllm.log && { log "vLLM crashed"; exit 1; }; sleep 10; done
log "vLLM up"
mkdir -p /root/grpo
(setsid nohup /root/train-venv/bin/python -u -m fs_env.grpo_train --init-adapter /root/sft/grpo20 --out /root/grpo/run_a1 \
   --task-set a1_brake_train.json --causes A1-I,A1-II,A1-III,A1-VI --replay-causes A3,D,E --replay-per-step 1 \
   --steps 40 --tasks-per-step 4 --group 8 --probe-every 10 > /root/grpo/run_a1.log 2>&1 < /dev/null &)
sleep 20
(RUN=/root/grpo/run_a1 EVERY=10 setsid nohup /root/pod/prune_ckpt.sh > /dev/null 2>&1 < /dev/null &)
log "GRPO run_a1 launched"

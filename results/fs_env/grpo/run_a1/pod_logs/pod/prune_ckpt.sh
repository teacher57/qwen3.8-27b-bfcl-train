#!/bin/bash
# While a GRPO run trains: keep adapters/step_N only for N % EVERY == 0 and the newest step, and drop the _vllm copies
# except the newest (a _vllm copy is a rename of its adapter, rebuildable with fs_env.adapter_fix). The 30 GB pod disk
# can't hold one ~0.9 GB checkpoint per step.   RUN=/root/grpo/run_a1 EVERY=10 setsid nohup /root/pod/prune_ckpt.sh &
A=${RUN:?}/adapters; EVERY=${EVERY:-10}; LOG=${RUN}/prune.log
while true; do
  newest=$(ls -d $A/step_* 2>/dev/null | grep -v _vllm | sed 's/.*step_//' | sort -n | tail -1)
  if [ -n "$newest" ]; then
    for d in $(ls -d $A/step_* 2>/dev/null); do
      n=$(basename $d | sed -E 's/step_([0-9]+).*/\1/')
      [ "$n" -ge "$newest" ] && continue
      if [[ $d == *_vllm ]] || [ $((n % EVERY)) -ne 0 ]; then rm -rf "$d" && echo "$(date -u +%T) removed $(basename $d) (newest step_$newest)" >> $LOG; fi
    done
  fi
  pgrep -f "[f]s_env.grpo_train" >/dev/null || { echo "$(date -u +%T) GRPO ended, pruner exits" >> $LOG; exit 0; }
  sleep 30
done

#!/bin/bash
# While GRPO run_v1 trains: keep adapters/step_N only for N%5==0 and the newest step; drop _vllm copies except the newest
# (a _vllm copy is a pure rename of its adapter, rebuildable with fs_env.adapter_fix). Log: /root/grpo/prune.log
A=/root/grpo/run_v1/adapters
while true; do
  newest=$(ls -d $A/step_* 2>/dev/null | grep -v _vllm | sed 's/.*step_//' | sort -n | tail -1)
  if [ -n "$newest" ]; then
    for d in $(ls -d $A/step_* 2>/dev/null); do
      n=$(basename $d | sed -E 's/step_([0-9]+).*/\1/')
      [ "$n" -ge "$newest" ] && continue
      if [[ $d == *_vllm ]] || [ $((n % 5)) -ne 0 ]; then rm -rf "$d" && echo "$(date -u +%T) removed $(basename $d) (newest step_$newest)" >> /root/grpo/prune.log; fi
    done
  fi
  pgrep -f "[f]s_env.grpo_train" >/dev/null || { echo "$(date -u +%T) GRPO ended, pruner exits" >> /root/grpo/prune.log; exit 0; }
  sleep 30
done

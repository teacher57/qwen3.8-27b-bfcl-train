#!/bin/bash
# Qwen3.8-27B in bf16 on one A100 80GB, 127.0.0.1:8000. Same flags as the τ³ run (tau-train/qwen3.8-27b/pod/serve.sh)
# except --max-model-len 65536: BFCL's multi_turn_long_context entries plus kept-in-history thinking can pass 32k.
# Only 16 of 64 layers are full attention (the rest Gated DeltaNet), so the longer context is cheap.
export PATH=/root/venv/bin:/usr/local/cuda/bin:$PATH   # ninja/nvcc for JIT kernels
export VLLM_USE_FLASHINFER_SAMPLER=0
exec /root/venv/bin/vllm serve /dev/shm/q38 \
  --served-model-name qwen3.8-27b \
  --host 127.0.0.1 --port 8000 \
  --max-model-len 65536 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --limit-mm-per-prompt "{\"image\":0,\"video\":0}"

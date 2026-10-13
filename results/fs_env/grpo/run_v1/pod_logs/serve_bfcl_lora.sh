#!/bin/bash
# vLLM for BFCL on a LoRA (one H200 141GB): base Qwen3.8-27B bf16 + the LoRA in LORA_PATH served as LORA_NAME.
# --max-model-len 131072: the baseline (serve.sh, 65536) lost 11 multi_turn_long_context entries to the limit.
#   LORA_NAME=qwen3.8-27b-grpo30 LORA_PATH=/dev/shm/step_30_vllm ./serve_bfcl_lora.sh   (LORA_PATH: adapter_fix output)
export PATH=/root/venv/bin:/usr/local/cuda/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0
exec /root/venv/bin/vllm serve /dev/shm/q38 \
  --served-model-name qwen3.8-27b \
  --host 127.0.0.1 --port 8000 \
  --max-model-len 131072 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --limit-mm-per-prompt "{\"image\":0,\"video\":0}" \
  --enable-lora --max-lora-rank 16 --max-loras 1 \
  --lora-modules "$LORA_NAME=$LORA_PATH"

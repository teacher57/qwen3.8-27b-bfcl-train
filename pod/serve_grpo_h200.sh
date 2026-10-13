#!/bin/bash
# vLLM for GRPO on one H200 141GB, sharing the GPU with the QLoRA trainer (fs_env/grpo_train.py):
#  - bf16 weights (~54 GB), no quantization; --gpu-memory-utilization 0.58 (~82 GB incl. KV cache), so the trainer's
#    ~45 GB peak fits beside it and the trainer can run with --no-sleep (sleep mode stays enabled as a fallback)
#  - runtime LoRA loading: the trainer hot-loads each new adapter via /v1/load_lora_adapter
# Driver 570 pod: /root/venv has the vllm 0.30.0+cu129 wheel.
export PATH=/root/venv/bin:/usr/local/cuda/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_SERVER_DEV_MODE=1 VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
exec /root/venv/bin/vllm serve /dev/shm/q38 \
  --served-model-name qwen3.8-27b \
  --host 127.0.0.1 --port 8000 \
  --max-model-len 32768 --max-num-seqs 32 --gpu-memory-utilization ${GPU_UTIL:-0.58} \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --limit-mm-per-prompt "{\"image\":0,\"video\":0}" \
  --enable-lora --max-lora-rank 16 --max-loras 2 \
  --enable-sleep-mode ${VLLM_EXTRA:-}

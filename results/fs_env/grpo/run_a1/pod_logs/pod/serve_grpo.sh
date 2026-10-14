#!/bin/bash
# vLLM for GRPO on one A100 80GB (-O0: torch.compile off; with FP8 + this hybrid model inductor fails with "auto_functionalized was not removed")
#, sharing the GPU with the QLoRA trainer (fs_env/grpo_train.py):
#  - FP8 weights (weight-only Marlin kernels on Ampere): ~27 GB instead of ~54 GB in bf16
#  - --gpu-memory-utilization 0.55 (~44 GB incl. KV cache), so the trainer's ~25 GB resident model fits beside it
#  - sleep mode: the trainer calls /sleep?level=1 before each update (weights go to CPU RAM) and /wake_up after
#  - runtime LoRA loading: the trainer hot-loads each new adapter via /v1/load_lora_adapter
export PATH=/root/venv/bin:/usr/local/cuda/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_SERVER_DEV_MODE=1 VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
# A100 has no FP8 compute: force the weight-only Marlin FP8 kernel (else vLLM picks cutlass w8a8, which fails on sm80)
export VLLM_TEST_FORCE_FP8_MARLIN=1
exec /root/venv/bin/vllm serve /dev/shm/q38 \
  --served-model-name qwen3.8-27b \
  --host 127.0.0.1 --port 8000 \
  --quantization ${VLLM_QUANT:-bitsandbytes} \
  --max-model-len 32768 --max-num-seqs 32 --gpu-memory-utilization ${GPU_UTIL:-0.55} \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --limit-mm-per-prompt "{\"image\":0,\"video\":0}" \
  --enable-lora --max-lora-rank 16 --max-loras 2 \
  --enable-sleep-mode ${VLLM_EXTRA:--O0}

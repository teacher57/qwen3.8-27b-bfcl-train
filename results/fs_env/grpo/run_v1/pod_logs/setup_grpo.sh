#!/bin/bash
# GRPO pod setup (H200, driver 570): vLLM cu129 venv, model on /dev/shm, training venv. Progress: /root/pipeline.log
cd /root
log() { echo "$(date -u +%FT%TZ) $*" >> /root/pipeline.log; }
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh >/root/uv_install.log 2>&1; }
export PATH=/root/.local/bin:$PATH UV_LINK_MODE=copy UV_NO_CACHE=1
log "setup: start"
VLLM_WHL="https://github.com/vllm-project/vllm/releases/download/v0.30.0/vllm-0.30.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
( uv venv /root/venv -p 3.12 -q && uv pip install -p /root/venv -q --torch-backend=cu129 "vllm @ $VLLM_WHL" "huggingface_hub[hf_xet]" > /root/vllm_install.log 2>&1 \
  && { uv pip uninstall -p /root/venv -q torchcodec >> /root/vllm_install.log 2>&1; echo VLLM_DONE >> /root/vllm_install.log; } || echo VLLM_FAILED >> /root/vllm_install.log ) &
( uv venv /root/hf-venv -p 3.12 -q && uv pip install -p /root/hf-venv -q "huggingface_hub[hf_xet]" > /root/dl.log 2>&1 \
  && /root/hf-venv/bin/hf download Qwen/Qwen3.8-27B --local-dir /dev/shm/q38 >> /root/dl.log 2>&1 \
  && echo DL_DONE >> /root/dl.log || echo DL_FAILED >> /root/dl.log ) &
( uv venv /root/train-venv -p 3.12 -q && uv pip install -p /root/train-venv -q --torch-backend=cu129 torch transformers peft bitsandbytes accelerate flash-linear-attention requests > /root/train_install.log 2>&1 \
  && echo TRAIN_DONE >> /root/train_install.log || echo TRAIN_FAILED >> /root/train_install.log ) &
wait
for f in vllm_install:VLLM dl:DL train_install:TRAIN; do grep -q "${f#*:}_DONE" /root/${f%:*}.log || { log "FAILED: ${f%:*} (see /root/${f%:*}.log)"; exit 1; }; done
log "SETUP_DONE (model $(du -sh /dev/shm/q38 | cut -f1))"

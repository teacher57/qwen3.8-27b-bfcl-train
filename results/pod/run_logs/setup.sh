#!/bin/bash
# Pod setup for BFCL on Qwen3.8-27B: vLLM venv, BFCL (gorilla repo, editable) in its own venv, model on the RAM disk.
# The pod disk is 30 GB and the bf16 model is ~56 GB, so the model goes to /dev/shm (lost if the pod restarts).
# Progress goes to /root/pipeline.log. Does not start vLLM (serve.sh) or the eval (run.sh).
cd /root
log() { echo "$(date -u +%FT%TZ) $*" >> /root/pipeline.log; }
export UV_LINK_MODE=copy UV_NO_CACHE=1
log "setup: start"
# Driver 580 (CUDA 13) runs the default vLLM wheel (torch cu130). On a driver-570 pod use the +cu129 wheel instead,
# see tau-train/qwen3.8-27b/pod/setup.sh.
( uv venv /root/venv -p 3.12 -q && uv pip install -p /root/venv -q "vllm==0.30.0" "huggingface_hub[hf_xet]" > /root/vllm_install.log 2>&1 \
  && echo VLLM_DONE >> /root/vllm_install.log || echo VLLM_FAILED >> /root/vllm_install.log ) &
( git clone -q --depth 1 https://github.com/ShishirPatil/gorilla /root/gorilla \
  && uv venv /root/bfcl-venv -p 3.12 -q \
  && uv pip install -p /root/bfcl-venv -q -e /root/gorilla/berkeley-function-call-leaderboard > /root/bfcl_install.log 2>&1 \
  && echo BFCL_DONE >> /root/bfcl_install.log || echo BFCL_FAILED >> /root/bfcl_install.log ) &
( uv venv /root/hf-venv -p 3.12 -q && uv pip install -p /root/hf-venv -q "huggingface_hub[hf_xet]" > /root/dl.log 2>&1 \
  && /root/hf-venv/bin/hf download Qwen/Qwen3.8-27B --local-dir /dev/shm/q38 >> /root/dl.log 2>&1 \
  && echo DL_DONE >> /root/dl.log || echo DL_FAILED >> /root/dl.log ) &
wait
for f in vllm_install:VLLM bfcl_install:BFCL dl:DL; do grep -q "${f#*:}_DONE" /root/${f%:*}.log || { log "FAILED: ${f%:*} (see /root/${f%:*}.log)"; exit 1; }; done
log "SETUP_DONE (model $(du -sh /dev/shm/q38 | cut -f1))"

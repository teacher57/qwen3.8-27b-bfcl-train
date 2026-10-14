#!/bin/bash
# BFCL v4 multi-turn (all 4 categories) on GRPO run_a1 step 40, same settings as the base and step-20 runs. Log: /root/pipeline.log
log() { echo "$(date -u +%FT%TZ) $*" >> /root/pipeline.log; }
fail() { log "BFCL STOPPED: $*"; exit 1; }
M=qwen3.8-27b-a1s40; cd /root
export PATH=/root/.local/bin:$PATH UV_LINK_MODE=copy TMPDIR=/dev/shm/uvtmp UV_CACHE_DIR=/dev/shm/uvcache; mkdir -p $TMPDIR $UV_CACHE_DIR
if [ ! -x /root/bfcl-venv/bin/bfcl ]; then
  log "BFCL install: start"
  [ -d /root/gorilla ] || git clone -q https://github.com/ShishirPatil/gorilla /root/gorilla || fail "git clone"
  (cd /root/gorilla && git checkout -q 6ea5797) || fail "checkout 6ea5797"
  uv venv /root/bfcl-venv -p 3.12 -q && uv pip install -p /root/bfcl-venv -q -e /root/gorilla/berkeley-function-call-leaderboard soundfile > /root/bfcl_install.log 2>&1 || fail "BFCL install (see /root/bfcl_install.log)"
fi
rm -rf /dev/shm/uvtmp /dev/shm/uvcache
log "BFCL installed at $(cd /root/gorilla && git log -1 --format=%h); disk free $(df -h / | tail -1 | awk '{print $4}')"
/root/bfcl-venv/bin/python /root/pod/install_handler.py /root/gorilla/berkeley-function-call-leaderboard $M >> /root/pipeline.log 2>&1 || fail "install_handler"
pkill -f "vllm serve"; for i in $(seq 1 30); do pgrep -f "vllm serve" >/dev/null || break; sleep 2; done
(LORA_NAME=$M LORA_PATH=/root/grpo/run_a1/adapters/step_40_vllm setsid nohup /root/pod/serve_bfcl_lora.sh > /root/vllm_bfcl.log 2>&1 < /dev/null &)
for i in $(seq 1 120); do curl -s -m5 localhost:8000/v1/models | grep -q "$M" && break; grep -q Traceback /root/vllm_bfcl.log && fail "vLLM crashed"; sleep 10; done
curl -s -m5 localhost:8000/v1/models | grep -q "$M" || fail "vLLM not serving $M"
for m in qwen3.8-27b $M; do
  curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d "{\"model\":\"$m\",\"temperature\":0,\"max_tokens\":200,\"messages\":[{\"role\":\"user\",\"content\":\"Create a file called notes.txt in my home folder for my meeting notes.\"}],\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
    | /root/venv/bin/python -c 'import sys,json; d=json.load(sys.stdin); print(d["choices"][0]["message"]["content"][:300] if "choices" in d else "ERROR "+str(d))' > /root/bfcl_sanity_$m.txt
done
grep -q "^ERROR" /root/bfcl_sanity_*.txt && fail "sanity request errored"
cmp -s /root/bfcl_sanity_qwen3.8-27b.txt /root/bfcl_sanity_$M.txt && fail "LoRA not applied"
log "BFCL sanity ok: $M changes the output"
(MODEL=$M-FC setsid nohup /root/pod/partial_score.sh > /root/partial_score.out 2>&1 < /dev/null &)
MODEL=$M-FC /root/pod/run_multiturn.sh

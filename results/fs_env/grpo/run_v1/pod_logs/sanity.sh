#!/bin/bash
# fix adapter names, hot-load as sft-v1, compare temp-0 outputs base vs LoRA (must differ, and both must be real outputs)
cd /root
rm -rf /root/sft/run_v1/final_adapter_vllm
/root/train-venv/bin/python -m fs_env.adapter_fix /root/sft/run_v1/final_adapter /root/sft/run_v1/final_adapter_vllm || exit 1
curl -s -X POST localhost:8000/v1/unload_lora_adapter -H 'Content-Type: application/json' -d '{"lora_name":"sft-v1"}' >/dev/null
r=$(curl -s -X POST localhost:8000/v1/load_lora_adapter -H 'Content-Type: application/json' -d '{"lora_name":"sft-v1","lora_path":"/root/sft/run_v1/final_adapter_vllm"}'); echo "load: $r"
echo "$r" | grep -q '"error"' && { echo "SANITY_FAIL: adapter did not load"; exit 1; }
for m in qwen3.8-27b sft-v1; do
  curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d "{\"model\":\"$m\",\"temperature\":0,\"max_tokens\":200,\"messages\":[{\"role\":\"user\",\"content\":\"Create a file called notes.txt in my home folder for my meeting notes.\"}],\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
   | /root/train-venv/bin/python -c 'import sys,json; d=json.load(sys.stdin); print(repr(d["choices"][0]["message"]["content"][:300]) if "choices" in d else "ERROR "+str(d))' > /root/sanity_$m.txt
  echo "== $m"; cat /root/sanity_$m.txt
done
grep -q '^ERROR' /root/sanity_*.txt && { echo "SANITY_FAIL: a request errored"; exit 1; }
cmp -s /root/sanity_qwen3.8-27b.txt /root/sanity_sft-v1.txt && echo "SANITY_FAIL: identical (adapter not applied)" || echo "SANITY_OK: outputs differ"

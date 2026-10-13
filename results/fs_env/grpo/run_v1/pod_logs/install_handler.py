"""Register Qwen3.8-27B in a BFCL checkout (idempotent).

Copies qwen38_vllm.py into bfcl_eval/model_handler/api_inference/ and adds a "qwen3.8-27b-FC" entry to
bfcl_eval/constants/model_config.py, modeled on the official "qwen3-32b-FC" entry.
Extra NAME arguments also register "NAME-FC" for a LoRA that vLLM serves as NAME (the handler sends the registry
name minus "-FC" as the model), e.g. a GRPO checkpoint loaded with --lora-modules qwen3.8-27b-grpo30=<path>.
    /root/bfcl-venv/bin/python install_handler.py /root/gorilla/berkeley-function-call-leaderboard [NAME ...]
"""

import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/gorilla/berkeley-function-call-leaderboard")
here = Path(__file__).resolve().parent
shutil.copy(here / "qwen38_vllm.py", root / "bfcl_eval/model_handler/api_inference/qwen38_vllm.py")

cfg = root / "bfcl_eval/constants/model_config.py"
s = cfg.read_text()

imp = "from bfcl_eval.model_handler.api_inference.qwen38_vllm import Qwen38VLLMHandler\n"
anchor = "from bfcl_eval.model_handler.api_inference.qwen import ("
assert anchor in s, "BFCL layout changed: Qwen import not found"
if imp not in s:
    s = s.replace(anchor, imp + anchor, 1)

# display names without commas: BFCL's summary CSVs are comma-separated and shift columns otherwise
TEMPLATE = '''    "{name}-FC": ModelConfig(
        model_name="{name}",
        display_name="{display}",
        url="https://huggingface.co/Qwen/Qwen3.8-27B",
        org="Qwen",
        license="apache-2.0",
        model_handler=Qwen38VLLMHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=True,
    ),
'''
anchor = '    "qwen3-32b-FC": ModelConfig('
assert anchor in s, "BFCL layout changed: qwen3-32b-FC entry not found"
names = [("qwen3.8-27b", "Qwen3.8-27B (FC thinking local vLLM)")] + [(n, f"{n} (LoRA FC thinking local vLLM)") for n in sys.argv[2:]]
for name, display in names:
    if f'"{name}-FC"' in s:
        print(f"already registered: {name}-FC")
        continue
    s = s.replace(anchor, TEMPLATE.format(name=name, display=display) + anchor, 1)
    print(f"registered {name}-FC")
cfg.write_text(s)

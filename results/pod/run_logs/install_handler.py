"""Register Qwen3.8-27B in a BFCL checkout (idempotent).

Copies qwen38_vllm.py into bfcl_eval/model_handler/api_inference/ and adds a "qwen3.8-27b-FC" entry to
bfcl_eval/constants/model_config.py, modeled on the official "qwen3-32b-FC" entry.
    /root/bfcl-venv/bin/python install_handler.py /root/gorilla/berkeley-function-call-leaderboard
"""

import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/gorilla/berkeley-function-call-leaderboard")
here = Path(__file__).resolve().parent
shutil.copy(here / "qwen38_vllm.py", root / "bfcl_eval/model_handler/api_inference/qwen38_vllm.py")

cfg = root / "bfcl_eval/constants/model_config.py"
s = cfg.read_text()
if '"qwen3.8-27b-FC"' in s:
    print("already registered")
    sys.exit(0)

imp = "from bfcl_eval.model_handler.api_inference.qwen38_vllm import Qwen38VLLMHandler\n"
anchor = "from bfcl_eval.model_handler.api_inference.qwen import ("
assert anchor in s, "BFCL layout changed: Qwen import not found"
s = s.replace(anchor, imp + anchor, 1)

entry = '''    "qwen3.8-27b-FC": ModelConfig(
        model_name="qwen3.8-27b",
        display_name="Qwen3.8-27B (FC, thinking, local vLLM)",
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
s = s.replace(anchor, entry + anchor, 1)
cfg.write_text(s)
print("registered qwen3.8-27b-FC")

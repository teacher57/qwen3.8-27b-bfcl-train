"""Make a LoRA trained on the text-only model class usable by vLLM, which serves Qwen3.8 as the multimodal class.

Training loads the checkpoint with AutoModelForCausalLM (text-only), so PEFT saves tensors as
  base_model.model.model.layers.N...
vLLM serves Qwen3_5ForConditionalGeneration, whose language model lives under model.language_model, so it expects
  base_model.model.model.language_model.layers.N...
With the text-only names vLLM loads the adapter but matches no module and silently serves the base model (found
2026-10-12: base and "sft" gave identical outputs at temperature 0). This writes a copy with the names vLLM expects.

  python -m fs_env.adapter_fix /root/sft/run_v1/final_adapter /root/sft/run_v1/final_adapter_vllm
"""

import json
import shutil
import sys
from pathlib import Path

FROM, TO = "base_model.model.model.layers.", "base_model.model.model.language_model.layers."


def fix(src, dst) -> int:
    from safetensors.torch import load_file, save_file
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name != "adapter_model.safetensors" and f.is_file():
            shutil.copy2(f, dst / f.name)
    tensors = load_file(str(src / "adapter_model.safetensors"))
    renamed = {(TO + k[len(FROM):] if k.startswith(FROM) else k): v for k, v in tensors.items()}
    n = sum(1 for k in tensors if k.startswith(FROM))
    save_file(renamed, str(dst / "adapter_model.safetensors"), metadata={"format": "pt"})
    return n


if __name__ == "__main__":
    n = fix(sys.argv[1], sys.argv[2])
    print(f"renamed {n} tensors -> {sys.argv[2]}")

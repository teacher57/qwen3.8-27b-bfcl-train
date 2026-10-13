# Invalid runs

`q38_sft_v1_A3H10_t1_x4` (2026-10-12, 22:30–23:20) was meant to evaluate the SFT warm start v1, but vLLM loaded the LoRA
with tensor names it couldn't match (`model.layers.*` from the text-only training class vs `model.language_model.layers.*` in
the served multimodal class) and silently served the **base model**: base and "sft" gave identical outputs at temperature 0.
So these 360 conversations are a second base-model run, not SFT. Fixed by `fs_env/adapter_fix.py` (renames the tensors);
the real SFT eval is `q38_sft_v1fix_A3H10_t1_x4`.

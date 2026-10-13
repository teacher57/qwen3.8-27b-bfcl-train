"""SFT warm start: QLoRA on Qwen3.8-27B with the fs_env turn examples, over-reaction probes every N steps.

Data: JSONL from sft_export.py / sft_e_refusals.py ({"tools", "messages", "train_from", ...}). Each example is rendered with
the chat template and ITS OWN tool list; the loss covers only the assistant messages from `train_from` on (the turn being
taught), so earlier turns and tool results are context only.

Every --probe-every steps (and at step 0) sft_probes.ProbeCallback scores the held-out probes, saves the LoRA, and stops
training if the model starts refusing/asking where it should act. Everything is logged as JSON for the notebook:
  <out>/train_log.jsonl   one line per logged step (loss, lr, grad norm, tokens)
  <out>/probes.jsonl      one line per probe check
  <out>/status.json       stage, step, ETA, GPU memory, data stats (rewritten as training goes)

  /root/train-venv/bin/python -m fs_env.sft_train --data /root/sft/train_v1.jsonl --out /root/sft/run_v1
  add --smoke to run 3 steps with 10 probes (memory/speed check)
"""

import argparse
import json
import time
from pathlib import Path

import torch

from .sft_probes import ProbeCallback, build_probes


def norm_messages(msgs):
    """tool_call arguments as dicts (vLLM gives JSON strings; the template wants a mapping)."""
    out = []
    for m in msgs:
        m = dict(m)
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                f = dict(c["function"])
                if isinstance(f.get("arguments"), str):
                    try: f["arguments"] = json.loads(f["arguments"]) if f["arguments"].strip() else {}
                    except ValueError: pass
                calls.append({**c, "function": f})
            m["tool_calls"] = calls
        out.append(m)
    return out


def render_text(tok, msgs, tools, gen=False):
    return tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=gen, tokenize=False, enable_thinking=True)


def encode(tok, ex, max_len):
    """input_ids + labels (-100 outside the taught assistant messages), or None with the reason it was dropped.

    The taught spans are found on the rendered TEXT (character offsets), then every token overlapping them is labelled:
    token boundaries can differ between a cut-off prompt and the full conversation (e.g. "<think>\n" + "\n</think>" merges
    into a "\n\n" token), so comparing token prefixes would wrongly reject good examples."""
    msgs, tools, start = norm_messages(ex["messages"]), ex["tools"], ex["train_from"]
    text = render_text(tok, msgs, tools)
    spans = []
    for i in range(start, len(msgs)):
        if msgs[i]["role"] != "assistant":
            continue
        a = render_text(tok, msgs[:i], tools, gen=True)      # ... "<|im_start|>assistant\n<think>\n"
        b = render_text(tok, msgs[: i + 1], tools)            # ... including this assistant message
        if not (text.startswith(a) and text.startswith(b)):
            return None, "template prefix mismatch"
        spans.append((len(a), len(b)))
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    if len(ids) > max_len:
        return None, "too long"
    labels = [t if any(e > a and s_ < b for a, b in spans) else -100 for t, (s_, e) in zip(ids, offs)]
    if all(x == -100 for x in labels):
        return None, "nothing to train"
    return {"input_ids": ids, "labels": labels}, None


def load_model(path):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, BitsAndBytesConfig
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                           bnb_4bit_use_double_quant=True)
    last = None
    for cls in (AutoModelForCausalLM, AutoModelForImageTextToText):   # Qwen3.8 ships as a vision-language model class
        try:
            return cls.from_pretrained(path, quantization_config=q, dtype=torch.bfloat16, device_map={"": 0}), cls.__name__
        except (ValueError, KeyError, TypeError) as e:
            last = e
    raise RuntimeError(f"could not load {path}: {last}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="/dev/shm/q38")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--probe-every", type=int, default=10)
    ap.add_argument("--max-act-drop", type=float, default=0.10, help="stop when the probes' act rate falls this far below step 0")
    ap.add_argument("--smoke", action="store_true", help="3 steps, 10 probes, check every 2 steps")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    status_path, log_path = out / "status.json", out / "train_log.jsonl"
    t0 = time.time()
    status = {"stage": "loading", "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "args": vars(args)}
    def write_status(**kw):
        status.update(kw, elapsed_s=round(time.time() - t0), updated=time.strftime("%Y-%m-%dT%H:%M:%S"))
        if torch.cuda.is_available():
            status["gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
        tmp = status_path.with_suffix(".tmp"); tmp.write_text(json.dumps(status, indent=1)); tmp.replace(status_path)
    write_status()

    tok = AutoTokenizer.from_pretrained(args.model)
    rows = [json.loads(l) for l in open(args.data)]
    data, dropped = [], {}
    for ex in rows:
        enc, why = encode(tok, ex, args.max_len)
        if enc: data.append(enc)
        else: dropped[why] = dropped.get(why, 0) + 1
    n_tok = sum(len(d["input_ids"]) for d in data); n_lab = sum(sum(x != -100 for x in d["labels"]) for d in data)
    write_status(stage="loading model", data={"examples": len(data), "dropped": dropped, "tokens": n_tok, "trained_tokens": n_lab})
    print(f"{len(data)} examples ({dropped or 'none dropped'}), {n_tok:,} tokens, {n_lab:,} trained", flush=True)

    model, cls = load_model(args.model)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, target_modules="all-linear",
                      exclude_modules=r".*(visual|vision|merger|lm_head).*", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    write_status(stage="probing (step 0)", model_class=cls, trainable_params=trainable)
    print(f"loaded {cls}; LoRA trainable params {trainable:,}", flush=True)

    probes = build_probes()
    if args.smoke: probes = probes[::10]
    steps_per_epoch = max(1, len(data) // args.grad_accum)
    max_steps = 3 if args.smoke else int(steps_per_epoch * args.epochs)

    class Log(TrainerCallback):
        def on_log(self, a, state, control, logs=None, **kw):
            if not logs or "loss" not in logs: return
            with log_path.open("a") as f:
                f.write(json.dumps({"step": state.global_step, **{k: logs[k] for k in ("loss", "learning_rate", "grad_norm", "epoch") if k in logs},
                                    "t": round(time.time() - t0)}) + "\n")
            done = state.global_step / max_steps
            el = time.time() - t_train
            write_status(stage="training", step=state.global_step, max_steps=max_steps,
                         eta_s=round(el / done * (1 - done)) if done else None, last_loss=logs["loss"])

    def collate(batch):
        L = max(len(b["input_ids"]) for b in batch); pad = tok.pad_token_id or 0
        ids = torch.tensor([b["input_ids"] + [pad] * (L - len(b["input_ids"])) for b in batch])
        lab = torch.tensor([b["labels"] + [-100] * (L - len(b["labels"])) for b in batch])
        att = torch.tensor([[1] * len(b["input_ids"]) + [0] * (L - len(b["input_ids"])) for b in batch])
        return {"input_ids": ids, "labels": lab, "attention_mask": att}

    targs = TrainingArguments(output_dir=str(out / "checkpoints"), per_device_train_batch_size=1, gradient_accumulation_steps=args.grad_accum,
                              learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=max(1, round(0.05 * max_steps)), max_steps=max_steps, bf16=True,
                              logging_steps=1, save_strategy="no", save_only_model=True, report_to=[], remove_unused_columns=False,
                              gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}, disable_tqdm=True)
    probe_cb = ProbeCallback(tok, probes, every=2 if args.smoke else args.probe_every, out=str(out / "probes.jsonl"),
                             max_act_drop=args.max_act_drop)
    trainer = Trainer(model=model, args=targs, train_dataset=data, data_collator=collate, callbacks=[Log(), probe_cb])
    t_train = time.time()
    write_status(stage="training", step=0, max_steps=max_steps)
    trainer.train()
    model.save_pretrained(str(out / "final_adapter"))
    last = [json.loads(l) for l in open(out / "probes.jsonl")][-1]
    write_status(stage="stopped: over-reacting" if last.get("stopped") else "finished", step=trainer.state.global_step, eta_s=0)
    print("done:", status["stage"], flush=True)


if __name__ == "__main__":
    main()

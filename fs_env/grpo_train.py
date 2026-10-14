"""GRPO on the fs_env tasks, one GPU: vLLM generates the rollouts, QLoRA updates the LoRA, vLLM sleeps during updates.

Each step:
  1. rollouts: --tasks-per-step TRAIN tasks (causes from --causes) x a group of --group conversations, temperature 1, with the
     current LoRA served by vLLM (the real fs_env runner, so tool calls hit the in-memory file system)
  2. reward per TURN: 1 if the turn passed (files match the target, reply check if any), else 0. The advantage is computed
     within the group for the same task and turn: A = (r - mean) / (std + eps). Groups where every rollout got the same
     result carry no signal and are dropped (DAPO's dynamic sampling).
  3. update: on-policy policy gradient on the LoRA, loss = -A * mean log-prob of that turn's assistant tokens (thinking,
     tool calls, reply), averaged over the batch; one optimizer step per GRPO step, gradient clipping. No KL term by
     default (--kl 0): the over-reaction probes and a small learning rate keep it close instead.
  4. the new LoRA is saved and hot-loaded into vLLM; every --probe-every steps the held-out probes run (sft_probes) and
     training stops if the act rate falls more than --max-act-drop below step 0.

Logs (JSON, for the notebook): <out>/grpo_log.jsonl (one line per step), <out>/probes.jsonl, <out>/status.json,
<out>/rollouts/step_<n>.jsonl (every rollout, full conversation). Adapters: <out>/adapters/step_<n>.

vLLM must run with --enable-lora --enable-sleep-mode and VLLM_ALLOW_RUNTIME_LORA_UPDATING=True, VLLM_SERVER_DEV_MODE=1
(see pod/serve_grpo.sh). Run from /root with the training venv:
  /root/train-venv/bin/python -m fs_env.grpo_train --init-adapter /root/sft/run_v1/final_adapter --out /root/grpo/run_v1
"""

import argparse
import json
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from .tasks import TASK_DIR, USER_NAMES, instantiate, load_templates, source_path, task_sources


# ---------------- vLLM control ----------------

def vllm(path, body=None, base="http://127.0.0.1:8000", timeout=600):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else b"",
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()


def load_adapter(name, path, previous=None):
    """Hot-load a LoRA into vLLM, after renaming its tensors to the served model's layout (see adapter_fix.py: without
    that, vLLM loads it but applies it to no layer and silently serves the base model)."""
    from .adapter_fix import fix
    vpath = Path(str(path) + "_vllm")
    fix(path, vpath)
    if previous:
        try: vllm("/v1/unload_lora_adapter", {"lora_name": previous})
        except Exception: pass
    vllm("/v1/load_lora_adapter", {"lora_name": name, "lora_path": str(vpath)})


# ---------------- rollouts and advantages ----------------

def rollouts(tasks, group, model_name, args):
    """group x each task, in parallel through the fs_env runner. Returns run.py result dicts with "group_task" set."""
    from . import run as runner
    rargs = SimpleNamespace(base_url="http://127.0.0.1:8000/v1", model=model_name, api_key="EMPTY", temperature=args.temperature,
                            max_tokens=args.max_tokens, max_steps=args.max_steps, timeout=args.timeout, no_thinking=False, system="")
    jobs = [(t, k) for t in tasks for k in range(group)]
    with ThreadPoolExecutor(args.threads) as pool:
        out = list(pool.map(lambda j: runner.run_task(rargs, *j), jobs))
    for r, (t, _) in zip(out, jobs):
        r["group_task"] = t["id"]
    return out


def turn_rewards(r):
    return [1.0 if t["passed"] else 0.0 for t in r["turns"]]


def advantages(results, eps=1e-4):
    """{(rollout index, turn index): advantage} for (task, turn) groups that have a spread; plus stats."""
    groups = {}
    for i, r in enumerate(results):
        if r.get("error"):
            continue
        for k, rew in enumerate(turn_rewards(r)):
            groups.setdefault((r["group_task"], k), []).append((i, rew))
    adv, mixed, flat = {}, 0, 0
    for key, items in groups.items():
        rs = [x for _, x in items]
        mean = sum(rs) / len(rs); std = (sum((x - mean) ** 2 for x in rs) / len(rs)) ** 0.5
        if std < 1e-8:
            flat += 1; continue
        mixed += 1
        for i, x in items:
            adv[(i, key[1])] = (x - mean) / (std + eps)
    return adv, {"turn_groups_mixed": mixed, "turn_groups_flat": flat}


def turn_items(results, adv, tasks_by_id, tok, max_len):
    """Training items: one per (rollout, turn) with a non-zero advantage; reuses the SFT turn split and masking."""
    from .sft_export import turn_examples
    from .sft_train import encode
    items, dropped = [], 0
    for i, r in enumerate(results):
        exs = turn_examples(r, tasks_by_id[r["group_task"]])
        for k, ex in enumerate(exs):
            a = adv.get((i, k))
            if a is None:
                continue
            enc, why = encode(tok, ex, max_len)
            if enc is None:
                dropped += 1; continue
            items.append({**enc, "adv": a, "task": r["group_task"], "turn": k + 1})
    return items, dropped


def pick_tasks(pool, flat_count, n, rng):
    """Balanced draw: one task from each cause in turn (A3, D, E, A3, ...), skipping tasks whose group was flat twice."""
    live = [t for t in pool if flat_count.get(t["id"], 0) < 2] or pool
    by = {}
    for t in live: by.setdefault(t["cause"], []).append(t)
    causes = sorted(by); picks = []
    order = causes * (n // len(causes)) + rng.sample(causes, n % len(causes))
    for c in order:
        left = [t for t in by[c] if t not in picks]
        if left: picks.append(rng.choice(left))
    return picks


# ---------------- policy update ----------------

def policy_loss(model, item, device):
    """-advantage x mean log-prob of the labelled tokens (one sequence)."""
    import torch
    ids = torch.tensor([item["input_ids"]], device=device)
    labels = torch.tensor([item["labels"]], device=device)
    logits = model(input_ids=ids).logits[:, :-1].float()
    tgt = labels[:, 1:]
    mask = tgt != -100
    lp = torch.log_softmax(logits, -1).gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    mean_lp = (lp * mask).sum() / mask.sum().clamp(min=1)
    return -item["adv"] * mean_lp, mean_lp.detach().item()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="/dev/shm/q38")
    ap.add_argument("--init-adapter", help="start from this LoRA (e.g. the SFT warm start); default: a fresh LoRA")
    ap.add_argument("--out", required=True)
    ap.add_argument("--causes", default="A3,D,E")
    ap.add_argument("--task-set", default="tasks", choices=task_sources(),
                    help="tasks/ (only its TRAIN split is used) or a JSON task set in fs_env/task_sets/ (all of it: it is the training set)")
    ap.add_argument("--replay-causes", default="", help="also draw these causes from tasks/'s train split every step, e.g. A3,D,E")
    ap.add_argument("--replay-per-step", type=int, default=0, help="how many replay tasks per step (on top of --tasks-per-step)")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--tasks-per-step", type=int, default=4)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--max-steps", type=int, default=10, help="model calls per turn in rollouts (eval uses 20)")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--probe-every", type=int, default=5)
    ap.add_argument("--max-act-drop", type=float, default=0.10)
    ap.add_argument("--no-sleep", action="store_true", help="don't put vLLM to sleep during updates")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="1 step, 2 tasks x group 4, probes on 10 probes")
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoTokenizer
    from .sft_probes import build_probes, score_probes, summarize
    from .sft_train import load_model

    if args.dry_run:
        args.steps, args.tasks_per_step, args.group, args.probe_every = 1, 2, 4, 1
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "rollouts").mkdir(exist_ok=True); (out / "adapters").mkdir(exist_ok=True)
    t0 = time.time()
    status = {"stage": "loading", "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "args": vars(args)}
    def write_status(**kw):
        status.update(kw, elapsed_s=round(time.time() - t0), updated=time.strftime("%Y-%m-%dT%H:%M:%S"))
        if torch.cuda.is_available(): status["gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
        tmp = out / "status.tmp"; tmp.write_text(json.dumps(status, indent=1)); tmp.replace(out / "status.json")
    write_status()

    rng = random.Random(args.seed)
    split = json.loads((TASK_DIR.parent / "split.json").read_text())
    causes = set(args.causes.split(","))
    if args.task_set == "tasks":
        pool = [t for t in load_templates() if t["id"] in set(split["train"]) and t["cause"] in causes]   # templates
    else:
        pool = [t for t in load_templates(source_path(args.task_set)) if t["cause"] in causes]
    replay = set(args.replay_causes.split(",")) - {""}
    replay_pool = [t for t in load_templates() if t["id"] in set(split["train"]) and t["cause"] in replay]
    if args.replay_per_step and not replay_pool:
        raise SystemExit(f"--replay-per-step {args.replay_per_step} but no train tasks for causes {sorted(replay)}")
    tasks_by_id = {}   # the instance used at each draw (a fresh user name, folder names and clutter per draw)
    flat_count = {}    # task id -> times its whole group came out the same (all pass or all fail)

    tok = AutoTokenizer.from_pretrained(args.model)
    model, cls = load_model(args.model)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear",
                                                 exclude_modules=r".*(visual|vision|merger|lm_head).*", task_type="CAUSAL_LM"))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    device = next(model.parameters()).device
    probes = build_probes()
    if args.dry_run: probes = probes[::10]

    def probe(step):
        model.eval()
        s = summarize(score_probes(model, tok, probes)); s["step"] = step
        model.train()
        return s

    # the policy vLLM serves: start from the init adapter (or the base model when there is none)
    name, prev = ("policy-0", None)
    if args.init_adapter: load_adapter(name, args.init_adapter)
    else: name = "qwen3.8-27b"
    write_status(stage="probing (step 0)", pool=len(pool), model_class=cls)
    base = probe(0); base["act_drop"] = 0.0; base["stopped"] = False
    with (out / "probes.jsonl").open("a") as f: f.write(json.dumps(base) + "\n")

    for step in range(1, args.steps + 1):
        st = time.time()
        if not args.no_sleep:
            torch.cuda.empty_cache()
            try: vllm("/wake_up")
            except Exception: pass
        write_status(stage="rollouts", step=step, max_steps=args.steps)
        picks = pick_tasks(pool, flat_count, args.tasks_per_step, rng)
        if args.replay_per_step:   # old causes mixed in, so what earlier training taught isn't forgotten
            picks += pick_tasks(replay_pool, flat_count, args.replay_per_step, rng)
        # one fresh instance per task per step: the same for all of its group (a fair comparison), new on the next draw
        batch = [instantiate(t, rng.choice(USER_NAMES), rng) for t in picks]
        tasks_by_id.update({t["id"]: t for t in batch})
        res = rollouts(batch, args.group, name, args)
        with (out / "rollouts" / f"step_{step}.jsonl").open("w") as f:
            for r in res: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        t_roll = time.time() - st
        adv, gstats = advantages(res)
        for t in batch:   # a task whose group is flat twice teaches nothing: it leaves the pool
            rews = [sum(turn_rewards(r)) for r in res if r["group_task"] == t["id"] and not r.get("error")]
            if rews and len(set(rews)) == 1:
                flat_count[t["id"]] = flat_count.get(t["id"], 0) + 1
        gstats["pool_left"] = sum(1 for t in pool if flat_count.get(t["id"], 0) < 2)
        items, dropped = turn_items(res, adv, tasks_by_id, tok, args.max_len)

        if not args.no_sleep:
            vllm("/sleep?level=1")
        write_status(stage="updating", step=step)
        loss_sum, lps = 0.0, []
        model.train(); opt.zero_grad()
        for it in items:
            loss, lp = policy_loss(model, it, device)
            (loss / max(1, len(items))).backward()
            loss_sum += loss.item(); lps.append(lp)
        gnorm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm).item() if items else 0.0
        if items: opt.step()
        opt.zero_grad(); torch.cuda.empty_cache()

        path = out / "adapters" / f"step_{step}"
        model.save_pretrained(str(path))
        if not args.no_sleep:
            vllm("/wake_up")
        prev, name = name, f"policy-{step}"
        load_adapter(name, path, previous=prev if prev.startswith("policy-") else None)

        by_cause = {}
        for r in res:
            by_cause.setdefault(r["cause"], []).append(r["passed"])
        line = {"step": step, "tasks": [t["id"] for t in batch], "conversations": len(res),
                "reward_conv": sum(r["passed"] for r in res) / len(res),
                "reward_turn": sum(sum(turn_rewards(r)) / max(1, len(r["turns"])) for r in res) / len(res),
                "pass_by_cause": {c: sum(v) / len(v) for c, v in by_cause.items()},
                "errors": sum(1 for r in res if r.get("error")), **gstats,
                "train_items": len(items), "dropped": dropped, "loss": loss_sum / max(1, len(items)),
                "mean_logprob": sum(lps) / len(lps) if lps else None, "grad_norm": gnorm,
                "rollout_s": round(t_roll), "step_s": round(time.time() - st)}
        with (out / "grpo_log.jsonl").open("a") as f: f.write(json.dumps(line) + "\n")
        print(f"step {step}: reward {line['reward_turn']:.2f}, mixed groups {gstats['turn_groups_mixed']}, items {len(items)}, {line['step_s']}s", flush=True)

        if step % args.probe_every == 0 or step == args.steps:
            if not args.no_sleep: vllm("/sleep?level=1")
            s = probe(step)
            if not args.no_sleep: vllm("/wake_up")
            s["act_drop"] = base["act_rate"] - s["act_rate"]
            s["stopped"] = s["act_drop"] > args.max_act_drop
            with (out / "probes.jsonl").open("a") as f: f.write(json.dumps(s) + "\n")
            if s["stopped"]:
                write_status(stage="stopped: over-reacting", step=step); print("over-reacting: stopping", flush=True); return
        elapsed = time.time() - t0
        write_status(stage="rollouts", step=step, max_steps=args.steps, eta_s=round(elapsed / step * (args.steps - step)))
    write_status(stage="finished", step=args.steps, eta_s=0)
    print("done", flush=True)


if __name__ == "__main__":
    main()

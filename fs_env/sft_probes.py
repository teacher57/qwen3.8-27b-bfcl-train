"""Over-reaction probes for SFT/GRPO: cheap checks of the model's FIRST MOVE, run every N training steps.

Every probe is a held-out situation (EVAL split only) where the right first move is known:
  act     the tool is there and nothing is missing: the reply should start with a tool call
          (first turns of act/answer tasks of every cause, D after the user supplied the missing info,
           E after the tool came back)
  refuse  E turn 1, the needed tool is withheld: no tool call, and a refusal rather than a question
  ask     D turn 1, information is missing: no tool call, and a question rather than a refusal

No generation: one forward pass per probe gives P(first token = <tool_call>) with the chat template's thinking turned
off (`<think>\n\n</think>\n\n` already in the prompt), so the decision is the next token. For refuse/ask probes, two
fixed replies (a refusal and a question) are also scored to see which kind of answer the model leans to. This is a
proxy for the real thinking-on behavior; confirm with real generations (run.py) on the saved checkpoints.

Over-reaction = the `act` rate falling: the model starts refusing or asking where it should just act.
"""

import json
import math
from pathlib import Path

from .tasks import TASK_DIR, instantiate, load_templates
from .tools import tools_for

REFUSAL = "I can't do that right now: the tool I'd need isn't available to me. If it gets enabled, I'll do it right away."
QUESTION = "Before I do that, could you tell me exactly which one you mean and what name or text to use?"


def build_probes() -> list:
    """Probes from the EVAL split, with a fixed user name and no clutter (only the messages matter here)."""
    eval_ids = set(json.loads((TASK_DIR.parent / "split.json").read_text())["eval"])
    probes = []
    for t in sorted(load_templates(), key=lambda t: t["id"]):
        if t["id"] not in eval_ids:
            continue
        task = instantiate(t, "alex")
        turns = task["turns"]
        first = turns[0]
        group = {"act": "act", "answer": "act", "refuse": "refuse", "ask": "ask"}[first["expect"]]
        probes.append({"id": task["id"], "cause": task["cause"], "group": group, "where": "turn 1",
                       "messages": [{"role": "user", "content": first["user"]}], "tools": tools_for(first["withheld_tools"])})
        # after a correct ask/refuse, the user supplies what was missing: now the model must act
        if first["expect"] in ("ask", "refuse") and len(turns) > 1 and turns[1]["expect"] == "act":
            reply = (first.get("answer_check") or {}).get("example") or (REFUSAL if first["expect"] == "refuse" else QUESTION)
            probes.append({"id": task["id"], "cause": task["cause"], "group": "act",
                           "where": "turn 2 (tool back)" if first["expect"] == "refuse" else "turn 2 (info given)",
                           "messages": [{"role": "user", "content": first["user"]}, {"role": "assistant", "content": reply},
                                        {"role": "user", "content": turns[1]["user"]}],
                           "tools": tools_for(turns[1]["withheld_tools"])})
    return probes


def _prompt_ids(tok, probe) -> list:
    return tok.apply_chat_template(probe["messages"], tools=probe["tools"], add_generation_prompt=True,
                                   tokenize=True, enable_thinking=False, return_dict=False)


def score_probes(model, tok, probes) -> list:
    """Per probe: p_tool = P(next token is <tool_call>); for refuse/ask probes also the mean log-prob of REFUSAL and QUESTION."""
    import torch
    tool_id = tok.convert_tokens_to_ids("<tool_call>")
    end = tok.encode("<|im_end|>", add_special_tokens=False)
    device = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for p in probes:
            ids = _prompt_ids(tok, p)
            if hasattr(ids, "input_ids"): ids = ids["input_ids"]
            logits = model(input_ids=torch.tensor([ids], device=device)).logits[0, -1].float()
            row = {"id": p["id"], "cause": p["cause"], "group": p["group"], "where": p["where"],
                   "p_tool": torch.softmax(logits, -1)[tool_id].item()}
            if p["group"] in ("refuse", "ask"):
                for name, text in (("lp_refusal", REFUSAL), ("lp_question", QUESTION)):
                    cand = tok.encode(text, add_special_tokens=False) + end
                    x = torch.tensor([ids + cand], device=device)
                    lp = torch.log_softmax(model(input_ids=x).logits[0, len(ids) - 1:-1].float(), -1)
                    row[name] = lp.gather(1, torch.tensor(cand, device=device)[:, None]).mean().item()
            out.append(row)
    return out


def summarize(rows: list) -> dict:
    g = lambda name: [r for r in rows if r["group"] == name]
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    act, ref, ask = g("act"), g("refuse"), g("ask")
    s = {
        "act_rate": mean([r["p_tool"] > 0.5 for r in act]),           # should be high; falling = over-reaction
        "act_p_tool": mean([r["p_tool"] for r in act]),
        "refuse_no_tool": mean([r["p_tool"] < 0.5 for r in ref]),     # target: rising
        "refuse_says_cant": mean([r["lp_refusal"] > r["lp_question"] for r in ref]),
        "ask_no_tool": mean([r["p_tool"] < 0.5 for r in ask]),        # target: rising
        "ask_asks": mean([r["lp_question"] > r["lp_refusal"] for r in ask]),
        "n": {"act": len(act), "refuse": len(ref), "ask": len(ask)},
    }
    by = {}
    for r in act:
        by.setdefault(r["where"] if r["where"] != "turn 1" else r["cause"], []).append(r["p_tool"] > 0.5)
    s["act_rate_by"] = {k: mean(v) for k, v in sorted(by.items())}
    return s


class ProbeCallback:
    """transformers TrainerCallback: probes at step 0 and every `every` steps; appends one JSON line per check to `out`
    (read by the notebook); saves a checkpoint at each check; stops training when the act rate falls more than
    `max_act_drop` below the step-0 baseline (over-reaction)."""

    def __new__(cls, *a, **k):   # subclass TrainerCallback lazily, so this module imports without transformers
        from transformers import TrainerCallback

        class _Probe(TrainerCallback):
            def __init__(self, tok, probes, every=25, out="results/fs_env/sft/probes.jsonl", max_act_drop=0.10):
                self.tok, self.probes, self.every, self.out, self.max_drop = tok, probes, every, Path(out), max_act_drop
                self.base = None
                self.out.parent.mkdir(parents=True, exist_ok=True)

            def _check(self, step, model, control):
                was_training = model.training
                model.eval()
                s = summarize(score_probes(model, self.tok, self.probes))
                if was_training: model.train()
                if self.base is None: self.base = s
                s["step"] = step
                s["act_drop"] = (self.base["act_rate"] - s["act_rate"]) if s["act_rate"] is not None else None
                s["stopped"] = bool(s["act_drop"] is not None and s["act_drop"] > self.max_drop)
                with self.out.open("a") as f:
                    f.write(json.dumps(s) + "\n")
                if s["stopped"]:
                    print(f"[probes] step {step}: act rate {s['act_rate']:.0%} is {s['act_drop']:.0%} below the start: over-reacting, stopping")
                    control.should_training_stop = True
                return s

            def on_train_begin(self, args, state, control, model=None, **kw):
                self._check(0, model, control)

            def on_step_end(self, args, state, control, model=None, **kw):
                if state.global_step % self.every == 0:
                    self._check(state.global_step, model, control)
                    control.should_save = True
                return control

        return _Probe(*a, **k)


if __name__ == "__main__":
    ps = build_probes()
    from collections import Counter
    print(len(ps), "probes:", dict(Counter(p["group"] for p in ps)), dict(Counter(p["where"] for p in ps)))

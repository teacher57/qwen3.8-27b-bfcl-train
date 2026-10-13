"""Turn rules-prompted runs into SFT warm-start data (self-distillation: the model's own passing conversations).

Input: run.py JSONLs generated on TRAIN tasks with --system-file fs_env/sft_rules_prompt.txt. Kept: conversations where
every turn passed, with no error or runner note, whose reasoning and replies never mention the prompt (the trained model
won't have it). The system prompt is dropped, so the model learns the behavior itself, not "follow the rules message".

Output: one JSON line per (conversation, turn), because tools can differ per turn (withheld ones), and the model must be
trained with the tool list it actually saw:
  {"id", "trial", "cause", "turn", "tools": [...], "messages": [...], "train_from": i}
messages = the conversation up to the end of that turn; only the assistant messages at index >= train_from are trained
(earlier ones are context, with their thinking removed as the chat template does at inference).

  python -m fs_env.sft_export results/fs_env/fs-results/sft_gen_*.jsonl --out results/fs_env/sft/warmstart.jsonl
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .tasks import TASK_DIR, load_templates
from .tools import tools_for

# reasoning or replies that point at the (now removed) system prompt would teach the model to cite rules it won't have
# Only references that make sense if the model saw OUR rules. Plain "system prompt" is normal (Qwen's template puts the tool
# list there) and "the instruction says" usually means the user's request, so neither counts.
PROMPT_LEAK = re.compile(r"\brules?\s*#?\d|\b(the|my|these|those|following|your) rules\b|\brules (say|tell|state|require)|"
                         r"according to (the|my) rules|system (prompt|message) (says|tells|instructs|requires|wants) (me )?(to|that|not)|"
                         r"\b(I was|I'm|I am) (told|instructed) to\b|never approximate|exactly the same result|change nothing until",
                         re.IGNORECASE)


def split_ids(part: str) -> set:
    return set(json.loads((TASK_DIR.parent / "split.json").read_text())[part])


def leaks(r: dict) -> bool:
    for m in r["messages"]:
        if m["role"] == "assistant":
            if PROMPT_LEAK.search(m.get("content") or "") or PROMPT_LEAK.search(m.get("reasoning_content") or ""):
                return True
    return False


def turn_examples(r: dict, task: dict) -> list:
    msgs = [m for m in r["messages"] if m["role"] != "system"]
    starts = [i for i, m in enumerate(msgs) if m["role"] == "user"]
    out = []
    for k, start in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(msgs)
        context = [{kk: v for kk, v in m.items() if kk != "reasoning_content"} if m["role"] == "assistant" else m
                   for m in msgs[:start]]
        out.append({"id": r["id"], "trial": r["trial"], "cause": r["cause"], "turn": k + 1,
                    "tools": tools_for(task["turns"][k]["withheld_tools"]),
                    "messages": context + msgs[start:end], "train_from": start + 1})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", help="run.py JSONL files")
    ap.add_argument("--out", default="results/fs_env/sft/warmstart.jsonl")
    ap.add_argument("--max-per-task", type=int, default=2, help="keep at most this many passing conversations per task")
    ap.add_argument("--allow-no-system", action="store_true", help="also accept runs without the rules prompt (testing only)")
    args = ap.parse_args()

    tasks = {t["id"]: t for t in load_templates()}
    train = split_ids("train")
    why = Counter(); kept = defaultdict(list)
    for path in args.runs:
        for line in open(path):
            r = json.loads(line)
            if r["id"] not in train: why["not a train task"] += 1; continue
            if not args.allow_no_system and (not r["messages"] or r["messages"][0]["role"] != "system"):
                why["no rules prompt"] += 1; continue
            if not r["passed"] or r.get("error") or any(t.get("note") for t in r["turns"]):
                why["failed / error / hit a limit"] += 1; continue
            if leaks(r): why["mentions the prompt"] += 1; continue
            if len(kept[r["id"]]) >= args.max_per_task: why["over the per-task cap"] += 1; continue
            kept[r["id"]].append(r)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as f:
        for tid in sorted(kept):
            for r in kept[tid]:
                for ex in turn_examples(r, tasks[tid]):
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n"); n += 1
    convs = sum(len(v) for v in kept.values())
    by_cause = Counter(tasks[t]["cause"] for t, v in kept.items() for _ in v)
    print(f"kept {convs} conversations from {len(kept)} tasks -> {n} turn examples in {out}")
    print("  per cause:", "  ".join(f"{c}:{k}" for c, k in sorted(by_cause.items())))
    print("  dropped:", dict(why))
    missing = sorted(t for t in train if t not in kept)
    print(f"  train tasks with no passing conversation: {len(missing)}" + (f" (e.g. {', '.join(missing[:5])})" if missing else ""))


if __name__ == "__main__":
    main()

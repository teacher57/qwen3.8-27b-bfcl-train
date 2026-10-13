"""Train/eval split of the fs_env tasks, written to fs_env/split.json (run once; re-running gives the same split).

eval  = the 30 tasks of the post-fix baseline run (results/fs_env/fs-results/q38_fs_fixed10_t1_x4.tasks.json, so every
        eval task already has a "before" number) + the 11 hand-written originals + for each other cause the original and
        4 random tasks of its 10 (a regression check: F and G guard against learning to over-ask or under-act)
train = everything else (SFT warm start data and GRPO prompts come only from here)

Run from bfcl-train/:  python -m fs_env.split
"""

import json
import random
from pathlib import Path

from .tasks import TASK_DIR, load_templates

PICKED = Path(__file__).parents[1] / "results/fs_env/fs-results/q38_fs_fixed10_t1_x4.tasks.json"
OUT = TASK_DIR.parent / "split.json"
FOCUS = ("A3", "D", "E")
SEED = 0


def is_original(tid: str) -> bool:
    return not tid.split("_")[1].isdigit()


def make_split() -> dict:
    tasks = sorted(load_templates(), key=lambda t: t["id"])
    picked = {t for ids in json.loads(PICKED.read_text())["tasks"].values() for t in ids}
    rng = random.Random(SEED)
    eval_ids = set(picked) | {t["id"] for t in tasks if is_original(t["id"])}
    for cause in sorted({t["cause"] for t in tasks} - set(FOCUS)):
        rest = sorted(t["id"] for t in tasks if t["cause"] == cause and not is_original(t["id"]))
        eval_ids |= set(rng.sample(rest, 4))
    train_ids = [t["id"] for t in tasks if t["id"] not in eval_ids]
    by = lambda ids: {c: sum(1 for t in tasks if t["id"] in ids and t["cause"] == c) for c in sorted({t["cause"] for t in tasks})}
    return {"about": __doc__.split("\n\n")[1].strip(), "seed": SEED,
            "counts": {"train": by(set(train_ids)), "eval": by(eval_ids)},
            "train": train_ids, "eval": sorted(eval_ids)}


if __name__ == "__main__":
    split = make_split()
    OUT.write_text(json.dumps(split, indent=1) + "\n")
    print(f"wrote {OUT}: {len(split['train'])} train, {len(split['eval'])} eval")
    for part in ("train", "eval"):
        print(f"  {part:5}", "  ".join(f"{c}:{n}" for c, n in split["counts"][part].items()))

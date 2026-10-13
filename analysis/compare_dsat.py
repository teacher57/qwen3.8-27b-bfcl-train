"""Compares two DSAT runs entry by entry (same BFCL entries, two models) and prints markdown tables for final_state_dsat.md:
all failures by cluster, wrong-final-state failures by root cause and category, and what happened to each base-model
failure (fixed by the new model, same cause, another cause) plus the new model's new failures.
    python3 analysis/compare_dsat.py [A_dir B_dir A_name B_name]   (default: analysis vs analysis/grpo20)"""
import collections, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
A_DIR, B_DIR, A, B = (sys.argv[1:5] if len(sys.argv) >= 5 else ("analysis", "analysis/grpo20", "base", "GRPO step 20"))
CATS = ["base", "miss_func", "miss_param", "long_context"]


def load(d):
    dsat = {r["id"]: r for r in json.load(open(HERE / d / "multi_turn_dsat.json"))}
    roots = json.load(open(HERE / d / "final_state_root_causes.json"))
    return dsat, roots


def short(rc):           # "D  Guessed missing ..." -> "D"; A4/H variants carry details after the code
    return rc.split()[0]


LABEL = {"A1": "A1 extra step: released / re-pressed the brake", "A2": "A2 extra step: started navigation", "A3": "A3 wrote invented content into a new file",
         "A4": "A4 other unrequested action", "B": "B rewrote the user's exact text", "C": "C small exact-value mistake",
         "D": "D guessed missing info instead of asking", "E": "E worked around a missing function", "F": "F asked when it should act",
         "G": "G did nothing / just answered", "H": "H other (skipped action, different approach...)"}
ORDER = ["A1", "A2", "A3", "A4", "B", "C", "D", "E", "F", "G", "H"]
(da, ra), (db, rb) = load(A_DIR), load(B_DIR)


def signed(x): return f"{x:+d}" if x else "0"


# 1. all failures by cluster
ca = collections.Counter(r["cluster"] for r in da.values()); cb = collections.Counter(r["cluster"] for r in db.values())
print(f"**All failed multi-turn entries, by first divergence** ({A}: {len(da)}, {B}: {len(db)} of 800)\n")
print(f"| What went wrong first | {A} | {B} | change |\n|---|---|---|---|")
for k in sorted(set(ca) | set(cb), key=lambda k: -(ca[k] + cb[k])):
    print(f"| {k} | {ca[k]} | {cb[k]} | {signed(cb[k] - ca[k])} |")
print(f"| **total** | **{len(da)}** | **{len(db)}** | **{signed(len(db) - len(da))}** |\n")

# 2. wrong-final-state root causes, per category
ta = collections.Counter((short(v["root_cause"]), v["category"]) for v in ra.values())
tb = collections.Counter((short(v["root_cause"]), v["category"]) for v in rb.values())
print(f"**Wrong-final-state failures by root cause** ({A} → {B}, per category)\n")
print("| Root cause | all | " + " | ".join(CATS) + " |\n|---|---|" + "---|" * len(CATS))
for k in ORDER:
    a = sum(ta[(k, c)] for c in CATS); b = sum(tb[(k, c)] for c in CATS)
    if a or b:
        print(f"| {LABEL[k]} | {a} → **{b}** | " + " | ".join(f"{ta[(k, c)]} → {tb[(k, c)]}" for c in CATS) + " |")
print(f"| **total** | {len(ra)} → **{len(rb)}** | " + " | ".join(f"{sum(ta[(k, c)] for k in ORDER)} → {sum(tb[(k, c)] for k in ORDER)}" for c in CATS) + " |\n")

# 3. entry by entry: what happened to each base wrong-state failure, and where the new model's failures came from
print(f"**Entry by entry** (the same 800 entries): what happened to each {A} wrong-final-state failure, and where {B}'s came from\n")
print(f"| Root cause | {A} failures | fixed (now pass) | same cause | other failure | {B} failures | new (passed before) | from another failure |\n|---|---|---|---|---|---|---|---|")
tot = collections.Counter()
for k in ORDER:
    ids_a = [i for i, v in ra.items() if short(v["root_cause"]) == k]
    ids_b = [i for i, v in rb.items() if short(v["root_cause"]) == k]
    if not ids_a and not ids_b: continue
    fixed = sum(1 for i in ids_a if i not in db); same = sum(1 for i in ids_a if i in rb and short(rb[i]["root_cause"]) == k)
    other = len(ids_a) - fixed - same
    new = sum(1 for i in ids_b if i not in da); moved = len(ids_b) - new - same
    tot.update(a=len(ids_a), fixed=fixed, same=same, other=other, b=len(ids_b), new=new, moved=moved)
    print(f"| {LABEL[k]} | {len(ids_a)} | {fixed} | {same} | {other} | {len(ids_b)} | {new} | {moved} |")
print(f"| **total** | **{tot['a']}** | **{tot['fixed']}** | **{tot['same']}** | **{tot['other']}** | **{tot['b']}** | **{tot['new']}** | **{tot['moved']}** |")

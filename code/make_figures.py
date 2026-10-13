"""Builds every figure in figures/ (used by README.md) from the result files in results/. Run from the repo root:
    python code/make_figures.py
Colors: orange = the reference model (base or SFT), blue = GRPO; checkpoints 10/20/30 = light -> dark blue (ordered);
fixed / broken = blue / red, also told apart by bar direction and labels."""
import json
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"; OUT.mkdir(exist_ok=True)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
REF, GRPO, BROKEN = "#eb6834", "#2a78d6", "#e34948"
RAMP = ["#86b6ef", "#2a78d6", "#104281"]          # steps 10, 20, 30
MUTED = "#b7d3f6"                                 # causes we did not train on
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "legend.frameon": False})


def jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def style(ax, grid_axis="y"):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8); ax.set_axisbelow(True)


def mcnemar(b, c):
    n, k = b + c, min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name, dpi=200); plt.close(fig); print("wrote figures/" + name)


# ---------- 1. where the base model fails BFCL multi-turn ----------
rc = json.load(open(ROOT / "analysis/final_state_root_causes.json"))
cnt = Counter(v["root_cause"].split()[0] for v in rc.values())
order = sorted(cnt, key=lambda k: cnt[k])
label = {"A1": "A1 extra step: released the brake", "A2": "A2 extra step: started navigation", "A3": "A3 wrote invented content into a new file",
         "A4": "A4 set a budget nobody asked for", "B": "B rewrote the user's exact text", "C": "C small exact-value mistakes",
         "D": "D guessed missing info instead of asking", "E": "E worked around a missing function", "F": "F asked when it should act",
         "G": "G did nothing when action was expected", "H": "H skipped a required action"}
trained = {"A3", "D", "E"}
fig, ax = plt.subplots(figsize=(8.6, 4.6))
ax.barh([label.get(k, k) for k in order], [cnt[k] for k in order], color=[GRPO if k in trained else MUTED for k in order], height=0.62)
for i, k in enumerate(order):
    ax.text(cnt[k] + 0.6, i, str(cnt[k]), va="center", color=INK, fontsize=9)
style(ax, "x"); ax.set_xlabel("multi-turn conversations ending in the wrong state (base model, 199 total)")
ax.set_title("Why the base model fails BFCL multi-turn", loc="left")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=GRPO, label="trained first in fs_env (A3, D, E)"), Patch(color=MUTED, label="other causes")], loc="lower right")
save(fig, "fig1_bfcl_failure_causes.png")

# ---------- 2. fs_env: base vs SFT warm start, first 10 tasks of each cause x 4 trials ----------
R = ROOT / "results/fs_env/fs-results"
cfg = json.load(open(ROOT / "results/fs_env/fs_notebook_config.json"))["part3"]
tasks = json.load(open(R / cfg["tasks"]))
tasks = tasks["first10"] if isinstance(tasks, dict) else tasks
base = {}
for run in cfg["baseline_runs"]:                       # per task, the LAST baseline run that has it
    rows = defaultdict(list)
    for r in jsonl(R / f"{run}.jsonl"): rows[r["id"]].append(r)
    base.update(rows)
sft = defaultdict(list)
for r in jsonl(R / f"{cfg['sft_run']}.jsonl"): sft[r["id"]].append(r)
causes = ["A3", "D", "E", "A4", "B", "C", "F", "G", "H"]
bp, sp, n = Counter(), Counter(), Counter()
for t in tasks:
    tid = t if isinstance(t, str) else t["id"]; c = tid.split("_")[0]
    if c not in causes or tid not in sft or tid not in base: continue
    bp[c] += sum(r["passed"] for r in base[tid][:4]); sp[c] += sum(r["passed"] for r in sft[tid][:4]); n[c] += min(4, len(sft[tid]))
print("fs_env base -> SFT:", {c: f"{bp[c]}->{sp[c]}/{n[c]}" for c in causes})
fig, ax = plt.subplots(figsize=(8.6, 3.9))
w = 0.38; xs = range(len(causes))
for off, data, col, name in ((-w / 2, bp, REF, "base model"), (w / 2, sp, GRPO, "SFT warm start v1")):
    ax.bar([x + off for x in xs], [data[c] / n[c] for c in causes], w - 0.03, color=col, label=name)
    for x, c in zip(xs, causes):
        ax.text(x + off, data[c] / n[c] + 0.02, f"{data[c]}", ha="center", fontsize=8, color=INK)
ax.set_xticks(list(xs), [f"{c}{'*' if c in trained else ''}" for c in causes]); ax.set_ylim(0, 1.12); ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
style(ax); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
ax.set_title("fs_env: passed of 40 per cause (10 tasks x 4 trials, temp 1)", loc="left")
ax.set_xlabel("* = causes trained on", loc="right")
save(fig, "fig2_fs_env_base_vs_sft.png")

# ---------- 3. GRPO training reward ----------
G = ROOT / "results/fs_env/grpo/run_v1"
log = jsonl(G / "grpo_log.jsonl")
steps = [l["step"] for l in log]; rew = [l["reward_turn"] for l in log]
roll = [sum(rew[max(0, i - 4):i + 1]) / len(rew[max(0, i - 4):i + 1]) for i in range(len(rew))]
fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.6, 5.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
a1.plot(steps, rew, color=MUTED, linewidth=2, marker="o", markersize=4, label="turns passed, each step (4 random tasks x 8)")
a1.plot(steps, roll, color=GRPO, linewidth=2.4, label="5-step mean")
a1.set_ylim(0, 1); a1.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}"); style(a1); a1.legend(loc="lower left")
a1.set_title("GRPO run_v1: training reward per step (flat and noisy: it can't pick a checkpoint)", loc="left")
a2.bar(steps, [l["turn_groups_mixed"] for l in log], color=GRPO, width=0.7)
a2.set_ylabel("mixed groups\n(of 12)"); a2.set_xlabel("GRPO step"); style(a2)
for s in (10, 20, 30): a1.axvline(s, color=GRID, linewidth=1, zorder=0)
save(fig, "fig3_grpo_training.png")

# ---------- 4. held-out fs_env: SFT vs checkpoints ----------
E = G / "eval"
runs = [("sft-v1", "SFT v1 (start)", REF), ("grpo-s10", "GRPO step 10", RAMP[0]), ("grpo-s20", "GRPO step 20", RAMP[1]), ("grpo-final", "GRPO step 30", RAMP[2])]
groups = [("A3", {"A3"}), ("D", {"D"}), ("E", {"E"}), ("other 8 causes", {"A1", "A2", "A4", "B", "C", "F", "G", "H"}), ("all 73 tasks", None)]
res = {k: jsonl(E / f"{k}.jsonl") for k, _, _ in runs}
fig, ax = plt.subplots(figsize=(8.6, 4.2))
w = 0.2
for j, (k, name, col) in enumerate(runs):
    for i, (g, sel) in enumerate(groups):
        rs = [r for r in res[k] if sel is None or r["cause"] in sel]; p = sum(r["passed"] for r in rs)
        x = i + (j - 1.5) * w
        ax.bar(x, p / len(rs), w - 0.025, color=col, label=name if i == 0 else None)
        ax.text(x, p / len(rs) + 0.015, str(p), ha="center", fontsize=7.5, color=INK)
ax.set_xticks(range(len(groups)), [f"{g}\n(of {len([r for r in res['sft-v1'] if sel is None or r['cause'] in sel])})" for g, sel in groups])
ax.set_ylim(0, 1.12); ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}"); style(ax); ax.legend(ncol=4, loc="upper left")
ax.set_title("Held-out fs_env (73 unseen tasks x 4 trials, same seed): passed conversations", loc="left")
save(fig, "fig4_heldout_checkpoints.png")

# ---------- 5 + 6. BFCL multi-turn: base vs GRPO step 20 ----------
B = ROOT / "results/pod/bfcl-results/score/qwen3.8-27b-FC/multi_turn"
S = G / "bfcl/score/qwen3.8-27b-grpo20-FC/multi_turn"
cats = ["multi_turn_miss_func", "multi_turn_miss_param", "multi_turn_base", "multi_turn_long_context"]
def score(p):
    L = jsonl(p); return L[0], {r["id"]: (r.get("error") or {}).get("error_type") for r in L[1:]}
acc, fb = {}, {}
for c in cats:
    (s, f), (bs, bf) = score(S / f"BFCL_v4_{c}_score.json"), score(B / f"BFCL_v4_{c}_score.json")
    ids = [f"{c}_{i}" for i in range(200)]
    acc[c] = (bs["accuracy"], s["accuracy"])
    fb[c] = (sum(1 for i in ids if i in bf and i not in f), sum(1 for i in ids if i not in bf and i in f))
    if c.endswith("long_context"):
        ctx = {i for i, e in bf.items() if e == "multi_turn:inference_error"}   # baseline's 65k context-limit failures
        lc = (sum(1 for i in ids if i in bf and i not in f and i not in ctx), sum(1 for i in ids if i not in bf and i in f and i not in ctx))
short = lambda c: c.replace("multi_turn_", "")
fig, ax = plt.subplots(figsize=(8.6, 3.9))
names = [short(c) for c in cats] + ["all 800"]
bv = [acc[c][0] for c in cats] + [sum(acc[c][0] for c in cats) / 4]; gv = [acc[c][1] for c in cats] + [sum(acc[c][1] for c in cats) / 4]
w = 0.38
for off, vals, col, name in ((-w / 2, bv, REF, "base Qwen3.8-27B"), (w / 2, gv, GRPO, "after SFT + GRPO (step 20)")):
    ax.bar([i + off for i in range(len(names))], vals, w - 0.03, color=col, label=name)
    for i, v in enumerate(vals): ax.text(i + off, v + 0.015, f"{v:.1%}", ha="center", fontsize=8, color=INK)
ax.set_xticks(range(len(names)), names); ax.set_ylim(0, 1); ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}"); style(ax)
ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.08))
ax.set_title("BFCL v4 multi-turn accuracy (temp 0.001, 200 entries each)", loc="left")
ax.text(3, 0.74, "long_context: +9 of these passes are\nonly the bigger context (131k vs 65k);\nlike-for-like it is 119 vs 128 of 189", ha="center", fontsize=7.5, color=INK2)
save(fig, "fig5_bfcl_accuracy.png")

rows = [(short(c), *fb[c]) for c in cats] + [("long_context\n(189 that fit 65k)", *lc), ("all 800", sum(fb[c][0] for c in cats), sum(fb[c][1] for c in cats))]
fig, ax = plt.subplots(figsize=(8.6, 4.4))
for i, (name, fx, br) in enumerate(rows):
    y = len(rows) - 1 - i
    ax.barh(y, fx, color=GRPO, height=0.6, label="fixed (base failed, now passes)" if i == 0 else None)
    ax.barh(y, -br, color=BROKEN, height=0.6, label="broken (base passed, now fails)" if i == 0 else None)
    ax.text(fx + 1.2, y, f"+{fx}", va="center", fontsize=8.5, color=INK); ax.text(-br - 1.2, y, f"-{br}", va="center", ha="right", fontsize=8.5, color=INK)
    ax.text(1.02, y, f"p = {mcnemar(fx, br):.3f}", va="center", ha="left", fontsize=8.5, color=INK2, transform=ax.get_yaxis_transform(), clip_on=False)
ax.text(1.02, len(rows) - 0.35, "McNemar\nexact p", va="bottom", ha="left", fontsize=8, color=INK2, transform=ax.get_yaxis_transform(), clip_on=False)
ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1]); ax.axvline(0, color=INK2, linewidth=1)
ax.set_xlim(-68, 92); style(ax, "x"); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8.5)
ax.set_xlabel("entries (the same 200 entries per category for both models)")
ax.set_title("BFCL multi-turn: entries fixed vs broken by SFT + GRPO", loc="left")
save(fig, "fig6_bfcl_fixed_broken.png")

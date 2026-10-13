"""Creates grpo_q38.ipynb ONCE (refuses to overwrite it: the notebook is then edited by hand).
It reads only JSON: results/fs_env/grpo/grpo_notebook_config.json (which run) and the run's status.json, grpo_log.jsonl,
probes.jsonl and rollouts/step_<n>.jsonl, which ./grpo_updater.sh copies from the pod every 2 min.   python3 build_grpo_notebook.py"""
import sys
from pathlib import Path

import nbformat as nbf

md = lambda s: nbf.v4.new_markdown_cell(s.strip("\n"))
code = lambda s, tags=None: nbf.v4.new_code_cell(s.strip("\n"), metadata={"tags": tags} if tags else {})

INTRO = """
# GRPO on fs_env: Qwen3.8-27B, starting from the SFT warm start

Each step draws **training-split tasks** (A3, D, E first; balanced: one of each cause in turn, a fresh random user name,
random folder names and clutter per draw), plays **a group of conversations per task** with the current LoRA (vLLM,
temperature 1), and scores **every turn** with the fs_env checker (files match the target, reply check where the turn
needs one; no LLM judge). The advantage is computed within the group for the same task and turn; groups where every
conversation got the same result teach nothing and are dropped, and a task whose group is flat twice leaves the pool. The
update is an on-policy policy-gradient step on the LoRA, then the new adapter is hot-loaded into vLLM.

**How to read it:** the reward should rise over steps; *mixed groups* is how much learning signal each step had; the
over-reaction probes must stay above the red line (they score the first move with thinking off, a coarse check: real
held-out conversations are run at the end).

**Updates:** this notebook reads JSON only. `./grpo_updater.sh start` copies the run's files from the pod every 2 min; re-run
the cells or the live cell. The run is chosen in `results/fs_env/grpo/grpo_notebook_config.json`.
"""

SETUP = r'''
import json, datetime, statistics
from pathlib import Path
from collections import Counter, defaultdict
from IPython.display import HTML, display
ROOT = next(p for p in (Path("."), Path("..")) if (p / "results/fs_env/grpo").exists())
CFG = json.loads((ROOT / "results/fs_env/grpo/grpo_notebook_config.json").read_text())
RUN_DIR = ROOT / CFG["local_dir"]; PRICE_H = CFG.get("price_per_hour", 1.80)
C_MAIN = "#7A4E9C"
CAT = {"A3": "#2a78d6", "D": "#eb6834", "E": "#1baf7a", "A4": "#eda100", "B": "#e87ba4", "C": "#008300", "F": "#4a3aa7", "G": "#888", "H": "#444"}
def read_json(p, d=None):
    try: return json.loads(Path(p).read_text())
    except (OSError, ValueError): return d
def read_jsonl(p):
    rows = []
    if Path(p).exists():
        for l in open(p):
            try: rows.append(json.loads(l))
            except ValueError: pass
    return rows
def fmt_min(s):
    if s is None: return "-"
    m = int(round(s / 60)); return f"{m} min" if m < 60 else f"{m // 60} h {m % 60:02d} min"
def table(rows, head=None):
    h = "".join(f"<th style='text-align:left;padding:2px 10px'>{x}</th>" for x in head) if head else ""
    b = "".join("<tr>" + "".join(f"<td style='padding:2px 10px'>{x}</td>" for x in r) + "</tr>" for r in rows)
    return f"<table style='border-collapse:collapse;font-size:13px'>{'<tr>' + h + '</tr>' if h else ''}{b}</table>"
def load():
    return (read_json(RUN_DIR / "status.json", {}), read_jsonl(RUN_DIR / "grpo_log.jsonl"), read_jsonl(RUN_DIR / "probes.jsonl"),
            {int(p.stem.split("_")[1]): read_jsonl(p) for p in sorted((RUN_DIR / "rollouts").glob("step_*.jsonl"))} if (RUN_DIR / "rollouts").exists() else {})
S, LOG, PROBES, ROLL = load()
print(f"run {CFG['run']}: stage {S.get('stage', 'not started / not fetched')}, {len(LOG)} steps logged, {len(PROBES)} probe checks, {len(ROLL)} rollout files")
'''

INFO = r'''
def finish_time(S):
    if S.get("eta_s") is None or not S.get("updated"): return "-"
    if str(S.get("stage", "")).startswith(("finished", "stopped")): return "done"
    t = datetime.datetime.fromisoformat(S["updated"]).replace(tzinfo=datetime.timezone.utc) + datetime.timedelta(seconds=S["eta_s"])
    return f"<b>{t.astimezone().strftime('%H:%M')}</b>"
def info_html(S, LOG, PROBES):
    a = S.get("args", {}); last = LOG[-1] if LOG else {}; pct = lambda x: f"{x:.0%}" if isinstance(x, (int, float)) else "-"
    o = [f"<h4>{CFG['run']} &middot; stage: <span style='color:{C_MAIN}'>{S.get('stage', 'not started')}</span>"
         f"{' &middot; updated ' + S['updated'].replace('T', ' ') + ' (pod clock, UTC)' if S.get('updated') else ''}</h4>"]
    o.append("<b>Progress</b>" + table([
        ["step", f"{S.get('step', 0)} / {S.get('max_steps', a.get('steps', '-'))}"],
        ["elapsed / ETA", f"{fmt_min(S.get('elapsed_s'))} / {fmt_min(S.get('eta_s'))}"],
        ["estimated finish (your time)", finish_time(S)],
        ["last step", f"reward (turns) {pct(last.get('reward_turn'))}, conversations passed {pct(last.get('reward_conv'))}, "
                      f"mixed groups {last.get('turn_groups_mixed', '-')}/{(last.get('turn_groups_mixed') or 0) + (last.get('turn_groups_flat') or 0) or '-'}, "
                      f"{last.get('train_items', '-')} trained turns, {last.get('step_s', '-')} s" if last else "-"],
        ["tasks left in the pool", last.get("pool_left", "-")],
        ["GPU memory (peak, trainer)", f"{S.get('gpu_mem_gb', '-')} GB"],
        ["GPU cost so far", f"&#36;{S['elapsed_s'] / 3600 * PRICE_H:.2f}" if S.get("elapsed_s") else "-"]]))
    o.append("<b>Settings</b>" + table([
        ["causes / tasks per step / group", f"{a.get('causes', '-')} / {a.get('tasks_per_step', '-')} / {a.get('group', '-')}"],
        ["learning rate, grad clip", f"{a.get('lr', '-')}, {a.get('max_grad_norm', '-')}"],
        ["rollouts", f"temperature {a.get('temperature', '-')}, up to {a.get('max_steps', '-')} model calls per turn"],
        ["start adapter", a.get("init_adapter") or "fresh LoRA"],
        ["probes", f"every {a.get('probe_every', '-')} steps, stop if the act rate drops more than {pct(a.get('max_act_drop'))}"]]))
    return "".join(o)
display(HTML(info_html(S, LOG, PROBES)))
'''

LIVE = r'''
# live view: re-reads the JSON (kept current by ./grpo_updater.sh) every 30 s until the run ends (interrupt to stop)
import time
from IPython.display import clear_output
try:
    while True:
        S, LOG, PROBES, ROLL = load(); clear_output(wait=True); display(HTML(info_html(S, LOG, PROBES)))
        if str(S.get("stage", "")).startswith(("finished", "stopped")): break
        time.sleep(30)
except KeyboardInterrupt:
    pass
'''

PREP = r'''
import plotly.graph_objects as go, plotly.io as pio
from plotly.subplots import make_subplots
pio.renderers.default = "notebook_connected"; pio.templates.default = "plotly_white"
def placeholder(title, text="no data yet"):
    f = go.Figure(); f.add_annotation(text=text, showarrow=False, font=dict(size=16, color="#888"))
    f.update_layout(title=title, height=240, xaxis=dict(visible=False), yaxis=dict(visible=False)); f.show()
S, LOG, PROBES, ROLL = load()
'''

REWARD = r'''
if LOG:
    steps = [l["step"] for l in LOG]
    fig = go.Figure()
    fig.add_scatter(x=steps, y=[l["reward_turn"] for l in LOG], mode="lines+markers", name="turns passed (reward)", line=dict(color=C_MAIN, width=3))
    fig.add_scatter(x=steps, y=[l["reward_conv"] for l in LOG], mode="lines+markers", name="whole conversations passed", line=dict(color="#aaa", dash="dot"))
    by = defaultdict(dict)
    for st, rs in ROLL.items():
        for c in {r["cause"] for r in rs}:
            rr = [r for r in rs if r["cause"] == c]
            by[c][st] = sum(sum(t["passed"] for t in r["turns"]) / max(1, len(r["turns"])) for r in rr) / len(rr)
    for c, d in sorted(by.items()):
        xs = sorted(d); fig.add_scatter(x=xs, y=[d[x] for x in xs], mode="markers", name=f"{c} (turns passed)", marker=dict(color=CAT.get(c, "#666"), size=8, opacity=0.7))
    fig.update_layout(title="Reward per step (each step draws different tasks, so expect noise; the trend matters)", yaxis=dict(range=[0, 1.05], tickformat=".0%"),
                      xaxis_title="GRPO step", height=420, legend=dict(orientation="h", y=-0.25))
    fig.show()
else:
    placeholder("Reward per step", "no step finished yet")
'''

SIGNAL = r'''
if LOG:
    steps = [l["step"] for l in LOG]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Learning signal: (task, turn) groups per step", "Update: loss and gradient norm"), specs=[[{}, {"secondary_y": True}]])
    fig.add_bar(x=steps, y=[l["turn_groups_mixed"] for l in LOG], name="mixed (teaches)", marker_color=C_MAIN, row=1, col=1)
    fig.add_bar(x=steps, y=[l["turn_groups_flat"] for l in LOG], name="flat (dropped)", marker_color="#ccc", row=1, col=1)
    fig.add_scatter(x=steps, y=[l["loss"] for l in LOG], mode="lines+markers", name="loss", line=dict(color="#2a78d6"), row=1, col=2)
    fig.add_scatter(x=steps, y=[l["grad_norm"] for l in LOG], mode="lines", name="grad norm", line=dict(color="#eb6834", dash="dot"), row=1, col=2, secondary_y=True)
    fig.update_layout(barmode="stack", height=400, legend=dict(orientation="h", y=-0.25)); fig.show()
else:
    placeholder("Learning signal", "no step finished yet")
'''

PROBE = r'''
if PROBES:
    steps = [p["step"] for p in PROBES]; drop = S.get("args", {}).get("max_act_drop", 0.1); base = PROBES[0].get("act_rate")
    fig = go.Figure()
    if base is not None:
        fig.add_hrect(y0=0, y1=max(0, base - drop), fillcolor="#c0392b", opacity=0.07, line_width=0)
        fig.add_hline(y=base - drop, line=dict(color="#c0392b", dash="dash"), annotation_text=f"stop line {base - drop:.0%}", annotation_position="bottom right")
    for key, name, color in [("act_rate", "should act: starts with a tool call", "#2a78d6"), ("refuse_no_tool", "E, tool missing: no tool call", "#1baf7a"),
                             ("ask_no_tool", "D, info missing: no tool call", "#eb6834")]:
        fig.add_scatter(x=steps, y=[p.get(key) for p in PROBES], mode="lines+markers", name=name, line=dict(color=color))
    fig.update_layout(title="Over-reaction probes (held-out tasks, first move with thinking off: coarse)", yaxis=dict(range=[-0.02, 1.05], tickformat=".0%"),
                      xaxis_title="GRPO step", height=400, legend=dict(orientation="h", y=-0.25)); fig.show()
else:
    placeholder("Over-reaction probes", "no probe check yet")
'''

GRID = r'''
if ROLL:
    steps = sorted(ROLL); width = max(len({r["group_task"] for r in ROLL[s]}) for s in steps)
    z, txt, hov = [], [], []
    for s in steps[::-1]:
        tasks = list(dict.fromkeys(r["group_task"] for r in ROLL[s]))
        zr, tr, hr = [], [], []
        for j in range(width):
            tid = tasks[j] if j < len(tasks) else None; rs = [r for r in ROLL[s] if r["group_task"] == tid]
            zr.append(sum(r["passed"] for r in rs) / len(rs) if rs else None)
            tr.append(f"<b>{sum(r['passed'] for r in rs)}/{len(rs)}</b><br>{tid.split('_')[0]} {'_'.join(tid.split('_')[2:])[:16] if tid.split('_')[1].isdigit() else 'original'}" if rs else "")
            hr.append(f"<b>{tid}</b> step {s}<br>{sum(r['passed'] for r in rs)}/{len(rs)} conversations passed" if rs else "")
        z.append(zr); txt.append(tr); hov.append(hr)
    fig = go.Figure(go.Heatmap(z=z, x=[f"task {j + 1}" for j in range(width)], y=[f"step {s}" for s in steps[::-1]], text=txt, texttemplate="%{text}", textfont=dict(size=10),
                               customdata=hov, hovertemplate="%{customdata}<extra></extra>", zmin=0, zmax=1, xgap=2, ygap=2,
                               colorscale=[[0, "#F4EFF8"], [0.5, "#B48FCF"], [1, "#4A2A66"]], colorbar=dict(title="group<br>passed", tickformat=".0%")))
    fig.update_layout(title="Every group: conversations passed out of the group (mid shades = a learning signal; white/dark = nothing to learn)",
                      height=140 + 48 * len(steps), xaxis=dict(side="top")); fig.show()
else:
    placeholder("Groups per step", "no rollouts yet")
'''

SAMPLES = r'''
if ROLL:
    s = max(ROLL); rows = []
    for r in ROLL[s][:16]:
        t = r["turns"][0]
        rows.append([r["group_task"], "".join("✅" if x["passed"] else "❌" for x in r["turns"]), ", ".join(c["name"] for c in t["calls"]) or "none",
                     (t["answer"] or "").strip().replace("<", "&lt;")[:180]])
    display(HTML(f"<b>Step {s}</b>: first 16 conversations" + table(rows, ["task", "turns", "turn-1 tool calls", "turn-1 reply"])))
else:
    print("no rollouts yet")
'''

SECTIONS = [("## 1. Run info", None), ("## 2. Reward", "REWARD"), ("## 3. Learning signal and update", "SIGNAL"),
            ("## 4. Over-reaction probes", "PROBE"), ("## 5. Every group, every step", "GRID"), ("## 6. Latest step: what the model did", "SAMPLES")]


def build():
    cells = [md(INTRO), code(SETUP), md(SECTIONS[0][0]), code(INFO), code(LIVE, ["skip-execution"]), md("---\n# Graphs"), code(PREP)]
    for title, key in SECTIONS[1:]:
        cells += [md(title), code(globals()[key])]
    nb = nbf.v4.new_notebook(); nb.cells = cells
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return nb


if __name__ == "__main__":
    out = Path("grpo_q38.ipynb")
    if out.exists() and "--force" not in sys.argv:
        sys.exit(f"{out} exists and is edited by hand: not overwriting (use --force to rebuild from scratch and lose those edits)")
    nbf.write(build(), str(out))
    print(f"wrote {out}")

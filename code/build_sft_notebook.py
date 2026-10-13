"""Creates sft_q38_27b_train.ipynb ONCE (refuses to overwrite it: the notebook is then edited by hand).
The notebook reads only JSON: results/fs_env/sft/sft_notebook_config.json (which run) and the run's status.json,
train_log.jsonl and probes.jsonl, which ./sft_updater.sh copies from the pod every 2 min.   python3 build_sft_notebook.py"""
import sys
from pathlib import Path

import nbformat as nbf

md = lambda s: nbf.v4.new_markdown_cell(s.strip("\n"))
code = lambda s, tags=None: nbf.v4.new_code_cell(s.strip("\n"), metadata={"tags": tags} if tags else {})

INTRO = """
# SFT warm start for Qwen3.8-27B on fs_env (QLoRA)

Why: on the fs_env tasks the model almost never refuses when a needed tool is missing (E: 1/40 in Part 1, 5/40 after the fixes),
so GRPO would have nothing to learn from (every sample in a group fails the same way). A light SFT warm start makes the right
behavior appear *sometimes*, which is what GRPO needs. It must not make the model over-react (refuse or ask where it should act).

**Data** (`train_v1`, train split only): one-step E refusals (tool missing: say so at once, ask for it), E "tool is back: use it
right away" turns, and conversations the model already gets right (all causes), as a counterweight. Each example is one turn with
the tool list of that turn; the loss covers only that turn's assistant messages (thinking, tool calls, reply).

**Over-reaction probes** (`fs_env/sft_probes.py`), at step 0 and every N steps, on held-out EVAL tasks only: P(the reply starts with
a tool call) where the model should act (falling = over-reacting), should refuse (E) or should ask (D). Training stops by itself if
the act rate falls more than the set margin below step 0. A LoRA checkpoint is saved at every check.

**How this notebook updates:** it reads JSON only. `./sft_updater.sh start` copies `status.json`, `train_log.jsonl` and `probes.jsonl`
from the pod every 2 min; re-run the cells (or the live cell) to see new data. Which run is shown is set in
`results/fs_env/sft/sft_notebook_config.json`, so a new run never needs a change here.
"""

SETUP = r'''
import json, datetime
from pathlib import Path
from IPython.display import HTML, display
ROOT = next(p for p in (Path("."), Path("..")) if (p / "results/fs_env/sft").exists())
CFG = json.loads((ROOT / "results/fs_env/sft/sft_notebook_config.json").read_text())
RUN_DIR = ROOT / CFG["local_dir"]
PRICE_H = CFG.get("price_per_hour", 1.80)
C_MAIN, C_MUTED = "#7A4E9C", "#E6E6E6"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
def read_json(p, default=None):
    try: return json.loads(Path(p).read_text())
    except (OSError, ValueError): return default
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
    return read_json(RUN_DIR / "status.json", {}), read_jsonl(RUN_DIR / "train_log.jsonl"), read_jsonl(RUN_DIR / "probes.jsonl")
S, LOG, PROBES = load()
META = read_json(ROOT / CFG["data_meta"], {})
print(f"run {CFG['run']}: stage {S.get('stage', 'not started / not fetched')}, {len(LOG)} logged steps, {len(PROBES)} probe checks")
'''

INFO = r'''
def finish_time(S):   # pod's last update (UTC) + ETA, shown in this computer's time zone
    if S.get("eta_s") is None or not S.get("updated"): return "-"
    if str(S.get("stage", "")).startswith(("finished", "stopped")): return "done"
    t = datetime.datetime.fromisoformat(S["updated"]).replace(tzinfo=datetime.timezone.utc) + datetime.timedelta(seconds=S["eta_s"])
    return f"<b>{t.astimezone().strftime('%H:%M')}</b> (rough: steps vary, and each probe check adds a few minutes)"
def info_html(S, LOG, PROBES):
    a = S.get("args", {}); d = S.get("data", {})
    last = PROBES[-1] if PROBES else {}; base = PROBES[0] if PROBES else {}
    pct = lambda x: f"{x:.0%}" if isinstance(x, (int, float)) else "-"
    o = [f"<h4>{CFG['run']} &middot; stage: <span style='color:{C_MAIN}'>{S.get('stage', 'not started')}</span>"
         f"{' &middot; updated ' + S['updated'].replace('T', ' ') + ' (pod clock, UTC)' if S.get('updated') else ''}</h4>"]
    o.append("<b>Progress</b>" + table([
        ["step", f"{S.get('step', 0)} / {S.get('max_steps', '-')}"],
        ["last loss", f"{S['last_loss']:.3f}" if S.get("last_loss") is not None else "-"],
        ["elapsed / ETA", f"{fmt_min(S.get('elapsed_s'))} / {fmt_min(S.get('eta_s'))}"],
        ["estimated finish (your time)", finish_time(S)],
        ["GPU memory (peak)", f"{S.get('gpu_mem_gb', '-')} GB"],
        ["GPU cost so far", f"&#36;{S['elapsed_s'] / 3600 * PRICE_H:.2f}" if S.get("elapsed_s") else "-"]]))
    o.append("<b>Over-reaction probes</b> (step 0 &rarr; latest)" + table([
        ["act rate (should act: high)", f"{pct(base.get('act_rate'))} &rarr; <b>{pct(last.get('act_rate'))}</b>"
         + (f" &nbsp; (stop if below {pct(base['act_rate'] - a.get('max_act_drop', 0.1))})" if base.get('act_rate') is not None else "")],
        ["E: no tool call when the tool is missing", f"{pct(base.get('refuse_no_tool'))} &rarr; <b>{pct(last.get('refuse_no_tool'))}</b>"],
        ["E: prefers saying it can't (vs asking)", f"{pct(base.get('refuse_says_cant'))} &rarr; <b>{pct(last.get('refuse_says_cant'))}</b>"],
        ["D: no tool call when info is missing", f"{pct(base.get('ask_no_tool'))} &rarr; <b>{pct(last.get('ask_no_tool'))}</b>"],
        ["checks so far", f"{len(PROBES)} (last at step {last.get('step', '-')})" + (" &middot; <b style='color:#c0392b'>STOPPED: over-reacting</b>" if last.get("stopped") else "")]]))
    o.append("<b>Data and settings</b>" + table([
        ["examples", f"{d.get('examples', '-')} ({d.get('dropped') or 'none dropped'}) &middot; {d.get('tokens', 0):,} tokens, {d.get('trained_tokens', 0):,} trained"],
        ["model / LoRA", f"{S.get('model_class', '-')} &middot; rank {a.get('rank', '-')}, alpha {a.get('alpha', '-')}, {S.get('trainable_params', 0):,} trainable params"],
        ["optimizer", f"lr {a.get('lr', '-')}, cosine, grad accum {a.get('grad_accum', '-')}, {a.get('epochs', '-')} epoch(s), max len {a.get('max_len', '-')}"],
        ["probes", f"every {a.get('probe_every', '-')} steps, stop if the act rate drops more than {pct(a.get('max_act_drop'))}"]]))
    return "".join(o)
display(HTML(info_html(S, LOG, PROBES)))
'''

LIVE = r'''
# live view: re-reads the JSON (kept current by ./sft_updater.sh) every 30 s until the run ends (interrupt the kernel to stop)
import time
from IPython.display import clear_output
try:
    while True:
        S, LOG, PROBES = load(); clear_output(wait=True); display(HTML(info_html(S, LOG, PROBES)))
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
S, LOG, PROBES = load()
'''

DATA = r'''
if META.get("sources"):
    src = META["sources"]; about = META.get("about", {})
    fig = make_subplots(rows=1, cols=2, column_widths=[0.45, 0.55], subplot_titles=("Turn examples by source", "Turn examples by cause"))
    fig.add_bar(y=[about.get(k, k) for k in src], x=list(src.values()), orientation="h", marker_color=CAT[:len(src)], text=list(src.values()), textposition="outside", showlegend=False, row=1, col=1)
    bc = dict(sorted(META.get("by_cause", {}).items()))
    fig.add_bar(x=list(bc), y=list(bc.values()), marker_color=C_MAIN, text=list(bc.values()), textposition="outside", showlegend=False, row=1, col=2)
    fig.update_layout(title=f"{META.get('name', '')}: {META.get('examples', '?')} turn examples · refusals {META.get('refusal_share', 0):.0%}", height=380)
    fig.show()
else:
    placeholder("Training data", "no data meta file")
'''

LOSS = r'''
if LOG:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_scatter(x=[r["step"] for r in LOG], y=[r["loss"] for r in LOG], mode="lines+markers", name="loss", line=dict(color=C_MAIN))
    if any("learning_rate" in r for r in LOG):
        fig.add_scatter(x=[r["step"] for r in LOG], y=[r.get("learning_rate") for r in LOG], mode="lines", name="learning rate", line=dict(color="#aaa", dash="dot"), secondary_y=True)
    for p in PROBES[1:]:
        fig.add_vline(x=p["step"], line=dict(color="#ccc", width=1))
    fig.update_layout(title="Training loss per step (grey lines: probe checks / saved checkpoints)", height=380, legend=dict(orientation="h", y=-0.2))
    fig.update_yaxes(title_text="loss", secondary_y=False); fig.update_yaxes(title_text="learning rate", secondary_y=True)
    fig.show()
else:
    placeholder("Training loss", "no training steps logged yet")
'''

PROBE = r'''
if PROBES:
    steps = [p["step"] for p in PROBES]; drop = S.get("args", {}).get("max_act_drop", 0.1)
    lines = [("act_rate", "should act: starts with a tool call", CAT[0]), ("refuse_no_tool", "E, tool missing: no tool call", CAT[1]),
             ("refuse_says_cant", "E: says it can't (vs asks)", CAT[4]), ("ask_no_tool", "D, info missing: no tool call", CAT[2]),
             ("ask_asks", "D: asks (vs says it can't)", CAT[3])]
    fig = go.Figure()
    base = PROBES[0].get("act_rate")
    if base is not None:
        fig.add_hrect(y0=0, y1=max(0, base - drop), fillcolor="#c0392b", opacity=0.07, line_width=0)
        fig.add_hline(y=base - drop, line=dict(color="#c0392b", dash="dash"), annotation_text=f"stop line: act rate {base - drop:.0%}", annotation_position="bottom right")
    for key, name, color in lines:
        fig.add_scatter(x=steps, y=[p.get(key) for p in PROBES], mode="lines+markers", name=name, line=dict(color=color, width=3 if key == "act_rate" else 2))
    fig.update_layout(title="Over-reaction probes per check (held-out eval tasks; first move with thinking off)", yaxis=dict(range=[-0.02, 1.05], tickformat=".0%"),
                      xaxis_title="training step", height=440, legend=dict(orientation="h", y=-0.25))
    fig.show()
else:
    placeholder("Over-reaction probes", "no probe check yet (the first one runs before step 1)")
'''

GROUPS = r'''
if PROBES and PROBES[0].get("act_rate_by"):
    b, l = PROBES[0]["act_rate_by"], PROBES[-1]["act_rate_by"]; ks = sorted(set(b) | set(l))
    fig = go.Figure()
    fig.add_bar(x=ks, y=[b.get(k) for k in ks], name=f"step {PROBES[0]['step']}", marker_color=C_MUTED, text=[f"{b.get(k, 0):.0%}" for k in ks], textposition="outside")
    fig.add_bar(x=ks, y=[l.get(k) for k in ks], name=f"step {PROBES[-1]['step']} (latest)", marker_color=C_MAIN, text=[f"{l.get(k, 0):.0%}" for k in ks], textposition="outside")
    fig.update_layout(title="Act rate per group of 'should act' probes: start vs latest (a drop in one group = over-reacting there)",
                      yaxis=dict(range=[0, 1.15], tickformat=".0%"), height=400, legend=dict(orientation="h", y=-0.2))
    fig.show()
else:
    placeholder("Act rate per group", "no probe check yet")
'''

TABLE = r'''
pct = lambda x: f"{x:.0%}" if isinstance(x, (int, float)) else "-"
rows = [[p["step"], pct(p.get("act_rate")), pct(p.get("act_drop")), pct(p.get("refuse_no_tool")), pct(p.get("refuse_says_cant")),
         pct(p.get("ask_no_tool")), pct(p.get("ask_asks")), "<b style='color:#c0392b'>stop</b>" if p.get("stopped") else ""] for p in PROBES]
display(HTML(table(rows, ["step", "act rate", "drop vs step 0", "E no tool", "E says can't", "D no tool", "D asks", ""]) if rows else "no probe checks yet"))
'''

SECTIONS = [("## 1. Run info", None), ("## 2. Training data", "DATA"), ("## 3. Loss", "LOSS"),
            ("## 4. Over-reaction probes\nThe blue line must stay above the red dashed line (step-0 act rate minus the allowed drop); below it training stops by itself. The E/D lines are what the warm start should raise.", "PROBE"),
            ("## 5. Act rate per probe group", "GROUPS"), ("## 6. Every probe check", "TABLE")]


def build():
    cells = [md(INTRO), code(SETUP), md(SECTIONS[0][0]), code(INFO), code(LIVE, ["skip-execution"]), md("---\n# Graphs"), code(PREP)]
    for title, key in SECTIONS[1:]:
        cells += [md(title), code(globals()[key])]
    nb = nbf.v4.new_notebook(); nb.cells = cells
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return nb


if __name__ == "__main__":
    out = Path("sft_q38_27b_train.ipynb")
    if out.exists() and "--force" not in sys.argv:
        sys.exit(f"{out} exists and is edited by hand: not overwriting (use --force to rebuild from scratch and lose those edits)")
    nbf.write(build(), str(out))
    print(f"wrote {out}")

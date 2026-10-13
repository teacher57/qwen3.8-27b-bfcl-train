"""Creates bfcl_eval_after_grpo_q38_27b.ipynb ONCE (refuses to overwrite it: the notebook is then edited by hand).
It reads only JSON: results/fs_env/grpo/bfcl_mt_notebook_config.json (which run, baseline, price) and the run's BFCL
result/score files, progress.jsonl and status.json, which ./bfcl_mt_updater.sh copies from the pod every 2 min.
    ../.venv/bin/python build_bfcl_mt_notebook.py"""
import sys
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parent / "bfcl_eval_after_grpo_q38_27b.ipynb"
md = lambda s: nbf.v4.new_markdown_cell(s.strip("\n"))
code = lambda s, tags=None: nbf.v4.new_code_cell(s.strip("\n"), metadata={"tags": tags} if tags else {})

INTRO = """
# BFCL multi-turn: Qwen3.8-27B after SFT + GRPO on fs_env

Does training on our file-system environment (fs_env) transfer to BFCL's multi-turn tasks? The chosen GRPO checkpoint runs
BFCL v4's four multi-turn categories (200 entries each) with the **same settings as the base-model baseline**, so the
change per category is the transfer. **miss_func** (a needed function is missing: the model should say so) and
**miss_param** (a needed value is missing: the model should ask) run first: they are closest to what fs_env trained
(E: refuse without the tool; D: ask when information is missing). Then **base** and **long_context**.

**How to read it:** section 2 is the headline (accuracy vs baseline). Section 3 shows which entries changed: *fixed*
(baseline failed, now passes) vs *broken* (baseline passed, now fails); with 200 entries, a few points either way can be
noise, so look at fixed vs broken together. Section 4 shows how the failures changed.

**Updates:** this notebook reads JSON only. `./bfcl_mt_updater.sh start` copies the run's files from the pod every 2 min;
re-run the cells, or the live cell. The run is chosen in `results/fs_env/grpo/bfcl_mt_notebook_config.json`.
"""

SETUP = r'''
import json, datetime
from pathlib import Path
from collections import Counter
from IPython.display import HTML, display
ROOT = next(p for p in (Path("."), Path("..")) if (p / "results/fs_env/grpo").exists())
CFG_PATH = ROOT / "results/fs_env/grpo/bfcl_mt_notebook_config.json"
C_MAIN, C_BASE, C_FIX, C_BROKE = "#7A4E9C", "#aaa", "#1baf7a", "#c0392b"
def read_json(p, d=None):
    try: return json.loads(Path(p).read_text())
    except (OSError, ValueError): return d
def read_jsonl(p):
    rows = []
    if Path(p).exists():
        for l in open(p, errors="replace"):
            try: rows.append(json.loads(l))
            except ValueError: pass
    return rows
def table(rows, head=None):
    h = "".join(f"<th style='text-align:left;padding:2px 10px'>{x}</th>" for x in head) if head else ""
    b = "".join("<tr>" + "".join(f"<td style='padding:2px 10px'>{x}</td>" for x in r) + "</tr>" for r in rows)
    return f"<table style='border-collapse:collapse;font-size:13px'>{'<tr>' + h + '</tr>' if h else ''}{b}</table>"
def fmt_min(s):
    if s is None: return "-"
    m = int(round(s / 60)); return f"{m} min" if m < 60 else f"{m // 60} h {m % 60:02d} min"
def score_file(path):
    """BFCL score file: line 1 = summary, then one line per FAILED entry (id, error type)."""
    rows = read_jsonl(path) if path else []
    if not rows: return None
    return {"summary": rows[0], "failed": {r["id"]: (r.get("error") or {}).get("error_type", "?") for r in rows[1:] if "id" in r}}
def load():
    cfg = json.loads(CFG_PATH.read_text()); run = ROOT / cfg["local_dir"]; m = cfg["model"]
    cats = {}
    for c in cfg["categories"]:
        res = next(iter((run / "result" / m).glob(f"**/BFCL_v4_{c}_result.json")), None)
        sc = next(iter((run / "score" / m).glob(f"**/BFCL_v4_{c}_score.json")), None)
        cats[c] = {"done_ids": [r["id"] for r in read_jsonl(res) if "id" in r] if res else [],
                   "score": score_file(sc), "base": score_file(ROOT / cfg["baseline_dir"] / f"BFCL_v4_{c}_score.json")}
    return cfg, read_json(run / "status.json", {}), read_jsonl(run / "progress.jsonl"), cats
CFG, ST, PROG, CATS = load()
print(f"{CFG['model']}: {sum(len(v['done_ids']) for v in CATS.values())} entries generated, "
      f"{sum(1 for v in CATS.values() if v['score'])} of {len(CATS)} categories scored")
'''

INFO = r'''
def eta(cfg, prog, cats):
    """rate over the last ~30 min of progress lines -> time left for everything not yet generated"""
    total = cfg["total_per_category"] * len(cfg["categories"]); done = sum(len(v["done_ids"]) for v in cats.values())
    if len(prog) < 2: return done, total, None, None
    t = lambda p: datetime.datetime.fromisoformat(p["t"]).replace(tzinfo=datetime.timezone.utc)
    last = prog[-1]; first = next((p for p in prog if (t(last) - t(p)).total_seconds() <= 1800), prog[0])
    dt = (t(last) - t(first)).total_seconds(); dn = sum(last["done"].values()) - sum(first["done"].values())
    if dt <= 0 or dn <= 0: return done, total, None, None
    rate = dn / dt; left = (total - done) / rate
    return done, total, rate * 3600, t(last) + datetime.timedelta(seconds=left)
def info_html(cfg, st, prog, cats):
    done, total, per_h, fin = eta(cfg, prog, cats)
    start = datetime.datetime.fromisoformat(cfg["started_utc"]).replace(tzinfo=datetime.timezone.utc)
    now = datetime.datetime.fromisoformat(st["updated"]).replace(tzinfo=datetime.timezone.utc) if st.get("updated") else start
    el = (now - start).total_seconds(); p = cfg.get("price_per_hour", 0)
    stage = "finished" if st.get("finished") else ("running" if st.get("alive") else "idle / not started")
    o = [f"<h4>{cfg['model_label']} &middot; <span style='color:{C_MAIN}'>{stage}</span>"
         f"{' &middot; updated ' + st['updated'].replace('T', ' ') + ' UTC' if st.get('updated') else ''}</h4>"]
    o.append(table([
        ["entries generated", f"<b>{done} / {total}</b>"],
        ["per category", " &middot; ".join(f"{c[11:]} {len(v['done_ids'])}/{cfg['total_per_category']}" + (" &#10003;" if v["score"] else "") for c, v in cats.items())],
        ["pace (last 30 min)", f"{per_h:.0f} entries/h" if per_h else "-"],
        ["elapsed / estimated finish (your time)", f"{fmt_min(el)} / " + ("done" if st.get("finished") else f"<b>{fin.astimezone().strftime('%H:%M')}</b>" if fin else "-")],
        ["GPU cost so far / estimated total", f"&#36;{el / 3600 * p:.2f} / " + (f"&#36;{(fin - start).total_seconds() / 3600 * p:.2f}" if fin else "-")],
        ["why this model", cfg.get("chosen_because", "-")],
        ["settings", cfg.get("settings", "-")]]))
    return "".join(o)
def short(c): return c.replace("multi_turn_", "")
def acc_html(cats):
    rows = []
    for c, v in cats.items():
        s, b = v["score"], v["base"]
        a = s["summary"]["accuracy"] if s else None; ba = b["summary"]["accuracy"] if b else None
        d = (a - ba) * 100 if a is not None and ba is not None else None
        col = C_FIX if d and d > 0 else C_BROKE if d and d < 0 else "#888"
        rows.append([f"<b>{short(c)}</b>", f"{len(v['done_ids'])}/{CFG['total_per_category']}",
                     f"<b>{a:.1%}</b> ({s['summary']['correct_count']}/{s['summary']['total_count']})" if s else "<i>not scored yet</i>",
                     f"{ba:.1%}" if ba is not None else "-", f"<span style='color:{col}'><b>{d:+.1f} pts</b></span>" if d is not None else "-"])
    return f"<b>Accuracy: {CFG['model_label']} vs {CFG['baseline_label']}</b>" + table(rows, ["category", "generated", "accuracy", "baseline", "change"])
display(HTML(info_html(CFG, ST, PROG, CATS)))
'''

LIVE = r'''
# live view: re-reads the JSON (kept current by ./bfcl_mt_updater.sh) every 30 s until the run ends (interrupt to stop)
import time
from IPython.display import clear_output
try:
    while True:
        CFG, ST, PROG, CATS = load(); clear_output(wait=True)
        display(HTML(info_html(CFG, ST, PROG, CATS) + "<br>" + acc_html(CATS)))
        if ST.get("finished"): break
        time.sleep(30)
except KeyboardInterrupt:
    pass
'''

GRAPH_SETUP = r'''
import plotly.graph_objects as go, plotly.io as pio
pio.renderers.default = "notebook_connected"; pio.templates.default = "plotly_white"
def placeholder(title, text="no data yet"):
    f = go.Figure(); f.add_annotation(text=text, showarrow=False, font=dict(size=16, color="#888"))
    f.update_layout(title=title, height=240, xaxis=dict(visible=False), yaxis=dict(visible=False)); f.show()
CFG, ST, PROG, CATS = load()
'''

ACC = r'''
display(HTML(acc_html(CATS)))
scored = [c for c, v in CATS.items() if v["score"]]
if scored:
    fig = go.Figure()
    fig.add_bar(x=[short(c) for c in scored], y=[CATS[c]["base"]["summary"]["accuracy"] for c in scored], name=CFG["baseline_label"], marker_color=C_BASE,
                text=[f"{CATS[c]['base']['summary']['accuracy']:.1%}" for c in scored], textposition="outside")
    fig.add_bar(x=[short(c) for c in scored], y=[CATS[c]["score"]["summary"]["accuracy"] for c in scored], name=CFG["model_label"], marker_color=C_MAIN,
                text=[f"{CATS[c]['score']['summary']['accuracy']:.1%}" for c in scored], textposition="outside")
    fig.update_layout(title="BFCL multi-turn accuracy per category", barmode="group", yaxis=dict(range=[0, 1.1], tickformat=".0%"), height=420, legend=dict(orientation="h", y=-0.15))
    fig.show()
else:
    placeholder("Accuracy per category", "no category scored yet (a category is scored once all 200 entries are generated)")
'''

CHANGED = r'''
# per entry (scored categories only): passed = generated and not in the score file's failed list
STATES = [("both fail", "#E9E2F0"), ("broken (baseline passed)", C_BROKE), ("fixed (baseline failed)", C_FIX), ("both pass", "#4A2A66")]
scored = [c for c, v in CATS.items() if v["score"] and v["base"]]
if scored:
    rows, z, txt = [], [], []
    for c in scored:
        v = CATS[c]; ids = sorted(v["done_ids"], key=lambda i: int(i.rsplit("_", 1)[1]))
        st = [(i not in v["base"]["failed"]) * 1 + (i not in v["score"]["failed"]) * 2 for i in ids]   # 0 both fail, 1 broken, 2 fixed, 3 both pass
        n = Counter(st); rows.append([f"<b>{short(c)}</b>", f"<span style='color:{C_FIX}'><b>{n[2]}</b></span>", f"<span style='color:{C_BROKE}'><b>{n[1]}</b></span>",
                                      f"<b>{n[2] - n[1]:+d}</b>", n[3], n[0]])
        z.append([{0: 0, 1: 1, 2: 2, 3: 3}[s] for s in st]); txt.append([f"{i}: {STATES[s][0]}" for i, s in zip(ids, st)])
    display(HTML("<b>Entries that changed vs the baseline</b>" + table(rows, ["category", "fixed", "broken", "net", "both pass", "both fail"])))
    width = max(len(r) for r in z)
    cs = []
    for k, (_, col) in enumerate(STATES): cs += [[k / 4, col], [(k + 1) / 4, col]]
    fig = go.Figure(go.Heatmap(z=[r + [None] * (width - len(r)) for r in z], y=[short(c) for c in scored], x=list(range(width)),
                               customdata=[t + [""] * (width - len(t)) for t in txt], hovertemplate="%{customdata}<extra></extra>",
                               zmin=-0.5, zmax=3.5, colorscale=cs, xgap=1, ygap=4,
                               colorbar=dict(tickvals=[0, 1, 2, 3], ticktext=[s for s, _ in STATES], title="")))
    fig.update_layout(title="Every entry (hover for the id): green = fixed, red = broken, dark = both pass, pale = both fail",
                      xaxis_title="entry", height=160 + 70 * len(scored)); fig.show()
else:
    placeholder("Entries that changed", "no category scored yet")
'''

FAILS = r'''
# failure types: BFCL's error_type of each failed entry, model vs baseline (scored categories only)
scored = [c for c, v in CATS.items() if v["score"] and v["base"]]
if scored:
    for c in scored:
        a = Counter(CATS[c]["score"]["failed"].values()); b = Counter(CATS[c]["base"]["failed"].values())
        kinds = sorted(set(a) | set(b), key=lambda k: -(a[k] + b[k]))
        lab = [k.replace("multi_turn:", "") for k in kinds]
        fig = go.Figure()
        fig.add_bar(y=lab, x=[b[k] for k in kinds], name=CFG["baseline_label"], orientation="h", marker_color=C_BASE)
        fig.add_bar(y=lab, x=[a[k] for k in kinds], name=CFG["model_label"], orientation="h", marker_color=C_MAIN)
        fig.update_layout(title=f"{short(c)}: why entries failed (count of failed entries)", barmode="group", height=140 + 45 * len(kinds),
                          yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=-0.2))
        fig.show()
else:
    placeholder("Failure types", "no category scored yet")
'''

if OUT.exists():
    sys.exit(f"{OUT.name} exists: not overwriting (edit it by hand, or change the config JSON to show another run)")
nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"}
nb.cells = [md(INTRO), code(SETUP), md("## 1. Run info"), code(INFO), code(LIVE, ["skip-execution"]),
            md("---\n# Results"), code(GRAPH_SETUP),
            md("## 2. Accuracy vs the base model"), code(ACC),
            md("## 3. Which entries changed"), code(CHANGED),
            md("## 4. Why entries failed"), code(FAILS)]
nbf.write(nb, OUT)
print(f"created {OUT.name} ({len(nb.cells)} cells)")

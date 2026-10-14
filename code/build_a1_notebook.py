"""Creates a1_brake_eval.ipynb ONCE (refuses to overwrite it: the notebook is then edited by hand).
It reads only JSON: results/fs_env/a1_brake/a1_notebook_config.json (which run) and the run's JSONL, status.json and
progress.jsonl, which ./a1_updater.sh copies from the pod every 2 min.   ../.venv/bin/python build_a1_notebook.py"""
import sys
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parent / "a1_brake_eval.ipynb"
md = lambda s: nbf.v4.new_markdown_cell(s.strip("\n"))
code = lambda s, tags=None: nbf.v4.new_code_cell(s.strip("\n"), metadata={"tags": tags} if tags else {})

INTRO = """
# A1 "brake" tasks: does the model leave things as they are?

BFCL's biggest failure cause (A1): asked to start the engine, the model has to press the brake first (the car refuses
otherwise), and afterwards it **releases the brake**: an extra step nobody asked for. `fs_env/task_sets/a1_brake.json`
rebuilds that pattern in the file system, 10 multi-step tasks for each of six scenarios:

| | Scenario | What it has to set up | The tempting undo (wrong) |
|---|---|---|---|
| A1-I | unzip & take out | unzip the archive (reading inside it is refused) | re-zipping, deleting the extracted folder |
| A1-II | locked file | unlock it to write | locking it again |
| A1-III | maintenance mode | set `maintenance: on` to edit the folder | switching it back off |
| A1-IV | draft file | (a draft the user asked for) | deleting the draft after writing the final |
| A1-V | unzip withheld | turn 1 can't unzip: say so | working around it; later, undoing the extraction |
| A1-VI | conditional + archive | read the condition, unzip, copy the right file | deleting the extracted folder |

**How to read it:** section 2 is the pass rate per scenario; section 3 is the heatmap (every task: trials passed out of 4);
section 4 says *why* conversations failed, and whether that was the brake trap (the undo) or something else.

**Updates:** this notebook reads JSON only. `./a1_updater.sh start` copies the run's files from the pod every 2 min; re-run
the cells, or the live cell. The run is chosen in `results/fs_env/a1_brake/a1_notebook_config.json`.
"""

SETUP = r'''
import json, datetime
from pathlib import Path
from collections import Counter, defaultdict
from IPython.display import HTML, display
ROOT = next(p for p in (Path("."), Path("..")) if (p / "results/fs_env/a1_brake").exists())
CFG_PATH = ROOT / "results/fs_env/a1_brake/a1_notebook_config.json"
SCEN = ["A1-I", "A1-II", "A1-III", "A1-IV", "A1-V", "A1-VI"]
NAMES = {"A1-I": "unzip & take out", "A1-II": "locked file", "A1-III": "maintenance mode", "A1-IV": "draft file",
         "A1-V": "unzip withheld", "A1-VI": "conditional + archive"}
C_MAIN = "#7A4E9C"
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
TRAPS = {"locked the file again", "switched maintenance back off", "re-zipped the archive", "deleted the extracted folder",
         "deleted the draft", "worked around the missing unzip"}
def why(r):
    """(turn, reason) of a failed conversation's first failed turn; reasons in TRAPS are the brake trap (the undo)"""
    if r.get("error"): return None, "run error / timeout"
    for k, t in enumerate(r["turns"], 1):
        if t["passed"]: continue
        d = " | ".join(t.get("diffs") or [])
        if t["expect"] == "refuse":
            return k, "worked around the missing unzip" if t.get("diffs") else "didn't say it can't"
        if ".lock" in d and "extra file" in d: return k, "locked the file again"
        if "status.txt" in d: return k, "switched maintenance back off"
        if ".zip: content differs" in d: return k, "re-zipped the archive"
        if "missing directory" in d and r["cause"] in ("A1-I", "A1-V", "A1-VI"): return k, "deleted the extracted folder"
        if "missing file" in d and "draft" in d: return k, "deleted the draft"
        if d.startswith("reply") or not d: return k, "wrong or missing answer"
        return k, "other file mistake"
    return None, "?"
def load():
    cfg = json.loads(CFG_PATH.read_text()); run = ROOT / cfg["local_dir"]
    return cfg, read_json(run / "status.json", {}), read_jsonl(run / "progress.jsonl"), read_jsonl(run / cfg["results_file"])
CFG, ST, PROG, RES = load()
print(f"{CFG['run']}: {len(RES)} of {CFG['tasks'] * CFG['trials']} conversations")
'''

INFO = r'''
def info_html(cfg, st, prog, res):
    total = cfg["tasks"] * cfg["trials"]; done = len(res)
    t = lambda s: datetime.datetime.fromisoformat(s).replace(tzinfo=datetime.timezone.utc)
    start = t(cfg["started_utc"]); now = t(st["updated"]) if st.get("updated") else start
    el = (now - start).total_seconds(); price = cfg.get("price_per_hour", 0)
    fin = rate = None
    if len(prog) >= 2:
        last = prog[-1]; first = next((p for p in prog if (t(last["t"]) - t(p["t"])).total_seconds() <= 1800), prog[0])
        dt = (t(last["t"]) - t(first["t"])).total_seconds(); dn = last["done"] - first["done"]
        if dt > 0 and dn > 0:
            rate = dn / dt; fin = t(last["t"]) + datetime.timedelta(seconds=(total - done) / rate)
    stage = "finished" if st.get("finished") else "STOPPED (see the pipeline log)" if st.get("stopped") else \
            "running" if st.get("alive") else "setting up / not started"
    passed = sum(r["passed"] for r in res)
    o = [f"<h4>{cfg['model_label']} on {cfg['task_set']} &middot; <span style='color:{C_MAIN}'>{stage}</span>"
         f"{' &middot; updated ' + st['updated'].replace('T', ' ') + ' UTC' if st.get('updated') else ''}</h4>"]
    o.append(table([
        ["conversations", f"<b>{done} / {total}</b> ({cfg['tasks']} tasks x {cfg['trials']} trials)"],
        ["passed so far", f"<b>{passed} / {done}</b> ({passed / done:.1%})" if done else "-"],
        ["pace (last 30 min)", f"{rate * 3600:.0f} conversations/h" if rate else "-"],
        ["elapsed / estimated finish (your time)", f"{fmt_min(el)} / " + ("done" if st.get("finished") else
                                                    f"<b>{fin.astimezone().strftime('%H:%M')}</b>" if fin else "-")],
        ["GPU cost so far / estimated total", f"&#36;{el / 3600 * price:.2f} / " + (f"&#36;{(fin - start).total_seconds() / 3600 * price:.2f}" if fin else "-")
         + f" <span style='color:#888'>({cfg.get('price_note', '')})</span>"],
        ["settings", cfg.get("settings", "-")],
        ["pod pipeline (last lines)", "<br>".join(l.split("Z ", 1)[-1] for l in st.get("pipeline_log", [])[-4:]) or "-"]]))
    return "".join(o)
def scen_html(res):
    rows = []
    for s in SCEN:
        rs = [r for r in res if r["cause"] == s]; p = sum(r["passed"] for r in rs)
        traps = sum(1 for r in rs if not r["passed"] and why(r)[1] in TRAPS)
        rows.append([f"<b>{s}</b>", NAMES[s], f"<b>{p}/{len(rs)}</b>" + (f" ({p / len(rs):.0%})" if rs else ""),
                     f"{traps}" if rs else "-"])
    return "<b>Passed per scenario</b>" + table(rows, ["", "scenario", "passed", "fell for the trap (undo)"])
display(HTML(info_html(CFG, ST, PROG, RES)))
'''

LIVE = r'''
# live view: re-reads the JSON (kept current by ./a1_updater.sh) every 30 s until the run ends (interrupt to stop)
import time
from IPython.display import clear_output
try:
    while True:
        CFG, ST, PROG, RES = load(); clear_output(wait=True)
        display(HTML(info_html(CFG, ST, PROG, RES) + "<br>" + scen_html(RES)))
        if ST.get("finished") or ST.get("stopped"): break
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
CFG, ST, PROG, RES = load()
'''

PASS = r'''
display(HTML(scen_html(RES)))
if RES:
    by = {s: [r for r in RES if r["cause"] == s] for s in SCEN}
    xs = [f"{s}<br>{NAMES[s]}" for s in SCEN]; ys = [sum(r["passed"] for r in by[s]) / len(by[s]) if by[s] else None for s in SCEN]
    fig = go.Figure(go.Bar(x=xs, y=ys, marker_color=C_MAIN, text=[f"{sum(r['passed'] for r in by[s])}/{len(by[s])}" if by[s] else "" for s in SCEN],
                           textposition="outside"))
    fig.update_layout(title=f"{CFG['model_label']}: conversations passed per scenario", yaxis=dict(range=[0, 1.1], tickformat=".0%"), height=400)
    fig.show()
else:
    placeholder("Passed per scenario", "no conversation finished yet")
'''

HEAT = r'''
# every task: trials passed out of 4 (rows = scenarios, columns = the 10 tasks of each)
if RES:
    cell = defaultdict(list)
    for r in RES: cell[r["id"]].append(r["passed"])
    ids = {s: sorted({r["id"] for r in RES if r["cause"] == s} | set(), key=lambda i: int(i.split("_")[1])) for s in SCEN}
    width = 10
    z, txt, hov = [], [], []
    for s in SCEN[::-1]:
        zr, tr, hr = [], [], []
        for j in range(width):
            tid = next((i for i in ids[s] if int(i.split("_")[1]) == j + 1), None); v = cell.get(tid, [])
            zr.append(sum(v) / CFG["trials"] if v else None)
            tr.append(f"<b>{sum(v)}/{len(v)}</b><br>{'_'.join(tid.split('_')[2:])[:14]}" if v else "")
            hr.append(f"<b>{tid}</b><br>{sum(v)} of {len(v)} trials passed" + (f" (of {CFG['trials']})" if len(v) < CFG["trials"] else "") if v else "not run yet")
        z.append(zr); txt.append(tr); hov.append(hr)
    fig = go.Figure(go.Heatmap(z=z, x=[f"task {j + 1}" for j in range(width)], y=[f"{s} {NAMES[s]}" for s in SCEN[::-1]], text=txt,
                               texttemplate="%{text}", textfont=dict(size=10), customdata=hov, hovertemplate="%{customdata}<extra></extra>",
                               zmin=0, zmax=1, xgap=2, ygap=2, colorscale=[[0, "#F4EFF8"], [0.5, "#B48FCF"], [1, "#4A2A66"]],
                               colorbar=dict(title="trials<br>passed", tickvals=[0, 0.25, 0.5, 0.75, 1], ticktext=["0/4", "1/4", "2/4", "3/4", "4/4"])))
    fig.update_layout(title="Every task: trials passed out of 4 (dark = always right, pale = always wrong)", height=520, xaxis=dict(side="top"))
    fig.show()
else:
    placeholder("Every task", "no conversation finished yet")
'''

WHY = r'''
# why conversations failed: the first failed turn of each, and whether it was the brake trap (the undo) or something else
fails = [r for r in RES if not r["passed"]]
if fails:
    reasons = defaultdict(Counter)
    for r in fails: reasons[r["cause"]][why(r)[1]] += 1
    kinds = sorted({k for c in reasons.values() for k in c}, key=lambda k: (k not in TRAPS, k))
    rows = [[f"<b>{s}</b> {NAMES[s]}"] + [reasons[s][k] or "" for k in kinds] for s in SCEN]
    display(HTML("<b>Why conversations failed</b> (first failed turn; <i>italic</i> = the brake trap)" +
                 table(rows, ["scenario"] + [f"<i>{k}</i>" if k in TRAPS else k for k in kinds])))
    cols = ["#4A2A66", "#7A4E9C", "#B48FCF", "#2a78d6", "#86b6ef", "#1c5cab", "#eb6834", "#eda100", "#888"]
    fig = go.Figure()
    for i, k in enumerate(kinds):
        fig.add_bar(y=[f"{s} {NAMES[s]}" for s in SCEN], x=[reasons[s][k] for s in SCEN], name=k + (" (trap)" if k in TRAPS else ""),
                    orientation="h", marker_color=cols[i % len(cols)])
    fig.update_layout(barmode="stack", title="Failed conversations by reason", xaxis_title="failed conversations",
                      yaxis=dict(autorange="reversed"), height=420, legend=dict(orientation="h", y=-0.25))
    fig.show()
    turns = defaultdict(Counter)
    for r in fails: turns[r["cause"]][why(r)[0]] += 1
    display(HTML("<b>Which turn failed first</b>" + table([[s] + [turns[s][k] or "" for k in (1, 2, 3)] for s in SCEN], ["scenario", "turn 1", "turn 2", "turn 3"])))
else:
    placeholder("Why conversations failed", "no failed conversation yet")
'''

if OUT.exists():
    sys.exit(f"{OUT.name} exists: not overwriting (edit it by hand, or change the config JSON to show another run)")
nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"}
nb.cells = [md(INTRO), code(SETUP), md("## 1. Run info"), code(INFO), code(LIVE, ["skip-execution"]),
            md("---\n# Results"), code(GRAPH_SETUP),
            md("## 2. Passed per scenario"), code(PASS),
            md("## 3. Every task (heatmap)"), code(HEAT),
            md("## 4. Why conversations failed"), code(WHY)]
nbf.write(nb, OUT)
print(f"created {OUT.name} ({len(nb.cells)} cells)")

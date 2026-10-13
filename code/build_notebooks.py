"""Builds the two notebooks for the Qwen3.8-27B BFCL run from one cell list:
  bfcl_q38_27b_eval.ipynb             plotly (interactive, has a live-refresh cell)
  bfcl_q38_27b_eval_matplotlib.ipynb  matplotlib (static PNGs, renders on GitHub)
Both only READ results/bfcl_run.json (kept current by update_json.py / updater.sh).   python3 build_notebooks.py"""
import nbformat as nbf

md = lambda s: nbf.v4.new_markdown_cell(s.strip("\n"))
code = lambda s, tags=None: nbf.v4.new_code_cell(s.strip("\n"), metadata={"tags": tags} if tags else {})

INTRO = """
# Qwen3.8-27B on BFCL v4 (thinking on)

`Qwen/Qwen3.8-27B` used as is, with no fine-tuning, on the **Berkeley Function Calling Leaderboard (BFCL v4)**: every category except
web search (it needs a paid SerpAPI key) and format sensitivity (prompt-mode models only). That is about **4,900 test entries**:
non-live and live single-turn calls, multi-turn conversations with stateful tools, and the three memory backends.

The model runs in **FC mode** (native tool calls, parsed by vLLM's `qwen3_coder` parser) with **thinking on**, temperature 0.001,
32 requests at a time, served by vLLM 0.30.0 in bf16 on one A100 80GB. BFCL scores with its own checkers (AST matching, state
checks, answer matching), so no LLM judge is involved.

Data: `results/bfcl_run.json`, written every 30 s by `update_json.py` during the run (`./updater.sh start`).
"""

SETUP = r'''
import json, math, datetime
from pathlib import Path
from collections import Counter
import numpy as np
from IPython.display import HTML, display
P = next(p for p in (Path("results/bfcl_run.json"), Path("../results/bfcl_run.json")) if p.exists())
GROUP_COL = {"non-live": "#2F5D8A", "live": "#1E7A5C", "multi-turn": "#7A4E9C", "memory": "#E0883B"}
C_MAIN = "#7A4E9C"
def fmt_min(m):
    if m is None: return "-"
    m = int(round(m)); return f"{m} min" if m < 60 else f"{m // 60} h {m % 60:02d} min"
def hm(iso):   # "2026-10-11T08:09:41" -> "8:09" (24-hour, no leading zero)
    if not iso: return "-"
    t = datetime.datetime.fromisoformat(iso); return f"{t.hour}:{t.minute:02d}"
def table(rows, head=None):
    h = "".join(f"<th style='text-align:left;padding:2px 10px'>{x}</th>" for x in head) if head else ""
    b = "".join("<tr>" + "".join(f"<td style='padding:2px 10px'>{x}</td>" for x in r) + "</tr>" for r in rows)
    return f"<table style='border-collapse:collapse;font-size:13px'>{'<tr>' + h + '</tr>' if h else ''}{b}</table>"
def load():
    R = json.load(open(P))
    group_of = {c: g for g, cs in R["groups"].items() for c in cs}
    return R, group_of
def acc(c):   # (accuracy, binomial s.e., n) from BFCL's score file, or None if not evaluated yet
    s = c.get("score")
    if not s or not s.get("total"): return None
    p, n = s["accuracy"], s["total"]; return p, math.sqrt(p * (1 - p) / n), n
R, GROUP_OF = load()
print(f"stage: {R['stage']} | {R['plan']['done']}/{R['plan']['total']} entries | JSON updated {R['updated']}")
'''

INFO = r'''
def info_html(R):
    pl, pod, cfg, cats = R["plan"], R.get("pod", {}), R["config"], R["categories"]
    o = [f"<h4>JSON written {R['updated']} &middot; stage: <span style='color:{C_MAIN}'>{R['stage']}</span></h4>"]
    if not R.get("pod_reachable", True): o.append("<p style='color:#B3261E'><b>Pod unreachable</b>, showing the last known state.</p>")
    scored = [acc(c) for c in cats.values() if acc(c)]
    avg = sum(a[0] for a in scored) / len(scored) if scored else None
    official = (R.get("csv", {}).get("data_overall") or {}).get("Overall Acc")
    st = (pod.get("vllm_stats") or {})
    o.append("<b>Progress</b>" + table([
        ["entries done", f"{pl['done']} / {pl['total']} ({pl['done'] / pl['total']:.0%})"],
        ["categories scored", f"{len(scored)} / {len(cats)}"],
        ["BFCL overall accuracy", f"<b>{official}</b> (official weighting)" if official and R['stage'] == 'finished' else
            (f"{avg:.1%} plain average of the {len(scored)} scored categories (not BFCL's weighting)" if avg is not None else "-")],
        ["started", hm(pl.get("started_at"))], ["elapsed", fmt_min(pl["elapsed_h"] * 60) if pl.get("elapsed_h") else "-"],
        ["time left (rough)", fmt_min(pl.get("left_min")) if pl.get("left_min") else ("done" if pl["done"] >= pl["total"] else "measuring...")],
        ["estimated finish (local)", f"<b>{pl.get('done_at') or '-'}</b>"],
        ["BFCL's own estimate (assumes all entries are equally slow)", f"{fmt_min(pl['bfcl_eta']['left_min'])} left, done ≈ {pl['bfcl_eta']['done_at']} ({pl['bfcl_eta']['done']}/{pl['bfcl_eta']['total']} in this pass)" if pl.get("bfcl_eta") else "-"],
        ["pod cost so far / projected", f"${pl['cost_so_far']:.2f} / ${pl['cost_projected']:.2f}" if pl.get("cost_projected") else "-"],
        ["vLLM now", f"{st.get('running', 0):.0f} running, {st.get('waiting', 0):.0f} waiting, {st.get('gen_tps', 0):.0f} generated tokens/s" if st else "-"]]))
    rows = []
    for g, v in pl["groups"].items():
        rows.append([f"<span style='color:{GROUP_COL[g]}'>&#9632;</span> {g}", f"{v['done']} / {v['total']}",
                     f"{v['min_per_entry'] * 60:.1f} s" if v.get("min_per_entry") else "-", "measured" if v.get("measured") else "guess",
                     fmt_min(v["minutes"]) if v.get("minutes") else "-"])
    o.append(f"<b>Each group</b> (wall-clock seconds per entry, measured since the last restart, with {cfg.get('threads', '?')} running at once)" + table(rows, ["group", "entries", "s / entry", "", "time spent"]))
    o.append("<b>Settings</b>" + table([[k.replace("_", " "), v] for k, v in cfg.items()]))
    o.append("<b>Machine</b>" + table([["GPU (name, load, memory used, total)", pod.get("gpu") or "-"],
        ["disk / RAM disk", f"{pod.get('disk') or '-'} &middot; {pod.get('shm') or '-'}"], ["vLLM answering", "yes" if pod.get("vllm_up") else "no"],
        ["processes alive", ", ".join(k for k, v in (pod.get("procs") or {}).items() if v) or "none"],
        ["last BFCL log line", (pod.get("bfcl_log_tail") or "-").replace("<", "&lt;")[-150:]]]))
    o.append("<b>Recent events</b>" + table([[hm(e["time"]), e["event"].replace("<", "&lt;")] for e in R.get("events", [])[-8:]]))
    return "".join(o)
display(HTML(info_html(R)))
'''

LIVE = r'''
# live view: refreshes every 30 s until the run finishes (interrupt the kernel to stop)
import time
from IPython.display import clear_output
try:
    while True:
        R, GROUP_OF = load(); clear_output(wait=True); display(HTML(info_html(R)))
        if str(R["stage"]).startswith(("finished", "failed")): break
        time.sleep(30)
except KeyboardInterrupt:
    pass
'''

PREP = r'''
R, GROUP_OF = load()
CATS = R["categories"]; ORDER = [c for g in R["groups"].values() for c in g]
SCORED = [c for c in ORDER if acc(CATS[c])]
LABEL = {c: c.replace("_", " ") for c in ORDER}
def group_acc(g):   # plain average of the group's scored categories (BFCL averages categories, not entries)
    v = [acc(CATS[c])[0] for c in R["groups"][g] if acc(CATS[c])]
    return sum(v) / len(v) if v else None
FAILS = {c: CATS[c].get("fail_types", {}) for c in SCORED}
ERR_NAMES = [   # BFCL error_type (prefix:suffix) -> readable label; first match wins
    (":wrong_count", "wrong number of calls"), (":wrong_func_name", "wrong function"), ("missing_required", "missing required argument"),
    ("unexpected_param", "extra argument"), ("missing_optional", "missing optional argument"), ("value_error:string", "wrong value (text)"),
    ("value_error:list", "wrong value (list)"), ("value_error:dict", "wrong value (dict)"), ("value_error", "wrong value (number/other)"),
    ("type_error", "wrong argument type"), ("decoder_failed", "output not parseable"), ("irrelevance", "called a tool it shouldn't"),
    ("relevance", "no call when one was needed"), ("instance_state_mismatch", "wrong final state"), ("execution_response_mismatch", "wrong results returned"),
    ("empty_turn", "answered nothing in a turn"), ("force_terminated", "too many steps in a turn"), ("answer_not_found", "answer not found"),
]
def short_err(e):
    for key, name in ERR_NAMES:
        if key in e: return name
    return e.replace(":", " / ").replace("_", " ")
print(f"{len(SCORED)} of {len(ORDER)} categories scored so far")
'''

# ---------- plotly ----------
PL_IMPORT = r'''
import plotly.graph_objects as go, plotly.io as pio
from plotly.subplots import make_subplots
pio.renderers.default = "notebook_connected"; pio.templates.default = "plotly_white"
def placeholder(title, text="no data yet"):
    f = go.Figure(); f.add_annotation(text=text, showarrow=False, font=dict(size=16, color="#888"))
    f.update_layout(title=title, height=240, xaxis=dict(visible=False), yaxis=dict(visible=False)); f.show()
'''
PL_PROGRESS = r'''
fig = go.Figure()
cs = ORDER[::-1]
fig.add_bar(y=[LABEL[c] for c in cs], x=[CATS[c]["total"] for c in cs], orientation="h", marker_color="#E6E6E6", name="total", hoverinfo="skip")
fig.add_bar(y=[LABEL[c] for c in cs], x=[min(CATS[c]["done"], CATS[c]["total"]) for c in cs], orientation="h", marker_color=[GROUP_COL[GROUP_OF[c]] for c in cs], name="done",
            text=[f"{CATS[c]['done']}/{CATS[c]['total']}" for c in cs], textposition="outside")
fig.update_layout(title="Entries generated per category (grey = still to do)", barmode="overlay", height=620, showlegend=False, xaxis_title="entries")
fig.show()
'''
PL_ACC = r'''
if SCORED:
    cs = SCORED[::-1]; a = [acc(CATS[c]) for c in cs]
    fig = make_subplots(rows=1, cols=2, column_widths=[0.68, 0.32], subplot_titles=("Accuracy per category (±1 s.e.)", "Group averages (scored categories only)"))
    fig.add_bar(y=[f"{LABEL[c]} (n={x[2]})" for c, x in zip(cs, a)], x=[x[0] for x in a], error_x=dict(type="data", array=[x[1] for x in a]), orientation="h",
                marker_color=[GROUP_COL[GROUP_OF[c]] for c in cs], text=[f"{x[0]:.1%}" for x in a], textposition="outside", showlegend=False, row=1, col=1)
    gs = [g for g in R["groups"] if group_acc(g) is not None]
    fig.add_bar(x=gs, y=[group_acc(g) for g in gs], marker_color=[GROUP_COL[g] for g in gs], text=[f"{group_acc(g):.1%}" for g in gs], textposition="outside", showlegend=False, row=1, col=2)
    fig.update_xaxes(range=[0, 1.12], tickformat=".0%", row=1, col=1); fig.update_yaxes(range=[0, 1.1], tickformat=".0%", row=1, col=2)
    fig.update_layout(height=max(420, 26 * len(cs) + 120)); fig.show()
else:
    placeholder("Accuracy", "no category scored yet")
'''
PL_FAILS = r'''
if any(FAILS.values()):
    errs = Counter(); [errs.update({short_err(k): v}) for f in FAILS.values() for k, v in f.items()]
    top = [e for e, _ in errs.most_common(8)]
    cs = [c for c in SCORED if FAILS[c]][::-1]; fig = go.Figure()
    palette = ["#B3261E", "#E0883B", "#E8B04A", "#9DB8D9", "#2F5D8A", "#7A4E9C", "#1E7A5C", "#6B6B6B", "#CFCFCF"]
    for i, e in enumerate(top + ["other"]):
        vals = [sum(v for k, v in FAILS[c].items() if (short_err(k) == e) or (e == "other" and short_err(k) not in top)) for c in cs]
        if any(vals): fig.add_bar(y=[LABEL[c] for c in cs], x=vals, orientation="h", name=e, marker_color=palette[i % len(palette)])
    fig.update_layout(title="Why entries failed (BFCL error types)", barmode="stack", height=max(380, 26 * len(cs) + 160), xaxis_title="failed entries", legend=dict(orientation="h", y=-0.15))
    fig.show()
    print("most common:", ", ".join(f"{e} {n}" for e, n in errs.most_common(8)))
else:
    placeholder("Failures", "no failures recorded yet")
'''
PL_TOKENS = r'''
cs = [c for c in ORDER if CATS[c].get("out_tokens_each")]
if cs:
    fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38], subplot_titles=("Output tokens per entry, thinking included (log scale)", "Approximate time per category (minutes)"))
    for c in cs:
        fig.add_box(x=CATS[c]["out_tokens_each"], name=LABEL[c], marker_color=GROUP_COL[GROUP_OF[c]], boxmean=True, showlegend=False, orientation="h", row=1, col=1)
    tm = [c for c in ORDER if CATS[c].get("approx_min") is not None]
    fig.add_bar(y=[LABEL[c] for c in tm], x=[CATS[c]["approx_min"] for c in tm], orientation="h", marker_color=[GROUP_COL[GROUP_OF[c]] for c in tm],
                text=[fmt_min(CATS[c]["approx_min"]) for c in tm], textposition="outside", showlegend=False, row=1, col=2)
    fig.update_xaxes(type="log", row=1, col=1); fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=max(420, 26 * len(cs) + 120)); fig.show()
    tot_in = sum(CATS[c].get("input_tokens", 0) for c in ORDER); tot_out = sum(CATS[c].get("output_tokens", 0) for c in ORDER)
    print(f"tokens so far: {tot_in / 1e6:.1f} M in, {tot_out / 1e6:.2f} M out")
else:
    placeholder("Tokens", "no entries yet")
'''
PL_TABLE = r'''
rows = []
for c in ORDER:
    x, cc = acc(CATS[c]), CATS[c]
    rows.append([f"<span style='color:{GROUP_COL[GROUP_OF[c]]}'>&#9632;</span> {GROUP_OF[c]}", c, f"{cc['done']}/{cc['total']}",
                 f"{x[0]:.1%}" if x else "-", f"{cc['score']['correct']}/{cc['score']['total']}" if x else "-",
                 fmt_min(cc.get("approx_min")) if cc.get("approx_min") else "-", hm(cc.get("finished_at")), f"{cc.get('n_errors', 0)}"])
display(HTML(table(rows, ["group", "category", "generated", "accuracy", "correct", "≈ time", "finished at", "inference errors"])))
csv = R.get("csv", {}).get("data_overall")
if csv:
    keep = [k for k in csv if k.endswith("Acc") or k in ("Overall Acc", "Latency Mean (s)")]
    display(HTML("<b>BFCL's own summary (data_overall.csv)</b>" + table([[k, csv[k]] for k in keep])))
'''

# ---------- matplotlib ----------
MPL_IMPORT = r'''
import matplotlib.pyplot as plt, matplotlib.ticker as mtick
%matplotlib inline
plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#B8B8B8", "axes.grid": True, "grid.color": "#E6E6E6",
                     "grid.linewidth": 0.8, "axes.axisbelow": True, "lines.linewidth": 2, "legend.frameon": False, "xtick.color": "#555", "ytick.color": "#555"})
PCT = mtick.PercentFormatter(1)
def placeholder(title, text="no data yet"):
    fig, ax = plt.subplots(figsize=(10, 2.2)); ax.set_title(title); ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=14, color="#888", transform=ax.transAxes); plt.show()
'''
MPL_PROGRESS = r'''
cs = ORDER[::-1]; fig, ax = plt.subplots(figsize=(11, 6.5))
ax.barh([LABEL[c] for c in cs], [CATS[c]["total"] for c in cs], color="#E6E6E6")
b = ax.barh([LABEL[c] for c in cs], [min(CATS[c]["done"], CATS[c]["total"]) for c in cs], color=[GROUP_COL[GROUP_OF[c]] for c in cs])
for c, bar in zip(cs, b): ax.annotate(f"{CATS[c]['done']}/{CATS[c]['total']}", (CATS[c]["total"], bar.get_y() + bar.get_height() / 2), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color="#333")
ax.set_title("Entries generated per category (grey = still to do)"); ax.set_xlabel("entries"); ax.grid(axis="y", visible=False); fig.tight_layout(); plt.show()
'''
MPL_ACC = r'''
if SCORED:
    cs = SCORED[::-1]; a = [acc(CATS[c]) for c in cs]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, max(4.5, 0.32 * len(cs) + 1.5)), gridspec_kw=dict(width_ratios=[2.2, 1]))
    a1.barh([f"{LABEL[c]} (n={x[2]})" for c, x in zip(cs, a)], [x[0] for x in a], xerr=[x[1] for x in a], color=[GROUP_COL[GROUP_OF[c]] for c in cs], error_kw=dict(ecolor="#444", elinewidth=1, capsize=2))
    for i, x in enumerate(a): a1.annotate(f"{x[0]:.1%}", (x[0] + x[1], i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8)
    a1.set_xlim(0, 1.12); a1.xaxis.set_major_formatter(PCT); a1.set_title("Accuracy per category (±1 s.e.)"); a1.grid(axis="y", visible=False)
    gs = [g for g in R["groups"] if group_acc(g) is not None]
    b = a2.bar(gs, [group_acc(g) for g in gs], color=[GROUP_COL[g] for g in gs], width=0.6)
    for bar in b: a2.annotate(f"{bar.get_height():.1%}", (bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    a2.set_ylim(0, 1.1); a2.yaxis.set_major_formatter(PCT); a2.set_title("Group averages (scored categories only)")
    fig.tight_layout(); plt.show()
else:
    placeholder("Accuracy", "no category scored yet")
'''
MPL_FAILS = r'''
if any(FAILS.values()):
    errs = Counter(); [errs.update({short_err(k): v}) for f in FAILS.values() for k, v in f.items()]
    top = [e for e, _ in errs.most_common(8)]
    cs = [c for c in SCORED if FAILS[c]][::-1]; left = np.zeros(len(cs))
    palette = ["#B3261E", "#E0883B", "#E8B04A", "#9DB8D9", "#2F5D8A", "#7A4E9C", "#1E7A5C", "#6B6B6B", "#CFCFCF"]
    fig, ax = plt.subplots(figsize=(13, max(4, 0.32 * len(cs) + 1.8)))
    for i, e in enumerate(top + ["other"]):
        vals = np.array([sum(v for k, v in FAILS[c].items() if (short_err(k) == e) or (e == "other" and short_err(k) not in top)) for c in cs], float)
        if vals.any(): ax.barh([LABEL[c] for c in cs], vals, left=left, color=palette[i % len(palette)], label=e, edgecolor="white", linewidth=0.8); left += vals
    ax.set_title("Why entries failed (BFCL error types)"); ax.set_xlabel("failed entries"); ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4); fig.tight_layout(); plt.show()
    print("most common:", ", ".join(f"{e} {n}" for e, n in errs.most_common(8)))
else:
    placeholder("Failures", "no failures recorded yet")
'''
MPL_TOKENS = r'''
cs = [c for c in ORDER if CATS[c].get("out_tokens_each")]
if cs:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, max(4.5, 0.32 * len(cs) + 1.5)), gridspec_kw=dict(width_ratios=[1.6, 1]))
    bp = a1.boxplot([CATS[c]["out_tokens_each"] for c in cs], orientation="horizontal", tick_labels=[LABEL[c] for c in cs], widths=0.6, patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="#333", markersize=5), medianprops=dict(color="white", lw=2), flierprops=dict(marker="o", ms=2, alpha=.4))
    for patch, c in zip(bp["boxes"], cs): patch.set_facecolor(GROUP_COL[GROUP_OF[c]]); patch.set_edgecolor(GROUP_COL[GROUP_OF[c]])
    a1.set_xscale("log"); a1.invert_yaxis(); a1.set_title("Output tokens per entry, thinking included (log scale)"); a1.grid(axis="y", visible=False)
    tm = [c for c in ORDER if CATS[c].get("approx_min") is not None]
    b = a2.barh([LABEL[c] for c in tm], [CATS[c]["approx_min"] for c in tm], color=[GROUP_COL[GROUP_OF[c]] for c in tm])
    for c, bar in zip(tm, b): a2.annotate(fmt_min(CATS[c]["approx_min"]), (bar.get_width(), bar.get_y() + bar.get_height() / 2), xytext=(3, 0), textcoords="offset points", va="center", fontsize=8)
    a2.invert_yaxis(); a2.set_title("Approximate time per category (minutes)"); a2.set_xlabel("minutes"); a2.grid(axis="y", visible=False)
    fig.tight_layout(); plt.show()
    tot_in = sum(CATS[c].get("input_tokens", 0) for c in ORDER); tot_out = sum(CATS[c].get("output_tokens", 0) for c in ORDER)
    print(f"tokens so far: {tot_in / 1e6:.1f} M in, {tot_out / 1e6:.2f} M out")
else:
    placeholder("Tokens", "no entries yet")
'''
MPL_TABLE = PL_TABLE

SECTIONS = [("## Run info and estimated time", None), ("## Progress per category", "PROGRESS"),
            ("## Accuracy per category and per group\nBFCL averages categories (not entries), so a small category like live_parallel (16 entries) counts as much as live_multiple (1,053). Error bars are binomial standard errors.", "ACC"),
            ("## Why entries failed\nBFCL's own error types from the score files: wrong number of calls, wrong argument values, a call where none was needed, a missing state change in multi-turn, and so on.", "FAILS"),
            ("## How much it thinks, and roughly how long each category took\nAll categories ran in one overlapping pass, so there is no exact per-category timer. Real time per phase comes from the pod log (single-turn until its last category was scored, then multi-turn, then memory), split inside each phase by GPU work: output tokens + 0.1 × input tokens. Treat the bars as estimates.", "TOKENS"),
            ("## All categories in one table", "TABLE")]


def build(backend, live):
    G = globals(); pre = "PL_" if backend == "plotly" else "MPL_"
    num = lambda i, title: title.replace("## ", f"## {i}. ", 1)
    cells = [md(INTRO), code(SETUP), md(num(1, SECTIONS[0][0])), code(INFO)]
    if live: cells.append(code(LIVE, ["skip-execution"]))
    cells += [md("---\n# Graphs"), code(G[pre + "IMPORT"]), code(PREP)]
    for i, (title, key) in enumerate(SECTIONS[1:], start=2):
        cells += [md(num(i, title)), code(G[pre + key])]
    nb = nbf.v4.new_notebook(); nb.cells = cells
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return nb


if __name__ == "__main__":
    nbf.write(build("plotly", True), "bfcl_q38_27b_eval.ipynb")
    nbf.write(build("mpl", False), "bfcl_q38_27b_eval_matplotlib.ipynb")
    print("wrote bfcl_q38_27b_eval.ipynb and bfcl_q38_27b_eval_matplotlib.ipynb")

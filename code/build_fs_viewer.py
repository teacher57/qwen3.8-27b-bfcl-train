"""Builds one self-contained, offline HTML page for fs_env runs (fs_env.run JSONL), a tab per run: every conversation
grouped by task (failing tasks first), with the task's trap, each turn's expected behavior, gold calls, answer check,
the model's thinking / tool calls / results / replies, the per-turn verdict with differences, and the file trees.
    python3 build_fs_viewer.py run.jsonl          -> renders/fs_env_<run>.html
    python3 build_fs_viewer.py a.jsonl b.jsonl    -> renders/fs_env_all_runs.html (a tab per run)
    python3 build_fs_viewer.py --all              -> renders/fs_env_all_runs.html for every run in ALL_RUNS
    options: --task-set a1_brake.json --seed 13   tasks exactly as the run saw them (same random names; run.py --seed)
             --labels "tab 1|tab 2" --out name.html
Reads results/fs_env/fs-results/<run>.jsonl and fs_env/tasks/<id>.json (or the --task-set instances)."""
import json, os, sys
from collections import defaultdict
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results/fs_env/fs-results"
ALL_RUNS = {   # file -> tab label, in tab order
    "q38_fs90_t1_x4.jsonl": "A3–H · temp 1 · first 10 tasks × 4",
    "q38_A1A2_t1_x4_partial.jsonl": "A1 + part of A2 · temp 1 × 4 (stopped run)",
    "q38_fs_x4.jsonl": "11 originals · temp 0.001 × 4",
    "smoke.jsonl": "smoke test",
}
TASKS = HERE / "fs_env/tasks"
OUT = HERE / "renders"
EXPECT = {   # turn["expect"] -> what the model should do on that turn
    "act": "change the files to the target state",
    "answer": "answer from the files, change nothing",
    "ask": "ask for the missing information, change nothing",
    "refuse": "say it can't be done with the tools it has, change nothing",
}
CLIP = 8000


def clip(s, n=CLIP):
    s = s if isinstance(s, str) else json.dumps(s, ensure_ascii=False, indent=1)
    return s if len(s) <= n else s[:n] + f"\n... [{len(s) - n:,} more characters cut]"


def fmt_call(name, args):
    try: a = json.loads(args) if isinstance(args, str) else args
    except ValueError: return f"{name}({args})"
    if not isinstance(a, dict): return f"{name}({a})"
    return f"{name}({', '.join(f'{k}={json.dumps(v, ensure_ascii=False)}' for k, v in a.items())})"


INSTANCES = {}   # task id -> the task as the run saw it (--task-set/--seed), else the template from fs_env/tasks/


def task_spec(tid, user):
    if tid in INSTANCES:
        return INSTANCES[tid]
    p = TASKS / f"{tid}.json"
    if not p.exists(): return None
    return json.loads(p.read_text().replace("{user}", user))


def split_turns(messages):
    """messages -> [{user, steps:[{reasoning, calls, text, results}]}], a new turn at every user message"""
    turns = []
    for m in messages:
        role = m.get("role")
        if role == "user":
            turns.append({"user": m.get("content") or "", "steps": []})
        elif not turns:
            continue
        elif role == "assistant":
            turns[-1]["steps"].append({
                "reasoning": (m.get("reasoning_content") or "").strip(),
                "calls": [fmt_call(c["function"]["name"], c["function"]["arguments"]) for c in m.get("tool_calls") or []],
                "text": (m.get("content") or "").strip(), "results": []})
        elif role == "tool" and turns[-1]["steps"]:
            c = m.get("content", "")
            try: c = json.dumps(json.loads(c), ensure_ascii=False, indent=1)
            except (ValueError, TypeError): pass
            turns[-1]["steps"][-1]["results"].append(clip(c))
    return turns


def build(rows, key, label):
    items, groups = [], []
    by_task = defaultdict(list)
    for r in rows: by_task[r["id"]].append(r)
    for tid, rs in by_task.items():
        spec = task_spec(tid, rs[0]["user_name"]) or {}
        spec_turns = spec.get("turns") or []
        for r in sorted(rs, key=lambda x: x["trial"]):
            conv = split_turns(r.get("messages") or [])
            turns = []
            for k, t in enumerate(r["turns"]):
                st = spec_turns[k] if k < len(spec_turns) else {}
                turns.append({**(conv[k] if k < len(conv) else {"user": st.get("user", ""), "steps": []}),
                              "expect": t["expect"], "passed": t["passed"], "diffs": t.get("diffs") or [],
                              "note": t.get("note") or "", "withheld": st.get("withheld_tools") or [],
                              "gold": [fmt_call(c["name"], c["arguments"]) for c in st.get("gold_calls") or []],
                              "check": (st.get("answer_check") or {}).get("must_match") or []})
            items.append({"id": f"{tid} #{r['trial']}", "task": tid, "passed": r["passed"], "error": r.get("error"),
                          "seconds": round(r.get("seconds") or 0, 1), "user_name": r["user_name"], "turns": turns,
                          "n_steps": sum(len(t["steps"]) for t in turns),
                          "origin": clip(spec.get("origin", {}).get("tree")) if spec.get("origin") else "",
                          "final": clip(r.get("final_tree"))})
        n_ok = sum(x["passed"] for x in rs)
        groups.append({"key": tid, "label": tid, "n": len(rs), "ok": n_ok,
                       "about": spec.get("cause_title", ""), "trap": spec.get("trap", "")})
    groups.sort(key=lambda g: (g["ok"] == g["n"], g["key"]))   # failing tasks first
    n, ok = len(rows), sum(r["passed"] for r in rows)
    k_max = min(g["n"] for g in groups)
    pk = [sum(comb(g["ok"], k) / comb(g["n"], k) for g in groups) / len(groups) for k in range(1, k_max + 1)]
    sub = (f"{ok}/{n} conversations passed ({ok / n:.1%}) over {len(groups)} tasks. "
           + " · ".join(f"pass^{k + 1} {p:.1%}" for k, p in enumerate(pk))
           + f". {sum(1 for r in rows if r.get('error'))} ended with an error.")
    return {"key": key, "label": label, "acc": ok / n, "sub": sub, "items": items, "groups": groups}


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>fs_env Conversations</title>
<style>
:root{--bg:#fbfbfa;--panel:#fff;--ink:#16161a;--mut:#666;--line:#e3e2de;--soft:#f3f2ee;--user:#eaf1fb;--call:#eef7f1;--res:#f7f6f2;--bad:#b3261e;--good:#1e7a5c;--warn:#a8620a;--acc:#7a4e9c}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#121214;--panel:#1a1a1d;--ink:#ececf0;--mut:#9a9aa3;--line:#2c2c31;--soft:#222226;--user:#1b2533;--call:#18261f;--res:#202024;--bad:#ef6b62;--good:#4fc596;--warn:#e3a54f;--acc:#b892d8}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
header{padding:12px 18px 10px;border-bottom:1px solid var(--line);background:var(--panel)}h1{font-size:17px;margin:0}header p{margin:6px 0 0;color:var(--mut);font-size:13px}
.tabs{display:flex;gap:6px;margin-top:10px;flex-wrap:wrap}.tab{border:1px solid var(--line);background:var(--bg);color:var(--ink);border-radius:8px;padding:6px 12px;cursor:pointer;font:600 13px inherit}
.tab .pct{color:var(--mut);font-weight:500;margin-left:6px}.tab.on{background:var(--acc);border-color:var(--acc);color:#fff}.tab.on .pct{color:#fff;opacity:.85}
.wrap{display:flex;height:calc(100vh - var(--head,78px))}
aside{width:320px;min-width:240px;border-right:1px solid var(--line);overflow:auto;background:var(--panel)}
aside input{width:calc(100% - 20px);margin:10px;padding:7px 9px;border:1px solid var(--line);border-radius:7px;background:var(--bg);color:var(--ink)}
.grp>summary{cursor:pointer;padding:7px 12px;font-weight:600;font-size:13px;list-style:none;display:flex;justify-content:space-between;border-top:1px solid var(--line)}
.grp>summary .n{font-weight:600}.grp .about{padding:0 12px 6px;color:var(--mut);font-size:12px}
.it{padding:4px 12px 4px 22px;cursor:pointer;font:12px ui-monospace,Menlo,monospace;display:flex;justify-content:space-between}.it:hover{background:var(--soft)}.it.on{background:var(--soft);box-shadow:inset 3px 0 var(--acc)}
.it .meta{color:var(--mut)}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
main{flex:1;overflow:auto;padding:16px 22px 60px}
.badge{display:inline-block;padding:2px 9px;border-radius:99px;font-size:12px;font-weight:600;color:#fff}.ok{background:var(--good)}.ko{background:var(--bad)}
.tag{display:inline-block;padding:1px 8px;border-radius:99px;font-size:11.5px;font-weight:600;border:1px solid var(--line);color:var(--mut);margin-left:6px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:12px 0}
.err{border-left:4px solid var(--bad)}.turn{border-left:4px solid var(--good)}.turn.bad{border-left-color:var(--bad)}.trap{border-left:4px solid var(--warn)}
.turn h3{margin:0 0 8px;font-size:14px}.user{background:var(--user);padding:8px 10px;border-radius:8px;white-space:pre-wrap}
.step{margin:8px 0 0 10px;padding-left:10px;border-left:2px solid var(--line)}.lbl{color:var(--mut);font-size:12px}
.call{background:var(--call);font:12.5px ui-monospace,Menlo,monospace;padding:5px 8px;border-radius:6px;margin:3px 0;white-space:pre-wrap;word-break:break-word}
.text{background:var(--soft);padding:6px 9px;border-radius:6px;white-space:pre-wrap;margin:3px 0}
pre{background:var(--res);padding:6px 9px;border-radius:6px;white-space:pre-wrap;word-break:break-word;font:12px ui-monospace,Menlo,monospace;margin:3px 0;max-height:320px;overflow:auto}
details>summary{cursor:pointer;color:var(--mut);font-size:12.5px}.exp{border-left:3px solid var(--warn);padding-left:8px;margin:6px 0}
.exp .call{background:transparent;border:1px dashed var(--warn)}.empty{color:var(--warn);font-weight:600;font-size:12.5px}
.verdict{margin-top:10px;padding:6px 9px;border-radius:6px;background:var(--soft);font-size:12.5px}.verdict .d{font:12px ui-monospace,Menlo,monospace;white-space:pre-wrap;word-break:break-word;color:var(--bad);margin:2px 0}
.hint{color:var(--mut);font-size:12px;margin-top:4px}.cols{display:grid;grid-template-columns:1fr 1fr;gap:10px}
@media (max-width:760px){.wrap{flex-direction:column;height:auto}aside{width:auto;max-height:45vh}main{padding:12px 16px 40px}.cols{grid-template-columns:1fr}}
</style></head><body>
<header id="head"><h1>Qwen3.8-27B · fs_env conversations</h1><nav class="tabs" id="tabs"></nav><p id="sub"></p></header>
<div class="wrap"><aside><input id="q" placeholder="search tasks, user messages, calls…"><div id="nav"></div></aside><main id="main"></main></div>
<script>
const ALL = __DATA__;
const EXPECT = __EXPECT__;
const esc = s => String(s ?? "").replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
let D = null, byId = {}, G = {}, order = [], cur = null;
function tab(key, id) {
  D = ALL.find(t => t.key === key) || ALL[0];
  byId = Object.fromEntries(D.items.map(i => [i.id, i])); G = Object.fromEntries(D.groups.map(g => [g.key, g]));
  const tabs = document.getElementById("tabs"); tabs.style.display = ALL.length > 1 ? "" : "none";
  tabs.innerHTML = ALL.map(t => `<button class="tab${t === D ? " on" : ""}" data-k="${esc(t.key)}">${esc(t.label)}<span class="pct">${(t.acc * 100).toFixed(1)}%</span></button>`).join("");
  document.querySelectorAll(".tab").forEach(b => b.onclick = () => tab(b.dataset.k));
  document.getElementById("sub").textContent = D.sub + (ALL.length > 1 ? " Keys: ↑/↓ previous/next conversation, ←/→ switch run." : " Keys: ↑/↓ previous/next conversation.");
  document.documentElement.style.setProperty("--head", document.getElementById("head").offsetHeight + "px");
  document.getElementById("q").value = ""; cur = null; nav();
  show(byId[id] ? id : (D.items.find(i => !i.passed) || D.items[0]).id);
}
function nav(filter) {
  const f = (filter || "").toLowerCase(); order = [];
  document.getElementById("nav").innerHTML = D.groups.map(g => {
    const its = D.items.filter(i => i.task === g.key && (!f || JSON.stringify(i).toLowerCase().includes(f) || g.about.toLowerCase().includes(f)));
    if (f && !its.length) return "";
    order.push(...its.map(i => i.id));
    const col = g.ok === g.n ? "var(--good)" : g.ok ? "var(--warn)" : "var(--bad)";
    return `<details class="grp"${g.ok < g.n || f ? " open" : ""}><summary><span>${esc(g.label)}</span><span class="n" style="color:${col}">${g.ok}/${g.n}</span></summary>
      <div class="about">${esc(g.about)}</div>${its.map(i => `<div class="it${i.id === cur ? " on" : ""}" data-id="${esc(i.id)}"><span><span class="dot" style="background:${i.passed ? "var(--good)" : "var(--bad)"}"></span>trial ${i.id.split("#")[1]}</span><span class="meta">${i.n_steps} calls · ${i.seconds}s</span></div>`).join("")}</details>`;
  }).join("");
  document.querySelectorAll(".it").forEach(e => e.onclick = () => show(e.dataset.id));
}
function show(id) {
  cur = id; const i = byId[id], g = G[i.task]; history.replaceState(null, "", "#" + encodeURIComponent(D.key + "|" + id));
  document.querySelectorAll(".it").forEach(e => e.classList.toggle("on", e.dataset.id === id));
  let h = `<h2 style="margin:0 0 4px;font-size:18px">${esc(i.task)} <span class="tag">trial ${i.id.split("#")[1]}</span> <span class="badge ${i.passed ? "ok" : "ko"}">${i.passed ? "Passed" : "Failed"}</span></h2>
    <div class="hint">${i.turns.length} turns · ${i.n_steps} model calls · ${i.seconds}s · home folder /home/${esc(i.user_name)} · task ${g.ok}/${g.n} over all trials</div>`;
  if (g.about || g.trap) h += `<div class="card trap"><b>${esc(g.about)}</b>${g.trap ? `<div class="hint" style="font-size:13px">${esc(g.trap)}</div>` : ""}</div>`;
  if (i.error) h += `<div class="card err"><b>Run error:</b> ${esc(i.error)}</div>`;
  i.turns.forEach((t, k) => {
    h += `<div class="card turn${t.passed ? "" : " bad"}"><h3>Turn ${k + 1} <span class="tag">${esc(t.expect)}</span> <span class="badge ${t.passed ? "ok" : "ko"}">${t.passed ? "pass" : "fail"}</span></h3>
      <div class="user">${esc(t.user)}</div>
      <div class="exp"><div class="lbl" style="color:var(--warn)">expected: ${esc(EXPECT[t.expect] || t.expect)}${t.withheld.length ? ` · tools withheld this turn: ${esc(t.withheld.join(", "))}` : ""}</div>
        ${t.gold.length ? t.gold.map(c => `<div class="call">${esc(c)}</div>`).join("") : `<div class="empty">no calls that change files</div>`}
        ${t.check.length ? `<div class="hint">reply must match: <code>${t.check.map(esc).join("</code>, <code>")}</code></div>` : ""}</div>`;
    t.steps.forEach((s, j) => {
      h += `<div class="step"><div class="lbl">model call ${j + 1}</div>${s.reasoning ? `<details><summary>thinking (${s.reasoning.length.toLocaleString()} chars)</summary><div class="text">${esc(s.reasoning)}</div></details>` : ""}
        ${s.calls.map(c => `<div class="call">→ ${esc(c)}</div>`).join("")}${s.text ? `<div class="text">${esc(s.text)}</div>` : ""}
        ${s.results.map(r => `<details${r.length < 400 ? " open" : ""}><summary>tool result (${r.length.toLocaleString()} chars)</summary><pre>${esc(r)}</pre></details>`).join("")}</div>`;
    });
    if (!t.steps.length) h += `<div class="empty">no model call recorded in this turn</div>`;
    if (!t.passed || t.note) h += `<div class="verdict">${t.passed ? "" : "<b>Why it failed:</b>"}${t.note ? ` <i>${esc(t.note)}</i>` : ""}${t.diffs.map(d => `<div class="d">${esc(d)}</div>`).join("")}</div>`;
    h += `</div>`;
  });
  h += `<div class="card"><details><summary>file trees: start → end</summary><div class="cols"><div><div class="lbl">starting tree</div><pre>${esc(i.origin)}</pre></div><div><div class="lbl">final tree</div><pre>${esc(i.final)}</pre></div></div></details></div>`;
  const m = document.getElementById("main"); m.innerHTML = h; m.scrollTop = 0;
}
document.getElementById("q").oninput = e => nav(e.target.value);
document.onkeydown = e => {
  if (e.target.tagName === "INPUT" || !order.length) return;
  const k = order.indexOf(cur);
  if (e.key === "ArrowDown") { e.preventDefault(); show(order[Math.min(k + 1, order.length - 1)]); }
  if (e.key === "ArrowUp") { e.preventDefault(); show(order[Math.max(k - 1, 0)]); }
  const t = ALL.indexOf(D);
  if (e.key === "ArrowRight" && ALL.length > 1) tab(ALL[(t + 1) % ALL.length].key);
  if (e.key === "ArrowLeft" && ALL.length > 1) tab(ALL[(t - 1 + ALL.length) % ALL.length].key);
};
window.onresize = () => document.documentElement.style.setProperty("--head", document.getElementById("head").offsetHeight + "px");
const [hk, hid] = decodeURIComponent(location.hash.slice(1)).split("|");
tab(hk, hid);
</script></body></html>"""

if __name__ == "__main__":
    args = sys.argv[1:] or [str(RES / "q38_fs_x4.jsonl")]
    opt = {}
    for flag in ("--task-set", "--seed", "--labels", "--out"):
        if flag in args:
            i = args.index(flag); opt[flag] = args[i + 1]; del args[i:i + 2]
    if "--task-set" in opt:
        sys.path.insert(0, str(HERE))
        from fs_env.tasks import load_tasks, source_path
        seed = int(opt["--seed"]) if "--seed" in opt else None
        INSTANCES.update({t["id"]: t for t in load_tasks(source_path(opt["--task-set"]), seed=seed)})
    labels = opt["--labels"].split("|") if "--labels" in opt else []
    runs = [RES / f for f in ALL_RUNS] if args == ["--all"] else [Path(a) for a in args]
    data = []
    for i, run in enumerate(runs):
        rows = [json.loads(l) for l in open(run) if l.strip()]
        data.append(build(rows, run.stem, labels[i] if i < len(labels) else ALL_RUNS.get(run.name, run.stem)))
        print(f"{data[-1]['label']:44s} {data[-1]['sub']}")
    OUT.mkdir(exist_ok=True)
    path = OUT / (opt.get("--out") or (f"fs_env_{runs[0].stem}.html" if len(runs) == 1 else "fs_env_all_runs.html"))
    dump = lambda o: json.dumps(o, ensure_ascii=False).replace("</", "<\\/")
    path.write_text(TEMPLATE.replace("__EXPECT__", dump(EXPECT)).replace("__DATA__", dump(data)))
    print(f"wrote {path.relative_to(HERE)} ({os.path.getsize(path) / 1e6:.1f} MB)")

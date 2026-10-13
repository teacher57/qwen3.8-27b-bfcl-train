"""Builds one offline HTML page of the BFCL multi-turn DSAT (wrong-final-state failures) for two runs side by side:
a tab per root cause (A1-H, as in analysis/final_state_dsat.md), every conversation marked fixed / same / new / moved,
the first turn where each run went wrong (expected vs actual calls), and both full conversations with the thinking.
    python3 build_dsat_viewer.py      ->  renders/bfcl_dsat_base_vs_grpo20.html
Reads the DSAT labels (analysis/ and analysis/grpo20/) and the BFCL result/score files of both runs."""
import collections, json, os
from pathlib import Path

from build_mt_viewer import CATS, conversation, clip, CLIP_STATE

HERE = Path(__file__).resolve().parent
RUNS = {   # key: (label, DSAT folder, BFCL root, registry name)
    "base": ("Base Qwen3.8-27B", HERE / "analysis", HERE / "results/pod/bfcl-results", "qwen3.8-27b-FC"),
    "grpo": ("GRPO step 20", HERE / "analysis/grpo20", HERE / "results/fs_env/grpo/run_v1/bfcl", "qwen3.8-27b-grpo20-FC"),
}
OUT = HERE / "renders/bfcl_dsat_base_vs_grpo20.html"
CAUSES = {   # code: (label, what it looks like)
    "A1": ("A1 Released / re-pressed the brake", "Asked to start the engine, the model presses the brake, starts it, then releases or re-presses the brake: an extra step nobody asked for."),
    "A2": ("A2 Started navigation", "Asked to find a place, the model also sets the navigation to it."),
    "A3": ("A3 Wrote invented content into a file", "Asked only to create a file, the model also writes made-up content (templates, headings) into it."),
    "A4": ("A4 Other unrequested action", "Any other extra state change nobody asked for."),
    "B": ("B Rewrote the user's exact text", "The user gave exact text (tweet, ticket, message, file content); the model reworded or embellished it."),
    "C": ("C Small exact-value mistake", "Right action, slightly wrong value: format, unit, precision, prefix, wrong id or option."),
    "D": ("D Guessed instead of asking", "A needed value is missing (miss_param); the model invents it instead of asking."),
    "E": ("E Worked around a missing function", "A needed function is withheld (miss_func); the model reaches the goal another way instead of saying it can't."),
    "F": ("F Asked when it should act", "Everything needed was there, but the model stopped to confirm or ask."),
    "G": ("G Did nothing / just answered", "Action was expected; the model only answered or said it can't."),
    "H": ("H Other", "Skipped a required action, a different approach, or no call-level divergence found."),
}
STATUS = {   # status: (label, explanation)
    "fixed": ("Fixed by GRPO", "Base failed with this cause; GRPO passes."),
    "same": ("Same cause in both", "Both runs fail with this cause."),
    "moved_out": ("Base: this cause → GRPO: other failure", "Base failed with this cause; GRPO still fails, for another reason."),
    "moved_in": ("GRPO: this cause ← base: other failure", "GRPO fails with this cause; base failed for another reason."),
    "new": ("New in GRPO", "Base passed; GRPO fails with this cause."),
}


def load(key):
    label, dsat_dir, root, model = RUNS[key]
    dsat = {r["id"]: r for r in json.load(open(dsat_dir / "multi_turn_dsat.json"))}
    roots = json.load(open(dsat_dir / "final_state_root_causes.json"))
    res, fails, acc = {}, {}, {}
    for cat in CATS:
        for l in open(root / f"result/{model}/multi_turn/BFCL_v4_multi_turn_{cat}_result.json"):
            if l.strip(): r = json.loads(l); res[r["id"]] = r
        lines = [json.loads(l) for l in open(root / f"score/{model}/multi_turn/BFCL_v4_multi_turn_{cat}_score.json") if l.strip()]
        acc[cat] = lines[0]
        fails.update({f["id"]: f for f in lines[1:]})
    return {"dsat": dsat, "roots": roots, "res": res, "fails": fails, "acc": acc}


def side(run, i):
    """what one run did on entry i: outcome, root cause, DSAT divergence, BFCL error, full conversation"""
    f = run["fails"].get(i); d = run["dsat"].get(i, {}); rc = run["roots"].get(i)
    err = (f or {}).get("error") or {}
    out = {"passed": f is None, "bfcl_error": err.get("error_type", "").replace("multi_turn:", "") if f else "",
           "cause": rc["root_cause"] if rc else (f"(not a wrong-final-state failure: {err.get('error_type', '').replace('multi_turn:', '')})" if f else ""),
           "message": err.get("error_message", ""), "conv": conversation(run["res"][i]) if i in run["res"] else None}
    diffs = (err.get("details") or {}).get("differences")
    if diffs: out["differences"] = clip(diffs, CLIP_STATE)
    if d:
        out["dsat"] = {k: d.get(k) for k in ("cluster", "turn", "user_message", "expected", "model_calls", "model_text", "extra_actions",
                                               "missing_actions", "arg_diff", "missing_reads") if d.get(k)}
        if isinstance(out["dsat"].get("model_text"), str): out["dsat"]["model_text"] = clip(out["dsat"]["model_text"], 1500)
    return out


def main():
    A, B = load("base"), load("grpo")
    code = lambda rc: rc["root_cause"].split()[0]
    tabs, summary = [], []
    for c, (label, about) in CAUSES.items():
        ia = {i for i, v in A["roots"].items() if code(v) == c}; ib = {i for i, v in B["roots"].items() if code(v) == c}
        items = []
        for i in sorted(ia | ib, key=lambda x: (x.rsplit("_", 1)[0], int(x.rsplit("_", 1)[1]))):
            if i in ia and i in ib: st = "same"
            elif i in ia: st = "fixed" if i not in B["fails"] else "moved_out"
            else: st = "new" if i not in A["fails"] else "moved_in"
            expected = (A["fails"].get(i) or B["fails"].get(i) or {}).get("possible_answer")
            items.append({"id": i, "cat": i.replace("multi_turn_", "").rsplit("_", 1)[0], "status": st, "expected": expected,
                          "base": side(A, i), "grpo": side(B, i)})
        if not items: continue
        n = collections.Counter(x["status"] for x in items)
        tabs.append({"key": c, "label": label, "about": about, "na": len(ia), "nb": len(ib), "items": items,
                     "groups": [{"key": s, "label": STATUS[s][0], "about": STATUS[s][1], "n": n[s]} for s in STATUS if n[s]]})
        summary.append(f"{c}: {len(ia)} → {len(ib)} (fixed {n['fixed']}, new {n['new']})")
    data = {"tabs": tabs, "runs": {k: v[0] for k, v in RUNS.items()},
            "acc": {cat: [A["acc"][cat]["accuracy"], B["acc"][cat]["accuracy"]] for cat in CATS},
            "totals": [len(A["roots"]), len(B["roots"])]}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")))
    print("\n".join(summary)); print(f"wrote {OUT.relative_to(HERE)} ({os.path.getsize(OUT) / 1e6:.1f} MB)")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>BFCL Failure Causes</title>
<style>
:root{--bg:#fbfbfa;--panel:#fff;--ink:#16161a;--mut:#666;--line:#e3e2de;--soft:#f3f2ee;--user:#eaf1fb;--call:#eef7f1;--res:#f7f6f2;--bad:#b3261e;--good:#1e7a5c;--warn:#a8620a;--acc:#7a4e9c;--base:#c4521f;--grpo:#2a6fc9}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#121214;--panel:#1a1a1d;--ink:#ececf0;--mut:#9a9aa3;--line:#2c2c31;--soft:#222226;--user:#1b2533;--call:#18261f;--res:#202024;--bad:#ef6b62;--good:#4fc596;--warn:#e3a54f;--acc:#b892d8;--base:#e87a4a;--grpo:#5b9cec}}
:root[data-theme=dark]{--bg:#121214;--panel:#1a1a1d;--ink:#ececf0;--mut:#9a9aa3;--line:#2c2c31;--soft:#222226;--user:#1b2533;--call:#18261f;--res:#202024;--bad:#ef6b62;--good:#4fc596;--warn:#e3a54f;--acc:#b892d8;--base:#e87a4a;--grpo:#5b9cec}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
header{padding:12px 18px 10px;border-bottom:1px solid var(--line);background:var(--panel)}h1{font-size:17px;margin:0}header p{margin:6px 0 0;color:var(--mut);font-size:13px}
.tabs{display:flex;gap:6px;margin-top:10px;flex-wrap:wrap}.tab{border:1px solid var(--line);background:var(--bg);color:var(--ink);border-radius:8px;padding:5px 10px;cursor:pointer;font:600 13px inherit}
.tab .pct{color:var(--mut);font-weight:500;margin-left:6px}.tab.on{background:var(--acc);border-color:var(--acc);color:#fff}.tab.on .pct{color:#fff;opacity:.85}
.wrap{display:flex;height:calc(100vh - 150px)}
aside{width:310px;min-width:240px;border-right:1px solid var(--line);overflow:auto;background:var(--panel)}
.filters{padding:10px;display:flex;flex-direction:column;gap:6px}.filters input,.filters select{padding:6px 8px;border:1px solid var(--line);border-radius:7px;background:var(--bg);color:var(--ink);font:inherit;font-size:13px}
.grp>summary{cursor:pointer;padding:7px 12px;font-weight:600;font-size:13px;list-style:none;display:flex;justify-content:space-between;border-top:1px solid var(--line)}
.grp>summary .n{color:var(--mut);font-weight:500}.grp .about{padding:0 12px 6px;color:var(--mut);font-size:12px}
.it{padding:4px 12px 4px 22px;cursor:pointer;font:12px ui-monospace,Menlo,monospace;display:flex;justify-content:space-between}.it:hover{background:var(--soft)}.it.on{background:var(--soft);box-shadow:inset 3px 0 var(--acc)}
.it .meta{color:var(--mut)}
main{flex:1;overflow:auto;padding:16px 22px 60px}
.badge{display:inline-block;padding:2px 9px;border-radius:99px;font-size:12px;font-weight:600;color:#fff;margin-right:4px}.ok{background:var(--good)}.ko{background:var(--bad)}.st{background:var(--acc)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:12px 0}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:14px}.col h2{font-size:15px;margin:4px 0}.col.base h2{color:var(--base)}.col.grpo h2{color:var(--grpo)}
.col.base .card.first{border-left:4px solid var(--base)}.col.grpo .card.first{border-left:4px solid var(--grpo)}
.turn{border-left:4px solid var(--line)}.turn.div{border-left-color:var(--bad)}.turn h3{margin:0 0 8px;font-size:14px}
.user{background:var(--user);padding:8px 10px;border-radius:8px;white-space:pre-wrap}
.step{margin:8px 0 0 10px;padding-left:10px;border-left:2px solid var(--line)}.lbl{color:var(--mut);font-size:12px}
.call{background:var(--call);font:12.5px ui-monospace,Menlo,monospace;padding:5px 8px;border-radius:6px;margin:3px 0;white-space:pre-wrap;word-break:break-word}
.text{background:var(--soft);padding:6px 9px;border-radius:6px;white-space:pre-wrap;margin:3px 0}
pre{background:var(--res);padding:6px 9px;border-radius:6px;white-space:pre-wrap;word-break:break-word;font:12px ui-monospace,Menlo,monospace;margin:3px 0;max-height:320px;overflow:auto}
details>summary{cursor:pointer;color:var(--mut);font-size:12.5px}.exp{border-left:3px solid var(--warn);padding-left:8px;margin:6px 0}
.exp .call{background:transparent;border:1px dashed var(--warn)}.empty{color:var(--warn);font-weight:600;font-size:12.5px}
.note{color:var(--mut);font-size:12.5px;white-space:pre-wrap}.hint{color:var(--mut);font-size:12px;margin-top:4px}.pass{color:var(--good);font-weight:600}
@media (max-width:1100px){.cols{grid-template-columns:1fr}}
@media (max-width:760px){.wrap{flex-direction:column;height:auto}aside{width:auto;max-height:45vh}}
</style></head><body>
<header><h1>Qwen3.8-27B · BFCL multi-turn failure causes (A1–H) · base vs GRPO step 20</h1><nav class="tabs" id="tabs"></nav><p id="sub"></p></header>
<div class="wrap"><aside><div class="filters"><input id="q" placeholder="search ids, user messages, calls…">
<select id="cat"><option value="">all categories</option><option>base</option><option>miss_func</option><option>miss_param</option><option>long_context</option></select></div><div id="nav"></div></aside><main id="main"></main></div>
<script>
const ALL = __DATA__;
const esc = s => String(s ?? "").replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
let T = null, byId = {}, order = [], cur = null;
function tab(key, id) {
  T = ALL.tabs.find(t => t.key === key) || ALL.tabs[0];
  byId = Object.fromEntries(T.items.map(i => [i.id, i]));
  document.getElementById("tabs").innerHTML = ALL.tabs.map(t => `<button class="tab${t === T ? " on" : ""}" data-k="${t.key}">${esc(t.key)}<span class="pct">${t.na} → ${t.nb}</span></button>`).join("");
  document.querySelectorAll(".tab").forEach(b => b.onclick = () => tab(b.dataset.k));
  document.getElementById("sub").innerHTML = `<b>${esc(T.label)}</b>: ${esc(T.about)} <b>${ALL.runs.base}</b> ${T.na} → <b>${ALL.runs.grpo}</b> ${T.nb} failures. ` +
    `All wrong-final-state failures: ${ALL.totals[0]} → ${ALL.totals[1]} of 800. Tabs show base → GRPO counts. Keys: ↑/↓ entry, ←/→ cause.`;
  cur = null; nav(); const first = T.items[0]; show(byId[id] ? id : first.id);
}
function nav() {
  const f = document.getElementById("q").value.toLowerCase(), c = document.getElementById("cat").value; order = [];
  document.getElementById("nav").innerHTML = T.groups.map(g => {
    const its = T.items.filter(i => i.status === g.key && (!c || i.cat === c) && (!f || JSON.stringify(i).toLowerCase().includes(f)));
    order.push(...its.map(i => i.id));
    return `<details class="grp" open><summary><span>${esc(g.label)}</span><span class="n">${its.length}</span></summary><div class="about">${esc(g.about)}</div>` +
      its.map(i => `<div class="it${i.id === cur ? " on" : ""}" data-id="${i.id}"><span>${i.id.replace("multi_turn_", "")}</span><span class="meta">${i.cat}</span></div>`).join("") + `</details>`;
  }).join("");
  document.querySelectorAll(".it").forEach(e => e.onclick = () => show(e.dataset.id));
}
function first(s, name) {
  if (s.passed) return `<div class="card first"><span class="pass">✓ passed</span></div>`;
  const d = s.dsat || {};
  let h = `<div class="card first"><div><span class="badge ko">${esc(s.cause.split("  ").join(" "))}</span></div>`;
  if (d.turn) h += `<div class="hint">went wrong first in turn ${d.turn} · ${esc(d.cluster || "")}</div>`;
  if (d.user_message) h += `<div class="user" style="margin-top:6px">${esc(d.user_message)}</div>`;
  if (d.expected) h += `<div class="exp"><div class="lbl" style="color:var(--warn)">expected in that turn</div>${d.expected.length ? d.expected.map(c => `<div class="call">${esc(c)}</div>`).join("") : `<div class="empty">no calls: ask or say it can't</div>`}</div>`;
  if (d.model_calls) h += `<div class="lbl">the model called</div>${d.model_calls.length ? d.model_calls.map(c => `<div class="call">→ ${esc(c)}</div>`).join("") : `<div class="empty">nothing</div>`}`;
  if (d.extra_actions) h += `<div class="hint">extra actions: <b>${esc(d.extra_actions.join(", "))}</b></div>`;
  if (d.missing_actions) h += `<div class="hint">missing actions: <b>${esc(d.missing_actions.join(", "))}</b></div>`;
  if (d.arg_diff) h += `<details><summary>argument difference</summary><pre>${esc(JSON.stringify(d.arg_diff, null, 1))}</pre></details>`;
  if (d.model_text) h += `<details><summary>the model's reply in that turn</summary><div class="text">${esc(d.model_text)}</div></details>`;
  h += `<div class="hint">BFCL: ${esc(s.message)}</div>`;
  if (s.differences) h += `<details><summary>state differences (model vs expected)</summary><pre>${esc(s.differences)}</pre></details>`;
  return h + `</div>`;
}
function conv(s, expected) {
  if (!s.conv) return "";
  const dt = s.dsat && s.dsat.turn; let h = `<details${s.passed ? "" : ""}><summary>full conversation (${s.conv.turns.length} turns)</summary>`;
  if (s.conv.state.length) h += `<details><summary>starting state</summary>${s.conv.state.map(x => `<div class="lbl"><b>${esc(x.cls)}</b></div><pre>${esc(x.content)}</pre>`).join("")}</details>`;
  s.conv.turns.forEach((t, k) => {
    const exp = expected ? expected[k] : undefined;
    h += `<div class="card turn${dt === k + 1 ? " div" : ""}"><h3>Turn ${k + 1}${dt === k + 1 ? " · first wrong turn" : ""}</h3>${t.user.map(u => `<div class="user">${esc(u)}</div>`).join("")}${t.notes.map(n => `<div class="note">${esc(n)}</div>`).join("")}`;
    if (exp !== undefined) h += `<div class="exp"><div class="lbl" style="color:var(--warn)">expected calls</div>${exp.length ? exp.map(c => `<div class="call">${esc(c)}</div>`).join("") : `<div class="empty">none: ask or say it can't</div>`}</div>`;
    t.steps.forEach((st, j) => {
      h += `<div class="step"><div class="lbl">step ${j + 1}</div>${st.reasoning ? `<details><summary>thinking (${st.reasoning.length.toLocaleString()} chars)</summary><div class="text">${esc(st.reasoning)}</div></details>` : ""}` +
        st.calls.map(c => `<div class="call">→ ${esc(c)}</div>`).join("") + (st.text ? `<div class="text">${esc(st.text)}</div>` : "") +
        st.results.map(r => `<details><summary>tool result (${r.length.toLocaleString()} chars)</summary><pre>${esc(r)}</pre></details>`).join("") + `</div>`;
    });
    h += `</div>`;
  });
  return h + `</details>`;
}
function show(id) {
  cur = id; const i = byId[id]; history.replaceState(null, "", "#" + T.key + ":" + id);
  document.querySelectorAll(".it").forEach(e => e.classList.toggle("on", e.dataset.id === id));
  const g = T.groups.find(x => x.key === i.status);
  let h = `<h2 style="margin:0 0 4px;font-size:18px">${esc(id)} <span class="badge st">${esc(g.label)}</span></h2><div class="hint">${esc(g.about)} Category: ${esc(i.cat)}.</div><div class="cols">`;
  for (const k of ["base", "grpo"]) h += `<div class="col ${k}"><h2>${esc(ALL.runs[k])}</h2>${first(i[k])}${conv(i[k], i.expected)}</div>`;
  const m = document.getElementById("main"); m.innerHTML = h + `</div>`; m.scrollTop = 0;
}
document.getElementById("q").oninput = nav; document.getElementById("cat").onchange = nav;
document.onkeydown = e => {
  if (["INPUT", "SELECT"].includes(e.target.tagName) || !order.length) return;
  const k = order.indexOf(cur);
  if (e.key === "ArrowDown") { e.preventDefault(); show(order[Math.min(k + 1, order.length - 1)]); }
  if (e.key === "ArrowUp") { e.preventDefault(); show(order[Math.max(k - 1, 0)]); }
  const t = ALL.tabs.indexOf(T);
  if (e.key === "ArrowRight") tab(ALL.tabs[(t + 1) % ALL.tabs.length].key);
  if (e.key === "ArrowLeft") tab(ALL.tabs[(t - 1 + ALL.tabs.length) % ALL.tabs.length].key);
};
const [hk, hid] = decodeURIComponent(location.hash.slice(1)).split(":");
tab(hk, hid);
</script></body></html>"""

if __name__ == "__main__":
    main()

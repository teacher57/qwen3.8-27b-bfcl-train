"""Builds one self-contained, offline HTML page with a tab per BFCL multi-turn category (base, miss_func, miss_param,
long_context), every conversation grouped by outcome (passed, or BFCL's failure type):
    renders/bfcl_multi_turn.html
Reads results/pod/bfcl-results/{result,score}/qwen3.8-27b-FC/multi_turn/.   python3 build_mt_viewer.py"""
import html, json, os
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results/pod/bfcl-results/result/qwen3.8-27b-FC/multi_turn"
SCO = HERE / "results/pod/bfcl-results/score/qwen3.8-27b-FC/multi_turn"
OUT = HERE / "renders"
CATS = {"base": "Base", "miss_func": "Missing function", "miss_param": "Missing parameter", "long_context": "Long context"}
ABOUT = {
    "base": "Plain multi-step work over several user turns.",
    "miss_func": "A tool the user needs is withheld for one turn; the right move there is to say it can't be done yet (expected calls: none). It becomes available in a later turn.",
    "miss_param": "One turn leaves out information the task needs; the right move there is to ask the user (expected calls: none) instead of guessing.",
    "long_context": "Same kinds of tasks, but tools return very long outputs.",
}
GROUPS = {   # BFCL error_type -> (label, explanation)
    "passed": ("Passed", "Final state and results match the ground truth after every turn."),
    "multi_turn:instance_state_mismatch": ("Wrong final state", "After some turn, the simulated system (files, tweets, bookings...) differs from what the expected calls produce."),
    "multi_turn:execution_response_mismatch": ("Missing expected results", "The state may be right, but the model's calls didn't produce all the results the expected calls return in some turn."),
    "multi_turn:empty_turn_model_response": ("Did nothing in a turn", "The model made no tool call in a turn where calls were expected."),
    "multi_turn:force_terminated": ("Hit the 20-step limit", "The model kept calling tools until BFCL stopped the turn at 20 steps."),
    "multi_turn:inference_error": ("Context limit (our setup)", "The conversation grew past vLLM's --max-model-len 65536, so the request was refused. Not the model's fault."),
}
CLIP_TOOL, CLIP_STATE = 6000, 12000   # very long tool outputs/states are shortened to keep the page fast


def clip(s, n):
    s = s if isinstance(s, str) else json.dumps(s, ensure_ascii=False, indent=1)
    return s if len(s) <= n else s[:n] + f"\n... [{len(s) - n:,} more characters cut]"


def calls_text(content):
    if isinstance(content, list):
        out = []
        for c in content:
            if isinstance(c, dict):
                for name, args in c.items():
                    try: a = json.loads(args) if isinstance(args, str) else args
                    except ValueError: a = args
                    out.append(f"{name}({', '.join(f'{k}={json.dumps(v, ensure_ascii=False)}' for k, v in a.items()) if isinstance(a, dict) else a})")
            else: out.append(str(c))
        return out, None
    return [], (content or "").strip()


def conversation(r):
    """inference_log -> initial state + turns of {user, steps:[{reasoning, calls, text, results}]}"""
    log = r.get("inference_log") or []
    state = [s for s in (log[0] if log and isinstance(log[0], list) else []) if s.get("role") == "state_info"]
    turns = []
    for block in log[1:]:
        if not isinstance(block, dict): continue
        t = {"user": [], "steps": [], "notes": []}
        for key, msgs in block.items():
            if key == "begin_of_turn_query":
                t["user"] += [m.get("content", "") for m in msgs if m.get("role") == "user"]
                t["notes"] += [clip(m.get("content"), 600) for m in msgs if m.get("role") not in ("user",)]
            elif key.startswith("step_"):
                st = {"reasoning": "", "calls": [], "text": None, "results": []}
                for m in msgs:
                    role = m.get("role")
                    if role == "assistant":
                        st["reasoning"] = (m.get("reasoning_content") or "").strip()
                        st["calls"], st["text"] = calls_text(m.get("content"))
                    elif role == "tool":
                        st["results"].append(clip(m.get("content", ""), CLIP_TOOL))
                    elif role == "handler_log" and "Error" in str(m.get("content", "")):
                        st["results"].append("[handler] " + clip(m.get("content"), 600))
                t["steps"].append(st)
            elif isinstance(msgs, list):   # e.g. newly added functions in miss_func
                t["notes"] += [f"{key}: " + clip(m.get("content") if isinstance(m, dict) else m, 600) for m in msgs]
        turns.append(t)
    return {"state": [{"cls": s.get("class_name"), "content": clip(s.get("content"), CLIP_STATE)} for s in state], "turns": turns}


def build(cat):
    results = [json.loads(l) for l in open(RES / f"BFCL_v4_multi_turn_{cat}_result.json") if l.strip()]
    lines = [json.loads(l) for l in open(SCO / f"BFCL_v4_multi_turn_{cat}_score.json") if l.strip()]
    head, fails = lines[0], {f["id"]: f for f in lines[1:]}
    items = []
    for r in sorted(results, key=lambda x: int(x["id"].rsplit("_", 1)[1])):
        f = fails.get(r["id"])
        err = (f or {}).get("error") or {}
        gtype = err.get("error_type", "passed") if f else "passed"
        d = conversation(r)
        out_tok = sum(x for t in r.get("output_token_count") or [] for x in (t if isinstance(t, list) else [t]) if isinstance(x, (int, float)))
        d.update({"id": r["id"], "group": gtype, "n_steps": sum(len(t["steps"]) for t in d["turns"]), "out_tokens": int(out_tok)})
        if f:
            details = err.get("details") or {}
            d["error"] = {"message": err.get("error_message", ""), "differences": clip(details.get("differences"), CLIP_STATE) if details.get("differences") else None}
            d["expected"] = f.get("possible_answer")
            d["excluded"] = (f.get("prompt") or {}).get("excluded_function")
        items.append(d)
    groups = [(g, *GROUPS.get(g, (g, ""))) for g in [g for g in GROUPS if g != "passed"] + ["passed"]]   # failures first
    counts = {g: sum(1 for i in items if i["group"] == g) for g in GROUPS}
    tab = {"key": cat, "label": CATS[cat], "acc": head["accuracy"],
           "sub": f"{head['correct_count']}/{head['total_count']} passed ({head['accuracy']:.1%}). {ABOUT[cat]}",
           "items": items, "groups": [{"key": g, "label": l, "about": a, "n": counts.get(g, 0)} for g, l, a in groups if counts.get(g)]}
    return tab, counts


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>BFCL Multi-Turn Conversations</title>
<style>
:root{--bg:#fbfbfa;--panel:#fff;--ink:#16161a;--mut:#666;--line:#e3e2de;--soft:#f3f2ee;--user:#eaf1fb;--call:#eef7f1;--res:#f7f6f2;--bad:#b3261e;--good:#1e7a5c;--warn:#a8620a;--acc:#7a4e9c}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#121214;--panel:#1a1a1d;--ink:#ececf0;--mut:#9a9aa3;--line:#2c2c31;--soft:#222226;--user:#1b2533;--call:#18261f;--res:#202024;--bad:#ef6b62;--good:#4fc596;--warn:#e3a54f;--acc:#b892d8}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
header{padding:12px 18px 10px;border-bottom:1px solid var(--line);background:var(--panel)}h1{font-size:17px;margin:0}header p{margin:6px 0 0;color:var(--mut);font-size:13px}
.tabs{display:flex;gap:6px;margin-top:10px;flex-wrap:wrap}.tab{border:1px solid var(--line);background:var(--bg);color:var(--ink);border-radius:8px;padding:6px 12px;cursor:pointer;font:600 13px inherit}
.tab .pct{color:var(--mut);font-weight:500;margin-left:6px}.tab.on{background:var(--acc);border-color:var(--acc);color:#fff}.tab.on .pct{color:#fff;opacity:.85}
.wrap{display:flex;height:calc(100vh - 118px)}
aside{width:300px;min-width:240px;border-right:1px solid var(--line);overflow:auto;background:var(--panel)}
aside input{width:calc(100% - 20px);margin:10px;padding:7px 9px;border:1px solid var(--line);border-radius:7px;background:var(--bg);color:var(--ink)}
.grp>summary{cursor:pointer;padding:7px 12px;font-weight:600;font-size:13px;list-style:none;display:flex;justify-content:space-between;border-top:1px solid var(--line)}
.grp>summary .n{color:var(--mut);font-weight:500}.grp .about{padding:0 12px 6px;color:var(--mut);font-size:12px}
.it{padding:4px 12px 4px 22px;cursor:pointer;font:12px ui-monospace,Menlo,monospace;display:flex;justify-content:space-between}.it:hover{background:var(--soft)}.it.on{background:var(--soft);box-shadow:inset 3px 0 var(--acc)}
.it .meta{color:var(--mut)}
main{flex:1;overflow:auto;padding:16px 22px 60px}
.badge{display:inline-block;padding:2px 9px;border-radius:99px;font-size:12px;font-weight:600;color:#fff}.ok{background:var(--good)}.ko{background:var(--bad)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:12px 0}
.err{border-left:4px solid var(--bad)}.turn{border-left:4px solid var(--acc)}
.turn h3{margin:0 0 8px;font-size:14px}.user{background:var(--user);padding:8px 10px;border-radius:8px;white-space:pre-wrap}
.step{margin:8px 0 0 10px;padding-left:10px;border-left:2px solid var(--line)}.step .lbl{color:var(--mut);font-size:12px}
.call{background:var(--call);font:12.5px ui-monospace,Menlo,monospace;padding:5px 8px;border-radius:6px;margin:3px 0;white-space:pre-wrap;word-break:break-word}
.text{background:var(--soft);padding:6px 9px;border-radius:6px;white-space:pre-wrap;margin:3px 0}
pre{background:var(--res);padding:6px 9px;border-radius:6px;white-space:pre-wrap;word-break:break-word;font:12px ui-monospace,Menlo,monospace;margin:3px 0;max-height:320px;overflow:auto}
details>summary{cursor:pointer;color:var(--mut);font-size:12.5px}.exp{border-left:3px solid var(--warn);padding-left:8px;margin:6px 0}
.exp .call{background:transparent;border:1px dashed var(--warn)}.empty{color:var(--warn);font-weight:600;font-size:12.5px}
.note{color:var(--mut);font-size:12.5px;white-space:pre-wrap}.hint{color:var(--mut);font-size:12px;margin-top:4px}
@media (max-width:760px){.wrap{flex-direction:column;height:auto}aside{width:auto;max-height:45vh}}
</style></head><body>
<header><h1>Qwen3.8-27B · BFCL multi-turn conversations</h1><nav class="tabs" id="tabs"></nav><p id="sub"></p></header>
<div class="wrap"><aside><input id="q" placeholder="search ids, user messages, calls…"><div id="nav"></div></aside><main id="main"></main></div>
<script>
const ALL = __DATA__;
const esc = s => String(s ?? "").replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
let D = null, byId = {}, order = [], cur = null;
function tab(key, id) {
  D = ALL.find(t => t.key === key) || ALL[0];
  byId = Object.fromEntries(D.items.map(i => [i.id, i]));
  document.getElementById("tabs").innerHTML = ALL.map(t => `<button class="tab${t === D ? " on" : ""}" data-k="${t.key}">${esc(t.label)}<span class="pct">${(t.acc * 100).toFixed(1)}%</span></button>`).join("");
  document.querySelectorAll(".tab").forEach(b => b.onclick = () => tab(b.dataset.k));
  document.getElementById("sub").textContent = D.sub + " Keys: ↑/↓ previous/next conversation, ←/→ switch tab.";
  const q = document.getElementById("q"); q.value = "";
  cur = null; nav(); show(byId[id] ? id : (D.items.find(i => i.group !== "passed") || D.items[0]).id);
}
function nav(filter) {
  const f = (filter || "").toLowerCase(); order = [];
  document.getElementById("nav").innerHTML = D.groups.map(g => {
    const its = D.items.filter(i => i.group === g.key && (!f || JSON.stringify(i).toLowerCase().includes(f)));
    order.push(...its.map(i => i.id));
    return `<details class="grp"${g.key !== "passed" || f ? " open" : ""}><summary><span>${esc(g.label)}</span><span class="n">${its.length}${f ? " / " + g.n : ""}</span></summary>
      <div class="about">${esc(g.about)}</div>${its.map(i => `<div class="it${i.id === cur ? " on" : ""}" data-id="${i.id}"><span>${i.id.replace("multi_turn_", "")}</span><span class="meta">${i.n_steps} steps · ${i.out_tokens.toLocaleString()} tok</span></div>`).join("")}</details>`;
  }).join("");
  document.querySelectorAll(".it").forEach(e => e.onclick = () => show(e.dataset.id));
}
function show(id) {
  cur = id; const i = byId[id]; history.replaceState(null, "", "#" + D.key + ":" + id);
  document.querySelectorAll(".it").forEach(e => e.classList.toggle("on", e.dataset.id === id));
  const g = D.groups.find(x => x.key === i.group) || {label: i.group};
  let h = `<h2 style="margin:0 0 4px;font-size:18px">${esc(id)} <span class="badge ${i.group === "passed" ? "ok" : "ko"}">${esc(g.label)}</span></h2>
    <div class="hint">${i.turns.length} turns · ${i.n_steps} model steps · ${i.out_tokens.toLocaleString()} output tokens</div>`;
  if (i.error) h += `<div class="card err"><b>BFCL says:</b> ${esc(i.error.message)}${i.error.differences ? `<details open><summary>state differences (model vs expected)</summary><pre>${esc(i.error.differences)}</pre></details>` : ""}${i.excluded ? `<div class="hint">functions withheld at the start: ${esc(JSON.stringify(i.excluded))}</div>` : ""}</div>`;
  if (i.state.length) h += `<div class="card"><details><summary>starting state (${i.state.map(s => esc(s.cls)).join(", ")})</summary>${i.state.map(s => `<div class="lbl"><b>${esc(s.cls)}</b></div><pre>${esc(s.content)}</pre>`).join("")}</details></div>`;
  i.turns.forEach((t, k) => {
    const exp = i.expected ? i.expected[k] : undefined;
    h += `<div class="card turn"><h3>Turn ${k + 1}</h3>${t.user.map(u => `<div class="user">${esc(u)}</div>`).join("")}${t.notes.map(n => `<div class="note">${esc(n)}</div>`).join("")}`;
    if (exp !== undefined) h += `<div class="exp"><div class="lbl" style="color:var(--warn);font-size:12px">expected calls (ground truth)</div>${exp.length ? exp.map(c => `<div class="call">${esc(c)}</div>`).join("") : `<div class="empty">none: the model should ask or say it can't</div>`}</div>`;
    t.steps.forEach((s, j) => {
      h += `<div class="step"><div class="lbl">step ${j + 1}</div>${s.reasoning ? `<details><summary>thinking (${s.reasoning.length.toLocaleString()} chars)</summary><div class="text">${esc(s.reasoning)}</div></details>` : ""}
        ${s.calls.map(c => `<div class="call">→ ${esc(c)}</div>`).join("")}${s.text ? `<div class="text">${esc(s.text)}</div>` : ""}
        ${s.results.map(r => `<details${r.length < 400 ? " open" : ""}><summary>tool result (${r.length.toLocaleString()} chars)</summary><pre>${esc(r)}</pre></details>`).join("")}</div>`;
    });
    if (!t.steps.length) h += `<div class="empty">no model step recorded in this turn</div>`;
    h += `</div>`;
  });
  const m = document.getElementById("main"); m.innerHTML = h; m.scrollTop = 0;
}
document.getElementById("q").oninput = e => nav(e.target.value);
document.onkeydown = e => {
  if (e.target.tagName === "INPUT" || !order.length) return;
  const k = order.indexOf(cur);
  if (e.key === "ArrowDown") { e.preventDefault(); show(order[Math.min(k + 1, order.length - 1)]); }
  if (e.key === "ArrowUp") { e.preventDefault(); show(order[Math.max(k - 1, 0)]); }
  const t = ALL.indexOf(D);
  if (e.key === "ArrowRight") tab(ALL[(t + 1) % ALL.length].key);
  if (e.key === "ArrowLeft") tab(ALL[(t - 1 + ALL.length) % ALL.length].key);
};
const [hk, hid] = decodeURIComponent(location.hash.slice(1)).split(":");
tab(hk, hid);
</script></body></html>"""

if __name__ == "__main__":
    tabs = []
    for cat in CATS:
        t, counts = build(cat); tabs.append(t)
        print(f"{CATS[cat]:18s} {len(t['items'])} conversations |", ", ".join(f"{GROUPS[g][0]} {c}" for g, c in counts.items() if c))
    OUT.mkdir(exist_ok=True)
    path = OUT / "bfcl_multi_turn.html"
    path.write_text(TEMPLATE.replace("__DATA__", json.dumps(tabs, ensure_ascii=False).replace("</", "<\\/")))
    print(f"wrote {path.relative_to(HERE)} ({os.path.getsize(path) / 1e6:.1f} MB)")

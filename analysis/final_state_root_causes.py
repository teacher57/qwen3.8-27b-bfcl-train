"""Root causes for the multi-turn failures BFCL labels "wrong final state" (instance_state_mismatch, 199 conversations).
Needs analysis/multi_turn_dsat.json from dsat_multi_turn.py. Clusters were named after reading samples of each one.
    python3 analysis/final_state_root_causes.py  ->  analysis/final_state_root_causes.json + printed table"""
import collections, glob, json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
R = {r["id"]: r for r in json.load(open(HERE / "analysis/multi_turn_dsat.json"))}
TEXT = {"content", "description", "message", "body", "comment_content", "resolution", "details"}
CATS = ["base", "miss_func", "miss_param", "long_context"]

state = {}   # which system and which fields differ, from BFCL's own error details
for p in glob.glob(str(HERE / "results/pod/bfcl-results/score/qwen3.8-27b-FC/multi_turn/*_score.json")):
    for line in list(open(p))[1:]:
        f = json.loads(line)
        if f["error"]["error_type"] == "multi_turn:instance_state_mismatch":
            system = f["error"]["error_message"].split("Model instance for ")[1].split(" does not")[0]
            state[f["id"]] = {"system": system, "fields": sorted((f["error"]["details"].get("differences") or {}).keys())}


def root(r):
    c = r["cluster"]
    if c == "Did an extra, unrequested action":
        ex = set(r.get("extra_actions", []))
        if ex & {"releaseBrakePedal", "pressBrakePedal"}: return "A1 Extra step: released/re-pressed the brake after starting the engine"
        if "set_navigation" in ex: return "A2 Extra step: started navigation when only asked to find a place"
        if "echo" in ex: return "A3 Extra step: wrote invented content into a file it was only asked to create"
        return "A4 Extra step: other unrequested action (" + ",".join(sorted(ex)) + ")"
    if c == "Right action, wrong argument values":
        exp = [json.loads(a) for _, a in r["arg_diff"]["expected"]]; mod = [json.loads(a) for _, a in r["arg_diff"]["model"]]
        keys = {k for e in exp for m in mod for k in set(e) | set(m) if e.get(k) != m.get(k)}
        # free text only if the expected value is real text (a number written into `content` is an exact-value slip)
        long_text = any(isinstance(e.get(k), str) and len(e.get(k)) > 20 for e in exp for k in keys & TEXT)
        if long_text: return "B  Rewrote/embellished text the user gave (tweet, ticket, message, file content)"
        return "C  Small exact-value mistake (format, unit, precision, wrong id/option)"
    if c.startswith("Guessed"): return "D  Guessed missing information instead of asking (miss_param)"
    if c.startswith("Worked around"): return "E  Worked around a missing function instead of saying it can't (miss_func)"
    if c.startswith("Asked"): return "F  Asked for confirmation/clarification when it should have acted"
    if c.startswith("Took no action"): return "G  Did nothing / just answered when action was expected"
    return "H  " + c


out = {i: {**s, "root_cause": root(R[i]), "category": R[i]["category"], "turn": R[i]["turn"]} for i, s in state.items()}
(HERE / "analysis/final_state_root_causes.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
cnt = collections.Counter(v["root_cause"] for v in out.values())
tab = collections.Counter((v["root_cause"], v["category"]) for v in out.values())
print(f"{'root cause':88s} {'all':>4s} " + " ".join(f"{c[:10]:>10s}" for c in CATS))
for k in sorted(cnt): print(f"{k:88s} {cnt[k]:4d} " + " ".join(f"{tab[(k, c)]:10d}" for c in CATS))
print("total", sum(cnt.values()))

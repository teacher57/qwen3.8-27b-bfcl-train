"""Keeps results/bfcl_run.json current for the Qwen3.8-27B BFCL run on the pod.
Both notebooks (plotly and matplotlib) only READ that JSON.   python3 update_json.py [--once]
Reaches the pod through pod/rp.sh (RunPod SSH proxy); set RP_TARGET for a different pod."""
import datetime, json, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "bfcl_run.json")
RP = os.path.join(HERE, "pod", "rp.sh")

GROUPS = {  # BFCL leaderboard groups (web_search left out: needs a paid SerpAPI key)
    "non-live": ["simple_python", "simple_java", "simple_javascript", "parallel", "multiple", "parallel_multiple", "irrelevance"],
    "live": ["live_simple", "live_multiple", "live_parallel", "live_parallel_multiple", "live_irrelevance", "live_relevance"],
    "multi-turn": ["multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param", "multi_turn_long_context"],
    "memory": ["memory_kv", "memory_vector", "memory_rec_sum"],
}
CONFIG = {"benchmark": "BFCL v4 (gorilla 6ea5797), all categories except web_search and format_sensitivity",
          "model": "Qwen/Qwen3.8-27B (bf16, no LoRA), registry name qwen3.8-27b-FC",
          "mode": "FC (native tool calls through vLLM's qwen3_coder parser)", "thinking": "on (template default reasoning effort)",
          "temperature": 0.001, "max_tokens_per_call": 16384, "threads": 16, "max_model_len": 65536, "serving": "vLLM 0.30.0, 1x A100-SXM4-80GB",
          "pod_cost_per_hour": 1.80}


def local_iso(ts):
    return datetime.datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def clock(ts, now):
    """24-hour time without a leading zero (0:00-23:59), plus the day if it isn't today."""
    t, today = datetime.datetime.fromtimestamp(ts), datetime.datetime.fromtimestamp(now).date()
    hm = f"{t.hour}:{t.minute:02d}"
    days = (t.date() - today).days
    return hm if days == 0 else f"{hm} tomorrow" if days == 1 else f"{hm} on {t.strftime('%a %b %d')}"


def bfcl_eta(tail, now):
    """BFCL's own progress bar, e.g. '40/4518 [01:08<2:19:44, 1.87s/it]': its estimate assumes every entry takes as long."""
    m = re.search(r"(\d+)/(\d+) \[[\d:]+<([\d:]+)", tail or "")
    if not m: return None
    parts = [int(x) for x in m.group(3).split(":")]
    secs = sum(v * 60 ** i for i, v in enumerate(reversed(parts)))
    return {"done": int(m.group(1)), "total": int(m.group(2)), "left_min": secs / 60, "done_at": clock(now + secs, now)}


def remote():
    for _ in range(2):   # one retry: the proxy occasionally drops a session
        try:
            raw = subprocess.run([RP, "python3 /root/summarize_remote.py"], capture_output=True, text=True, timeout=120).stdout
            start = raw.find("{")
            if start >= 0:
                return json.loads(raw[start:])
        except Exception:
            pass
        time.sleep(5)
    return None


def build(prev):
    now = time.time()
    r = remote()
    if r is None:
        o = dict(prev or {}); o.update({"updated": local_iso(now), "pod_reachable": False}); return o
    cats = r["categories"]
    ev = []
    for line in r.get("pipeline_log", []):
        p = line.split(" ", 1)
        if len(p) == 2 and p[0].endswith("Z"):
            try: ev.append({"time": local_iso(datetime.datetime.fromisoformat(p[0].replace("Z", "+00:00")).timestamp()), "event": p[1][:200]})
            except ValueError: pass
    text = "\n".join(r.get("pipeline_log", []))
    total = sum(c["total"] for c in cats.values()); done = sum(min(c["done"], c["total"]) for c in cats.values())
    ts = lambda e: datetime.datetime.fromisoformat(e["time"]).timestamp()
    starts = [datetime.datetime.fromisoformat(e["time"]).timestamp() for e in ev if e["event"].startswith("run: start")]
    first_start, started = (starts[0], starts[-1]) if starts else (None, None)

    # Time per category: BFCL's latency field only measures time to the first streamed chunk, and all categories
    # run in one pass, so wall time between two updates is split over the categories that finished entries in it
    # (or, if none did, over the ones still in progress). Kept across updates in the JSON.
    prev_cats = (prev or {}).get("categories", {})
    prev_t = datetime.datetime.fromisoformat(prev["updated"]).timestamp() if prev and prev.get("updated") else None
    dt = min((now - prev_t) / 60, 2.0) if prev_t and started and now > started else 0.0   # cap: don't count time the updater was off
    restarted = not prev or (prev.get("plan", {}).get("run_started_at") != (local_iso(started) if started else None))
    for n, c in cats.items():
        pc = prev_cats.get(n, {})
        # timing restarts with every "run: start", counting only entries generated since then
        c["minutes"] = 0.0 if restarted else pc.get("minutes", 0.0)
        c["base_done"] = c["done"] if restarted else pc.get("base_done", 0)
    gains = {n: max(cats[n]["done"] - prev_cats.get(n, {}).get("done", 0), 0) for n in cats}
    active = [n for n, c in cats.items() if 0 < c["done"] < c["total"]]
    if dt and sum(gains.values()):
        for n, g in gains.items(): cats[n]["minutes"] += dt * g / sum(gains.values())
    elif dt and active:
        for n in active: cats[n]["minutes"] += dt / len(active)

    # Time left = remaining entries x measured minutes per entry, per group (a multi-turn entry takes far longer than a
    # single-turn one). Groups with too little data borrow a rough multiple of the single-turn speed.
    # Factors from the smoke test: a multi-turn entry makes ~32 calls (~3k output tokens; its ~200k input tokens are mostly
    # the same conversation re-sent and served from vLLM's prefix cache) and took ~2 min alone, about 3.5x a single-turn
    # entry's wall time at 32-way concurrency. Memory entries are short (their setup conversations run early, mixed in).
    GUESS_FACTOR = {"non-live": 1, "live": 1.3, "multi-turn": 3.5, "memory": 1.2}
    groups, left_min, base = {}, 0.0, None
    for g, names in GROUPS.items():
        gd = sum(min(cats[n]["done"], cats[n]["total"]) for n in names); gt = sum(cats[n]["total"] for n in names)
        mins = sum(cats[n]["minutes"] for n in names)
        new_done = sum(max(min(cats[n]["done"], cats[n]["total"]) - cats[n]["base_done"], 0) for n in names)
        per_entry = mins / new_done if new_done >= 10 and mins > 0 else None
        groups[g] = {"done": gd, "total": gt, "minutes": mins, "min_per_entry": per_entry, "measured": per_entry is not None}
    # Unmeasured groups borrow from a measured one, scaled by the rough factors; prefer the group measured most recently
    # (the last one in run order with data), since speed changes when concurrency or the mix of work changes.
    for g in reversed(list(GROUPS)):
        if groups[g]["measured"] and groups[g]["done"] < groups[g]["total"] or (groups[g]["measured"] and base is None):
            base = groups[g]["min_per_entry"] / GUESS_FACTOR[g]
            if groups[g]["done"] < groups[g]["total"]: break
    unknown = False
    for g, v in groups.items():
        if v["min_per_entry"] is None and base: v["min_per_entry"] = base * GUESS_FACTOR[g]
        if v["min_per_entry"] is not None: left_min += (v["total"] - v["done"]) * v["min_per_entry"]
        elif v["done"] < v["total"]: unknown = True   # work left with no speed at all: don't pretend it's free
    left_min = 0 if done >= total else (None if unknown else left_min)
    # When each category was first fully scored (eval_loop logs "scored: <cat>", checked every minute), in hours since
    # the first run start. Unlike "minutes" above, this is never reset by restarts, so it is what the timeline chart uses.
    for e in ev:
        if e["event"].startswith("scored: ") and first_start:
            n = e["event"].split(": ", 1)[1]
            if n in cats and not cats[n].get("finished_at"):
                cats[n]["finished_at"] = e["time"]
                cats[n]["finished_h"] = (ts(e) - first_start) / 3600
    # Approximate time per category. Everything ran in one overlapping pass, so no per-category timer exists.
    # Real wall time per phase comes from the log (a phase ends when its last category is scored; the next one starts
    # there), then each phase's time is split between its categories by GPU work: output tokens + 0.1 x input tokens
    # (re-read prompt tokens are much cheaper than generated ones; vLLM ran ~10x more prompt than output tokens/s).
    PHASES = [GROUPS["non-live"] + GROUPS["live"], GROUPS["multi-turn"], GROUPS["memory"]]
    phase_start = first_start
    for names in PHASES:
        ends = [datetime.datetime.fromisoformat(cats[n]["finished_at"]).timestamp() for n in names if cats[n].get("finished_at")]
        if not ends or len(ends) < len(names) or phase_start is None:
            break
        span = max(max(ends) - phase_start, 0) / 60
        work = {n: cats[n].get("output_tokens", 0) + 0.1 * cats[n].get("input_tokens", 0) for n in names}
        for n in names:
            cats[n]["approx_min"] = span * work[n] / sum(work.values()) if sum(work.values()) else None
        phase_start = max(ends)
    pace = done / ((now - first_start) / 60) if first_start and now > first_start + 60 and done else None
    elapsed_h = (now - first_start) / 3600 if first_start else 0
    plan = {"done": done, "total": total, "pace_per_min": pace, "left_min": left_min,
            "done_at": clock(now + left_min * 60, now) if left_min else None,
            "started_at": local_iso(first_start) if first_start else None, "elapsed_h": elapsed_h,
            "run_started_at": local_iso(started) if started else None, "bfcl_eta": bfcl_eta(r.get("bfcl_log_tail"), now),
            "cost_so_far": elapsed_h * CONFIG["pod_cost_per_hour"],
            "cost_projected": (elapsed_h + (left_min or 0) / 60) * CONFIG["pod_cost_per_hour"] if first_start else None,
            "groups": groups}

    procs = r.get("procs", {})
    if "RUN_DONE" in text: stage = "finished"
    elif procs.get("run") or done: stage = f"running ({done}/{total} entries)"
    elif "FAILED" in text: stage = "failed (see events)"
    elif r.get("vllm_up"): stage = "vLLM ready, waiting for the run to start"
    else: stage = "setting up the pod"

    pod = {k: r.get(k) for k in ("gpu", "disk", "shm", "vllm_up", "procs", "vllm_log_tail", "bfcl_log_tail", "vllm_stats")}
    return {"updated": local_iso(now), "pod_reachable": True, "stage": stage, "pod": pod, "config": CONFIG,
            "groups": GROUPS, "categories": cats, "csv": r.get("csv", {}), "plan": plan, "events": ev}


def write(o):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f: json.dump(o, f)
    os.replace(tmp, OUT)   # atomic, so a notebook never reads a half-written file


if __name__ == "__main__":
    prev = json.load(open(OUT)) if os.path.exists(OUT) else None
    while True:
        o = build(prev); write(o); prev = o; print(o.get("updated"), o.get("stage"), flush=True)
        if "--once" in sys.argv or str(o.get("stage", "")).startswith(("finished", "failed")): break
        time.sleep(30)

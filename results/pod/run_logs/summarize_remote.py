"""Runs ON THE POD: prints one compact JSON with pod state and a per-category summary of the BFCL run.
Read by ../update_json.py every 30 s. No model outputs are included, only counts, scores and timings."""
import csv, glob, json, os, re, subprocess, time

MODEL = "qwen3.8-27b-FC"
ROOT = os.environ.get("BFCL_PROJECT_ROOT", "/root/bfcl-results")
DATA = "/root/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data"
CATEGORIES = ["simple_python", "simple_java", "simple_javascript", "parallel", "multiple", "parallel_multiple", "irrelevance",
              "live_simple", "live_multiple", "live_parallel", "live_parallel_multiple", "live_irrelevance", "live_relevance",
              "multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param", "multi_turn_long_context",
              "memory_kv", "memory_vector", "memory_rec_sum"]


def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True, errors="replace").stdout.strip()


def jsonl(path):
    out = []
    try:
        with open(path, errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    try: out.append(json.loads(line))
                    except ValueError: pass   # line being written right now
    except OSError:
        pass
    return out


def find(kind, cat):
    # BFCL_v4_<cat>_result.json / _score.json, possibly in a group subfolder (non_live/, live/, multi_turn/, agentic/...)
    hits = glob.glob(f"{ROOT}/{kind}/{MODEL}/**/BFCL_v4_{cat}_{kind}.json", recursive=True)
    return max(hits, key=os.path.getmtime) if hits else None


def n_entries(cat):
    name = "memory" if cat.startswith("memory_") else cat
    return len(jsonl(f"{DATA}/BFCL_v4_{name}.json"))


def num(x):
    if isinstance(x, (int, float)): return x
    if isinstance(x, list):   # multi-turn entries store per-turn/per-step lists
        vals = [num(v) for v in x]; vals = [v for v in vals if v is not None]
        return sum(vals) if vals else None
    return None


out = {"now": time.time()}
out["pipeline_log"] = open("/root/pipeline.log").read().splitlines()[-120:] if os.path.exists("/root/pipeline.log") else []
out["gpu"] = sh("nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null")
out["disk"] = sh("df -h / | tail -1")
out["shm"] = sh("df -h /dev/shm | tail -1")
out["vllm_up"] = bool(sh("curl -s -m 3 localhost:8000/v1/models"))
out["procs"] = {"setup": bool(sh("pgrep -f '[s]etup.sh'")), "run": bool(sh("pgrep -f '[r]un.sh'")),
                "bfcl": bool(sh("pgrep -f '[b]fcl '")), "vllm": bool(sh("pgrep -f '[v]llm serve'"))}
out["vllm_log_tail"] = sh("tail -n 1 /root/vllm.log 2>/dev/null | cut -c1-200")
out["bfcl_log_tail"] = sh("tail -c 400 /root/bfcl_generate.log 2>/dev/null | tr '\\r' '\\n' | grep -v '^$' | tail -n 1 | cut -c1-200")
# vLLM's periodic stats line: throughput and running/waiting requests
m = re.findall(r"Avg prompt throughput: ([\d.]+).*?Avg generation throughput: ([\d.]+).*?Running: (\d+) reqs.*?Waiting: (\d+) reqs",
               sh("grep 'Avg generation throughput' /root/vllm.log 2>/dev/null | tail -n 1"))
out["vllm_stats"] = dict(zip(("prompt_tps", "gen_tps", "running", "waiting"), map(float, m[0]))) if m else None

cats = {}
for cat in CATEGORIES:
    c = {"total": n_entries(cat), "done": 0}
    rp = find("result", cat)
    if rp:
        rows = jsonl(rp)
        c["done"] = len(rows)
        c["result_mtime"] = os.path.getmtime(rp)
        lat = [num(r.get("latency")) for r in rows]; lat = [x for x in lat if x is not None]
        tin = [num(r.get("input_token_count")) for r in rows]; tin = [x for x in tin if x is not None]
        tout = [num(r.get("output_token_count")) for r in rows]; tout = [x for x in tout if x is not None]
        c["latency"] = lat; c["input_tokens"] = sum(tin); c["output_tokens"] = sum(tout)
        c["out_tokens_each"] = tout
        c["n_errors"] = sum(1 for r in rows if "Error during inference" in str(r.get("result", ""))[:200])
    sp = find("score", cat)
    if sp:
        s = jsonl(sp)
        if s and "accuracy" in s[0]:
            head = s[0]
            c["score"] = {"accuracy": head.get("accuracy"), "correct": head.get("correct_count"), "total": head.get("total_count")}
            fails = {}
            for r in s[1:]:
                err = r.get("error")
                k = r.get("error_type") or (err.get("error_type") if isinstance(err, dict) else None) or "unknown"
                fails[str(k)] = fails.get(str(k), 0) + 1
            c["fail_types"] = fails
            c["failed_ids"] = [r.get("id") for r in s[1:]][:400]
    cats[cat] = c
out["categories"] = cats

overall = {}
for name in ("data_overall", "data_non_live", "data_live", "data_multi_turn", "data_agentic"):
    p = f"{ROOT}/score/{name}.csv"
    if os.path.exists(p):
        lines = open(p).read().splitlines()
        if len(lines) >= 2:
            head = next(csv.reader([lines[0]]))
            for row in csv.reader(lines[1:]):
                if MODEL.replace("-FC", "") in ",".join(row).lower() or "qwen3.8" in ",".join(row).lower():
                    overall[name] = dict(zip(head, row))
out["csv"] = overall
print(json.dumps(out))

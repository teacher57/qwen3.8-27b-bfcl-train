"""Runs ON THE POD next to `bfcl generate`: scores each category as soon as all its entries are generated,
so accuracies show up during the run. Exits when generation has stopped and every complete category is scored.
    /root/bfcl-venv/bin/python /root/eval_loop.py "<space-separated categories>"   (started by run.sh)"""
import glob, os, subprocess, sys, time
from datetime import datetime, timezone

MODEL = "qwen3.8-27b-FC"
ROOT = os.environ.get("BFCL_PROJECT_ROOT", "/root/bfcl-results")
DATA = "/root/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data"
BFCL = "/root/bfcl-venv/bin/bfcl"
CATS = sys.argv[1].split()


def log(msg):
    with open("/root/pipeline.log", "a") as f:
        f.write(f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} {msg}\n")


def lines(path):
    try:
        with open(path, errors="replace") as f: return sum(1 for l in f if l.strip())
    except OSError: return 0


def total(cat):
    return lines(f"{DATA}/BFCL_v4_{'memory' if cat.startswith('memory_') else cat}.json")


def done(cat):
    hits = glob.glob(f"{ROOT}/result/{MODEL}/**/BFCL_v4_{cat}_result.json", recursive=True)
    return lines(hits[0]) if hits else 0


def generating():
    return subprocess.run("pgrep -f '[b]fcl generate'", shell=True, capture_output=True).returncode == 0


def score(cat):
    r = subprocess.run([BFCL, "evaluate", "--model", MODEL, "--test-category", cat], cwd="/root/gorilla/berkeley-function-call-leaderboard",
                       capture_output=True, text=True, errors="replace")
    with open("/root/bfcl_evaluate.log", "a") as f: f.write(r.stdout + r.stderr)
    log(f"scored: {cat}" if r.returncode == 0 else f"evaluate FAILED: {cat} (see /root/bfcl_evaluate.log)")


scored = set()
while True:
    gen = generating()
    for cat in CATS:
        if cat not in scored and done(cat) >= total(cat) > 0:
            score(cat); scored.add(cat)
    if not gen:   # generation over: one more pass above caught the last categories
        break
    time.sleep(60)
missing = [c for c in CATS if c not in scored]
if missing: log(f"not complete, not scored: {' '.join(missing)}")

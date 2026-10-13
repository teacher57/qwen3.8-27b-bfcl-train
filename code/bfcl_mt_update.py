"""Keeps the BFCL multi-turn run's local copy current for bfcl_eval_after_grpo_q38_27b.ipynb: every 2 min copies the model's BFCL
result/ and score/ files (and partial/score/: "passed so far", from pod/partial_score.sh) from the pod into
<local_dir>/, appends a progress line (entries and partial passes per category, time) to
<local_dir>/progress.jsonl and writes <local_dir>/status.json. Never touches the notebook. Stops after the pass that sees
"MULTITURN_DONE <model>" in the pod's pipeline log. Run named in results/fs_env/grpo/bfcl_mt_notebook_config.json.
  ./bfcl_mt_updater.sh start|stop|status      (FS_HOST/FS_PORT: direct SSH to the pod)"""
import base64, datetime, io, json, os, subprocess, tarfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "results/fs_env/grpo/bfcl_mt_notebook_config.json").read_text())
LOCAL, M = HERE / CFG["local_dir"], CFG["model"]
SSH = ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", "-i", os.path.expanduser("~/.ssh/id_ed25519"),
       "-p", os.environ.get("FS_PORT", "22"), f"root@{os.environ.get('FS_HOST', 'localhost')}"]
EVERY = 120


def log(msg): print(f"{datetime.datetime.now():%H:%M:%S} {msg}", flush=True)


def fetch():
    sh = (f'cd {CFG["pod_root"]} 2>/dev/null && P=$(ls -d result/{M} score/{M} partial/score/{M} 2>/dev/null); '
          f'[ -n "$P" ] && tar czf - $P | base64 -w0; echo; echo __LOG__; grep -E "multiturn|MULTITURN" /root/pipeline.log | tail -8; '
          f'pgrep -f "[b]fcl (generate|evaluate)" >/dev/null && echo ALIVE=yes || echo ALIVE=no')
    try: out = subprocess.run(SSH + [sh], capture_output=True, text=True, timeout=300, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired: return None
    if out.returncode != 0 or "__LOG__" not in out.stdout: return None
    blob, rest = out.stdout.split("__LOG__", 1)
    lines = rest.strip().splitlines()
    if not lines or not lines[-1].startswith("ALIVE="): return None
    if blob.strip():
        try:
            with tarfile.open(fileobj=io.BytesIO(base64.b64decode(blob.strip())), mode="r:gz") as tf:
                for m in tf.getmembers():
                    if not m.isfile(): continue
                    dest = LOCAL / m.name; dest.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dest.with_suffix(dest.suffix + ".tmp"); tmp.write_bytes(tf.extractfile(m).read()); os.replace(tmp, dest)
        except (ValueError, OSError, tarfile.TarError): return None
    return lines[:-1], lines[-1] == "ALIVE=yes"


def partial(cat):
    hits = list((LOCAL / "partial" / "score" / M).glob(f"**/BFCL_v4_{cat}_score.json"))
    try: d = json.loads(open(hits[0]).readline()) if hits else None
    except (OSError, ValueError): d = None
    return [d["correct_count"], d["total_count"]] if d else None


def count(cat):
    hits = list((LOCAL / "result" / M).glob(f"**/BFCL_v4_{cat}_result.json"))
    return sum(1 for l in open(hits[0], errors="replace") if l.strip()) if hits else 0


if __name__ == "__main__":
    LOCAL.mkdir(parents=True, exist_ok=True)
    log(f"start: {M} ({CFG['pod_root']} -> {LOCAL}), every {EVERY} s")
    while True:
        t0 = time.time(); r = fetch()
        if r is None: log("pod unreachable or transfer garbled, keeping the last copy")
        else:
            podlog, alive = r
            done = {c: count(c) for c in CFG["categories"]}
            finished = any(f"MULTITURN_DONE {M}" in l for l in podlog)
            now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
            passed = {c: partial(c) for c in CFG["categories"] if partial(c)}
            with (LOCAL / "progress.jsonl").open("a") as f: f.write(json.dumps({"t": now, "done": done, "passed": passed, "alive": alive}) + "\n")
            st = {"updated": now, "model": M, "done": done, "alive": alive, "finished": finished, "pod_log": podlog}
            tmp = LOCAL / "status.tmp"; tmp.write_text(json.dumps(st, indent=1)); tmp.replace(LOCAL / "status.json")
            log(f"{sum(done.values())} entries ({', '.join(f'{c[11:]} {n}' for c, n in done.items())}), bfcl {'running' if alive else 'idle'}")
            if finished: log("run finished"); break
        time.sleep(max(5, EVERY - (time.time() - t0)))

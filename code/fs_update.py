"""Keeps the fs_env run's local copy current: every 2 min copies the results JSONL and the runner's status from the pod
(results/fs_env/fs-results/<run>.jsonl + <run>.pod.json) and rebuilds renders/fs_env_<run>.html. It never touches the
notebook: fs_env_q38_27b_eval.ipynb is edited by hand and just reads these files when you run it (its live-view cell
re-reads them every 30 s). Stops after the pass that sees the run finished.   ./fs_updater.sh start|stop|status"""
import base64, datetime, gzip, json, os, subprocess, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = str(HERE.parent / ".venv/bin/python")
RES = HERE / "results/fs_env/fs-results"
# Which run and pod: set FS_RUN / FS_REMOTE_LOG / FS_HOST / FS_PORT (./fs_updater.sh passes the environment through).
NAME = os.environ.get("FS_RUN", "q38_fs90_t1_x4")
RUN, POD_JSON = RES / f"{NAME}.jsonl", RES / f"{NAME}.pod.json"
REMOTE_OUT, REMOTE_LOG = f"/root/fs-results/{NAME}.jsonl", os.environ.get("FS_REMOTE_LOG", "/root/fs90_t1_run.out")
SSH = ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", "-i", os.path.expanduser("~/.ssh/id_ed25519"),
       "-p", os.environ.get("FS_PORT", "19273"), f"root@{os.environ.get('FS_HOST', '64.247.196.128')}"]
# With FS_RP_TARGET set (RunPod proxy address, for when the pod's direct SSH port is unavailable), go through
# pod/rp.sh instead; the JSONL then travels gzipped + base64 so the proxy's terminal can't mangle it.
RP_TARGET = os.environ.get("FS_RP_TARGET")
EVERY = 120


def log(msg): print(f"{datetime.datetime.now():%H:%M:%S} {msg}", flush=True)


def fetch():
    """pod -> RUN and POD_JSON; returns the status dict, or None if the pod can't be reached"""
    data_cmd = f"gzip -c {REMOTE_OUT} 2>/dev/null | base64 -w0; echo" if RP_TARGET else f"cat {REMOTE_OUT} 2>/dev/null"
    sh = (f'{data_cmd}; echo "@@STATUS"; pid=$(pgrep -f "[p]ython -m fs_env.run" | head -1); '
          f'echo "alive=${{pid:+yes}}"; [ -n "$pid" ] && echo "elapsed_s=$(ps -o etimes= -p $pid | tr -d " ")"; '
          f'curl -s -m 5 127.0.0.1:8000/v1/models >/dev/null && echo vllm=up || echo vllm=down; '
          f'echo "gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"; tail -n 4 {REMOTE_LOG} | sed "s/^/log=/"')
    cmd = [str(HERE / "pod/rp.sh"), sh] if RP_TARGET else SSH + [sh]
    env = {**os.environ, "RP_TARGET": RP_TARGET} if RP_TARGET else None
    try: out = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=env)
    except subprocess.TimeoutExpired: return None
    if out.returncode != 0 or "@@STATUS" not in out.stdout: return None
    data, status = out.stdout.split("@@STATUS\n", 1)
    if RP_TARGET:
        try: data = gzip.decompress(base64.b64decode(data.strip())).decode() if data.strip() else ""
        except (ValueError, OSError): return None   # garbled transfer: keep the last copy, try again next pass
    if data.strip():
        tmp = RUN.with_suffix(".tmp"); tmp.write_text(data); os.replace(tmp, RUN)
    S = json.loads(POD_JSON.read_text()) if POD_JSON.exists() else {}
    S.update(fetched=datetime.datetime.now().isoformat(timespec="seconds"), alive=False, log=[])
    for line in status.splitlines():
        k, _, v = line.partition("=")
        if k == "log": S["log"].append(v)
        elif k == "alive": S["alive"] = v == "yes"
        elif k == "elapsed_s" and v.isdigit(): S["elapsed_s"] = int(v)   # kept from the last pass once the run has ended
        elif k in ("vllm", "gpu"): S[k] = v
    S["finished"] = not S["alive"] and any(l.startswith("passed ") for l in S["log"])
    tmp = POD_JSON.with_suffix(".tmp"); tmp.write_text(json.dumps(S, indent=1)); os.replace(tmp, POD_JSON)
    return S


def refresh():   # data only: the HTML viewer of the conversations (never the notebook)
    if not RUN.exists(): return
    rv = subprocess.run([PY, "build_fs_viewer.py", str(RUN)], capture_output=True, text=True, cwd=HERE)
    if rv.returncode: log("render FAILED: " + rv.stderr.strip().splitlines()[-1])


if __name__ == "__main__":
    log(f"start: {NAME}, every {EVERY} s")
    while True:
        t0 = time.time(); S = fetch()
        if S is None: log("pod unreachable, keeping the last copy")
        else:
            n = sum(1 for _ in open(RUN)) if RUN.exists() else 0
            refresh()
            log(f"{n} conversations, run {'alive' if S['alive'] else 'not running'}, vllm {S.get('vllm', '?')} ({time.time() - t0:.0f} s)")
            if not S["alive"]:
                log("run finished" if S["finished"] else "run process is gone without a final summary: stopping"); break
        time.sleep(max(5, EVERY - (time.time() - t0)))

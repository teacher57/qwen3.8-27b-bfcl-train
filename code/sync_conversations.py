"""Copies the BFCL run's conversations (result/, every model output with its reasoning) and score files (score/,
which entries failed and why) from the pod into results/pod/bfcl-results/ every 5 minutes.
Only files that changed since the last copy are sent. Stops by itself after the copy that follows RUN_DONE.
    python3 sync_conversations.py [--once]        (or ./conv_sync.sh start|stop|status)"""
import base64, hashlib, io, json, os, subprocess, sys, tarfile, time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
RP = os.path.join(HERE, "pod", "rp.sh")
REMOTE = "/root/bfcl-results"
LOCAL = os.path.join(HERE, "results", "pod", "bfcl-results")
STATE = os.path.join(HERE, "results", "pod", ".sync_state.json")
EVERY = 300


def rp(cmd, timeout=600):
    out = subprocess.run([RP, cmd], capture_output=True, text=True, timeout=timeout).stdout
    return out.strip()


def listing():
    # path size mtime for every file under result/ and score/
    raw = rp(f"cd {REMOTE} 2>/dev/null && find result score -type f ! -path '*/.file_locks/*' -printf '%p\\t%s\\t%T@\\n' 2>/dev/null; "
             "grep -c RUN_DONE /root/pipeline.log")
    lines = raw.splitlines()
    done = lines and lines[-1].strip().isdigit() and int(lines[-1]) > 0
    files = {}
    for l in lines[:-1] if lines else []:
        p = l.split("\t")
        if len(p) == 3: files[p[0]] = f"{p[1]}:{p[2]}"
    return files, done


def fetch(paths):
    # one tar.gz of just these files, built once on the pod, sent as base64 with its md5
    lst = " ".join(f"'{p}'" for p in paths)
    raw = rp(f"cd {REMOTE} && tar czf /tmp/sync.tgz {lst} && base64 -w0 /tmp/sync.tgz; echo; md5sum /tmp/sync.tgz | cut -d' ' -f1; rm -f /tmp/sync.tgz")
    b64, _, want = raw.rpartition("\n")
    data = base64.b64decode(b64.strip())
    if hashlib.md5(data).hexdigest() != want.strip():
        raise RuntimeError("checksum mismatch")
    with tarfile.open(fileobj=io.BytesIO(data)) as t:
        t.extractall(LOCAL, filter="data")
    return len(data)


def main():
    os.makedirs(LOCAL, exist_ok=True)
    state = json.load(open(STATE)) if os.path.exists(STATE) else {}
    while True:
        stamp = datetime.now().strftime("%H:%M:%S")
        try:
            files, run_done = listing()
            changed = sorted(p for p, sig in files.items() if state.get(p) != sig)
            if changed:
                size = 0
                for i in range(0, len(changed), 40):   # batches keep each transfer moderate
                    size += fetch(changed[i:i + 40])
                for p in changed: state[p] = files[p]
                with open(STATE, "w") as f: json.dump(state, f)
                print(f"{stamp} copied {len(changed)} changed file(s), {size / 1e6:.1f} MB compressed", flush=True)
            else:
                print(f"{stamp} no changes", flush=True)
        except Exception as e:
            print(f"{stamp} sync failed, will retry: {e}", flush=True)
            run_done = False
        if "--once" in sys.argv or run_done:
            if run_done: print(f"{stamp} run finished, final copy done", flush=True)
            break
        time.sleep(EVERY)


if __name__ == "__main__":
    main()

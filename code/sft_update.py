"""Keeps the SFT run's local copy current: every 2 min copies status.json, train_log.jsonl and probes.jsonl from the pod
into results/fs_env/sft/<run>/ (the run is named in results/fs_env/sft/sft_notebook_config.json). Never touches the
notebook: sft_q38_27b_train.ipynb reads these files when you run it. Stops after the pass that sees the run ended.
  ./sft_updater.sh start|stop|status      (FS_RP_TARGET=<pod proxy address> for the RunPod proxy, or FS_HOST/FS_PORT)"""
import base64, datetime, gzip, io, json, os, subprocess, tarfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "results/fs_env/sft/sft_notebook_config.json").read_text())
LOCAL = HERE / CFG["local_dir"]
REMOTE = CFG["remote_dir"]
RP_TARGET = os.environ.get("FS_RP_TARGET")
SSH = ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", "-i", os.path.expanduser("~/.ssh/id_ed25519"),
       "-p", os.environ.get("FS_PORT", "22"), f"root@{os.environ.get('FS_HOST', 'localhost')}"]
EVERY = 120
FILES = ["status.json", "train_log.jsonl", "probes.jsonl"]


def log(msg): print(f"{datetime.datetime.now():%H:%M:%S} {msg}", flush=True)


def fetch():
    """pod -> LOCAL; returns (status dict, training process alive) or None if the pod can't be reached"""
    sh = (f'cd {REMOTE} 2>/dev/null && tar czf - {" ".join(FILES)} 2>/dev/null | base64 -w0; echo; '
          f'pgrep -f "[f]s_env.sft_train" >/dev/null && echo ALIVE=yes || echo ALIVE=no')
    cmd = [str(HERE / "pod/rp.sh"), sh] if RP_TARGET else SSH + [sh]
    env = {**os.environ, "RP_TARGET": RP_TARGET} if RP_TARGET else None
    try: out = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=env)
    except subprocess.TimeoutExpired: return None
    lines = out.stdout.strip().splitlines()
    if out.returncode != 0 or not lines or not lines[-1].startswith("ALIVE="): return None
    alive = lines[-1] == "ALIVE=yes"
    blob = "".join(lines[:-1]).strip()
    if blob:
        try:
            with tarfile.open(fileobj=io.BytesIO(base64.b64decode(blob)), mode="r:gz") as tf:
                LOCAL.mkdir(parents=True, exist_ok=True)
                for m in tf.getmembers():
                    if m.name in FILES:
                        data = tf.extractfile(m).read()
                        tmp = LOCAL / (m.name + ".tmp"); tmp.write_bytes(data); os.replace(tmp, LOCAL / m.name)
        except (ValueError, OSError, tarfile.TarError): return None   # garbled transfer: keep the last copy
    S = json.loads((LOCAL / "status.json").read_text()) if (LOCAL / "status.json").exists() else {}
    return S, alive


if __name__ == "__main__":
    log(f"start: {CFG['run']} ({REMOTE} -> {LOCAL}), every {EVERY} s")
    while True:
        t0 = time.time(); r = fetch()
        if r is None: log("pod unreachable, keeping the last copy")
        else:
            S, alive = r
            log(f"stage {S.get('stage', '?')}, step {S.get('step', '-')}/{S.get('max_steps', '-')}, process {'alive' if alive else 'gone'}")
            if not alive and S.get("stage", "").startswith(("finished", "stopped")):
                log("run ended"); break
            if not alive and S:
                log("training process is gone without finishing: stopping"); break
        time.sleep(max(5, EVERY - (time.time() - t0)))

"""Keeps the GRPO run's local copy current: every 2 min copies status.json, grpo_log.jsonl, probes.jsonl and any NEW
rollouts/step_<n>.jsonl from the pod into results/fs_env/grpo/<run>/ (run named in results/fs_env/grpo/grpo_notebook_config.json).
Never touches the notebook (grpo_q38.ipynb reads these files). Stops after the pass that sees the run ended.
  ./grpo_updater.sh start|stop|status      (FS_RP_TARGET=<pod proxy address>, or FS_HOST/FS_PORT for direct SSH)"""
import base64, datetime, io, json, os, subprocess, tarfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "results/fs_env/grpo/grpo_notebook_config.json").read_text())
LOCAL, REMOTE = HERE / CFG["local_dir"], CFG["remote_dir"]
RP_TARGET = os.environ.get("FS_RP_TARGET")
SSH = ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", "-i", os.path.expanduser("~/.ssh/id_ed25519"),
       "-p", os.environ.get("FS_PORT", "22"), f"root@{os.environ.get('FS_HOST', 'localhost')}"]
EVERY = 120
FILES = ["status.json", "grpo_log.jsonl", "probes.jsonl"]


def log(msg): print(f"{datetime.datetime.now():%H:%M:%S} {msg}", flush=True)


def fetch():
    have = {p.name for p in (LOCAL / "rollouts").glob("step_*.jsonl")} if (LOCAL / "rollouts").exists() else set()
    skip = " ".join(f"--exclude=rollouts/{n}" for n in sorted(have))
    sh = (f'cd {REMOTE} 2>/dev/null && tar czf - {skip} {" ".join(FILES)} rollouts 2>/dev/null | base64 -w0; echo; '
          f'pgrep -f "[f]s_env.grpo_train" >/dev/null && echo ALIVE=yes || echo ALIVE=no')
    cmd = [str(HERE / "pod/rp.sh"), sh] if RP_TARGET else SSH + [sh]
    env = {**os.environ, "RP_TARGET": RP_TARGET} if RP_TARGET else None
    try: out = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    except subprocess.TimeoutExpired: return None
    lines = out.stdout.strip().splitlines()
    if out.returncode != 0 or not lines or not lines[-1].startswith("ALIVE="): return None
    blob = "".join(lines[:-1]).strip()
    if blob:
        try:
            with tarfile.open(fileobj=io.BytesIO(base64.b64decode(blob)), mode="r:gz") as tf:
                for m in tf.getmembers():
                    if not m.isfile(): continue
                    dest = LOCAL / m.name; dest.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dest.with_suffix(dest.suffix + ".tmp"); tmp.write_bytes(tf.extractfile(m).read()); os.replace(tmp, dest)
        except (ValueError, OSError, tarfile.TarError): return None
    S = json.loads((LOCAL / "status.json").read_text()) if (LOCAL / "status.json").exists() else {}
    return S, lines[-1] == "ALIVE=yes"


if __name__ == "__main__":
    log(f"start: {CFG['run']} ({REMOTE} -> {LOCAL}), every {EVERY} s")
    while True:
        t0 = time.time(); r = fetch()
        if r is None: log("pod unreachable or transfer garbled, keeping the last copy")
        else:
            S, alive = r
            log(f"stage {S.get('stage', '?')}, step {S.get('step', '-')}/{S.get('max_steps', '-')}, process {'alive' if alive else 'gone'}")
            if not alive and S:
                log("run ended" if str(S.get("stage", "")).startswith(("finished", "stopped")) else "training process is gone without finishing"); break
        time.sleep(max(5, EVERY - (time.time() - t0)))

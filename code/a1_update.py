"""Keeps the A1 brake eval's local copy current for a1_brake_eval.ipynb: every 2 min copies the run's JSONL and log from
the pod into <local_dir>/, appends a progress line to <local_dir>/progress.jsonl and writes <local_dir>/status.json
(conversations done, pipeline stage). Never touches the notebook. Stops after the pass that sees the run ended.
Run named in results/fs_env/a1_brake/a1_notebook_config.json; with "sync" set there, every listed run instead.   ./a1_updater.sh start|stop|status   (FS_HOST/FS_PORT)"""
import datetime, json, os, subprocess, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "results/fs_env/a1_brake/a1_notebook_config.json").read_text())
LOCAL, REMOTE = HERE / CFG["local_dir"], CFG["remote_dir"]
SSH = ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", "-i", os.path.expanduser("~/.ssh/id_ed25519"),
       "-p", os.environ.get("FS_PORT", "22"), f"root@{os.environ.get('FS_HOST', 'localhost')}"]
EVERY = 120


def log(msg): print(f"{datetime.datetime.now():%H:%M:%S} {msg}", flush=True)


def remote(cmd, binary=False):
    try:
        out = subprocess.run(SSH + [cmd], capture_output=True, timeout=300, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return None
    return out.stdout if out.returncode == 0 else None


def save(name, data: bytes):
    dest = LOCAL / name; tmp = dest.with_suffix(dest.suffix + ".tmp"); tmp.write_bytes(data); os.replace(tmp, dest)


def sync_runs(keys):
    """Several runs at once (CFG["sync"]): each run's results file and a status.json into its own local_dir.
    Returns True when every one of them is finished (DONE in its remote_dir)."""
    flags = remote(f"cat {CFG['runs'][keys[0]]['remote_dir']}/DONE 2>/dev/null; echo; pgrep -f '[f]s_env.run' >/dev/null && echo ALIVE || echo IDLE")
    if flags is None:
        log("pod unreachable, keeping the last copy"); return False
    done_flag, alive = (flags.decode(errors="replace").splitlines() + ["", ""])[:2]
    plog = (remote("tail -n 12 /root/pipeline.log 2>/dev/null; true") or b"").decode(errors="replace").splitlines()
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"); parts = []
    for key in keys:
        r = CFG["runs"][key]; local = HERE / r["local_dir"]; local.mkdir(parents=True, exist_ok=True)
        res = remote(f"cat {r['remote_dir']}/{r['results_file']} 2>/dev/null; true")
        if res is None: continue
        dest = local / r["results_file"]; tmp = dest.with_suffix(".tmp"); tmp.write_bytes(res); os.replace(tmp, dest)
        done = sum(1 for l in res.decode(errors="replace").splitlines() if l.strip())
        st = {"updated": now, "done": done, "total": r["tasks"] * r["trials"], "alive": alive == "ALIVE",
              "finished": done_flag.strip() == "DONE", "stopped": done_flag.strip() == "STOPPED", "pipeline_log": [l for l in plog if l.strip()]}
        (local / "status.json").write_text(json.dumps(st, indent=1))
        with (local / "progress.jsonl").open("a") as f: f.write(json.dumps({"t": now, "done": done}) + "\n")
        parts.append(f"{key} {done}/{st['total']}")
    log(", ".join(parts) + (" | FINISHED" if done_flag.strip() == "DONE" else " | STOPPED" if done_flag.strip() == "STOPPED" else ""))
    return done_flag.strip() in ("DONE", "STOPPED")


if __name__ == "__main__" and CFG.get("sync"):
    log(f"start: syncing {', '.join(CFG['sync'])} every {EVERY} s")
    while True:
        t0 = time.time()
        if sync_runs(CFG["sync"]):
            log("runs ended"); break
        time.sleep(max(5, EVERY - (time.time() - t0)))
    raise SystemExit(0)


if __name__ == "__main__":
    LOCAL.mkdir(parents=True, exist_ok=True)
    log(f"start: {CFG['run']} ({REMOTE} -> {LOCAL}), every {EVERY} s")
    while True:
        t0 = time.time()
        res = remote(f"cat {REMOTE}/{CFG['results_file']} 2>/dev/null; true")
        reps = {r["cause"]: remote(f"cat {REMOTE}/{r['file']} 2>/dev/null; true") for r in CFG.get("replacements", [])}
        meta = remote(f"cat {REMOTE}/DONE 2>/dev/null; echo ---; tail -n 12 /root/pipeline.log 2>/dev/null; echo ---; "
                      f"pgrep -f '[f]s_env.run' >/dev/null && echo ALIVE=yes || echo ALIVE=no")
        if res is None or meta is None:
            log("pod unreachable, keeping the last copy")
        else:
            done_flag, plog, alive = meta.decode(errors="replace").split("---\n")
            if reps and None not in reps.values():
                # re-run scenarios: keep the original run's lines for every other scenario, take these from the re-run
                (LOCAL / "invalid").mkdir(exist_ok=True); save("invalid/" + CFG["results_file"].replace(".jsonl", "_original.jsonl"), res)
                keep = [l for l in res.decode(errors="replace").splitlines() if l.strip() and json.loads(l)["cause"] not in reps]
                new = [l for b in reps.values() for l in b.decode(errors="replace").splitlines() if l.strip()]
                res = ("\n".join(keep + new) + "\n").encode()
            save(CFG["results_file"], res)
            done = sum(1 for l in res.decode(errors="replace").splitlines() if l.strip())
            now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
            st = {"updated": now, "done": done, "total": CFG["tasks"] * CFG["trials"], "alive": alive.strip() == "ALIVE=yes",
                  "finished": done_flag.strip() == "DONE" and alive.strip() != "ALIVE=yes" and done >= CFG["tasks"] * CFG["trials"],
                  "stopped": done_flag.strip() == "STOPPED",
                  "pipeline_log": [l for l in plog.splitlines() if l.strip()]}
            save("status.json", json.dumps(st, indent=1).encode())
            with (LOCAL / "progress.jsonl").open("a") as f: f.write(json.dumps({"t": now, "done": done}) + "\n")
            log(f"{done}/{st['total']} conversations, run {'alive' if st['alive'] else 'not running'}"
                f"{', FINISHED' if st['finished'] else ''}{', STOPPED' if st['stopped'] else ''}")
            if st["finished"] or st["stopped"]:
                log("run ended"); break
        time.sleep(max(5, EVERY - (time.time() - t0)))

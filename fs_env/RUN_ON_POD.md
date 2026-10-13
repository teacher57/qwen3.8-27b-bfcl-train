# Running the fs_env tasks on a RunPod GPU (4 trials each)

Runs the 380 file-system tasks (A3, D and E, which Qwen fails: 100 each = the hand-written original, 9 specified in
`make_tasks.py` and 90 from the pattern families in `task_families.py`; the other 8 causes: 10 each) against
`Qwen/Qwen3.8-27B` served by vLLM, **4 trials per task = 1,520 conversations**, with the same settings as the BFCL baseline: thinking on, temperature 0.001,
16,384 output tokens per model call, up to 20 model calls per turn, no system prompt.

Every load gives each task a random user name and 2–5 random background items in the home folder (`CLUTTER` in
`tasks.py`), so starting trees differ between runs; `--seed` makes them reproducible. Each turn passes only if the files match the turn's target state **and**, on turns that need a reply (answer / ask /
refuse), the model's final reply passes the turn's `answer_check`. A conversation passes only if every turn passes.

The pod part reuses the BFCL scripts in `bfcl-train/pod/` (`setup.sh`, `serve.sh`, `rp.sh`, `fetch_results.sh`).
All commands below run **on your Mac, from `bfcl-train/`**, unless marked otherwise.

## Time and cost

| Step | Time |
|---|---|
| Pod start + upload | ~5 min |
| `setup.sh` (vLLM install + 56 GB model download, in parallel) | ~4–6 min |
| vLLM startup | ~4 min |
| Smoke test (1 conversation) | ~3–8 min |
| **Full run: 380 tasks × 4 = 1,520 conversations**, 16 at a time | **~10–16 h** |
| or **originals only: 11 tasks × 4 = 44 conversations** (step 7b) | ~20–35 min |
| Fetch results | ~1 min |
| **Total, full run** | **~10.5–16.5 h → about $19–30** at $1.80/h (A100 80GB) |
| **Total, originals only** | ~40–60 min → about $1.20–1.80 |

The run estimate comes from the BFCL multi-turn run on the same model and GPU: about 60 s per model call at 16
threads. These tasks need about 6–8 model calls per conversation, so one conversation takes about 6–8 min, and 16 run
at a time: 1,520 conversations are about 95 rounds (~10–16 h), 44 are about 3 rounds. **What can make it slower:** a
thinking loop runs until the 16k-token cap (~10 min at ~28 tokens/s), holding one of the 16 slots. Check progress
every half hour (step 7): the JSONL should grow by roughly 50–70 lines per 30 min. If it's well below that, look at the
log before letting it continue.

**The pod bills until you terminate it.** Don't stop after fetching the results; terminate it (step 9).

## 0. Before starting (Mac)

```bash
cd ~/claude_trusted/bfcl-train
../.venv/bin/python -m fs_env.validate        # must end with "380/380 tasks valid"
../.venv/bin/python -m fs_env.test_fs_env     # must end with "12 tests passed"
```

Your SSH public key (`~/.ssh/id_ed25519.pub`) must be in RunPod → Settings → SSH Public Keys (it was for the BFCL run).

## 1. Create the pod (RunPod console)

- **GPU:** 1× A100 SXM 80GB (or H100 80GB).
- **Template:** the same PyTorch/CUDA template as the BFCL run. `setup.sh` needs `uv`; if the pod doesn't have it,
  run `pip install uv` on the pod first.
- **Container disk:** 50 GB is enough. The model doesn't go there: it goes to the RAM disk `/dev/shm` (~117 GB on these
  pods). **If the pod restarts, the model is gone**, and `setup.sh` has to download it again.
- No network volume or exposed ports are needed. Everything talks to vLLM on the pod's own `127.0.0.1:8000`.

## 2. Connect

Copy the proxy SSH address from the pod's **Connect** panel (the `<pod-id>-<n>@ssh.runpod.io` part):

```bash
export RP_TARGET=<pod-id>-<n>@ssh.runpod.io
pod/rp.sh 'nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; df -h /dev/shm | tail -1'
```

Check the **driver version**:
- **580.x** → go on, `setup.sh` installs the right vLLM wheel (`vllm==0.30.0`, CUDA 13).
- **570.x** → the default wheel won't run. Use the `+cu129` wheel with `--torch-backend=cu129`, then uninstall
  `torchcodec`, as in `../tau-train/qwen3.8-27b/pod/setup.sh` (edit the vLLM line of `pod/setup.sh` before uploading).

`rp.sh` uses the proxy's interactive shell (no `scp`, no `ssh host cmd`), so each call takes a few seconds.

## 3. Set up vLLM and the model

```bash
for f in setup.sh serve.sh; do pod/rp.sh --put pod/$f /root/$f; done
pod/rp.sh 'setsid nohup /root/setup.sh > /root/setup.out 2>&1 < /dev/null &'
pod/rp.sh 'tail -n 3 /root/pipeline.log'      # repeat until "SETUP_DONE (model 52G)" or similar (~4–6 min)
```

`setup.sh` also installs BFCL (the gorilla repo) in its own venv. These tasks don't need it, but it runs in parallel
and doesn't slow anything down. If the log says `FAILED: …`, read the named log: `pod/rp.sh 'tail -20 /root/dl.log'`
(or `vllm_install.log`, `bfcl_install.log`).

## 4. Start vLLM

```bash
pod/rp.sh 'setsid nohup /root/serve.sh > /root/vllm.log 2>&1 < /dev/null &'
pod/rp.sh 'grep -c "startup complete" /root/vllm.log'            # 1 when ready (~4 min); 0 = still loading
pod/rp.sh 'curl -s 127.0.0.1:8000/v1/models | head -c 300'       # must list "qwen3.8-27b"
```

**Don't start the run before this answers.** If vLLM isn't up, every conversation retries for ~30 s and is then
recorded as a failure. If startup fails: `pod/rp.sh 'tail -40 /root/vllm.log'`.

`serve.sh` flags that matter: `--served-model-name qwen3.8-27b` (the runner's default `--model`), `--max-num-seqs 32`
(the hybrid model's limit), `--tool-call-parser qwen3_coder` (Qwen3.8 writes XML tool calls), `--reasoning-parser qwen3`
(thinking comes back separately from the reply, and the answer check only looks at the reply).

## 5. Upload the tasks and runner

`fs_env` is standard-library Python only. It runs with the vLLM venv's Python 3.12 and doesn't need installing.
The playground isn't needed on the pod.

```bash
COPYFILE_DISABLE=1 tar czf /tmp/fs_env.tgz --exclude=__pycache__ --exclude=playground --exclude=.DS_Store fs_env
pod/rp.sh --put /tmp/fs_env.tgz /root/fs_env.tgz
pod/rp.sh 'cd /root && rm -rf fs_env && tar xzf fs_env.tgz && /root/venv/bin/python -m fs_env.validate | tail -1'
# -> 380/380 tasks valid
```

(`COPYFILE_DISABLE=1` keeps macOS from adding `._*` files to the archive. It's ~140 KB; `rp.sh --put` sends it as one
base64 line through the proxy shell, which so far was only used for files under ~30 KB. If the md5 it prints doesn't
match `md5 -q /tmp/fs_env.tgz`, split it: `split -b 40k /tmp/fs_env.tgz /tmp/fsp_`, `--put` each part, then
`cat /root/fsp_* > /root/fs_env.tgz` on the pod.)

## 6. Smoke test: one conversation

```bash
pod/rp.sh 'cd /root && /root/venv/bin/python -m fs_env.run --tasks A3_create_empty_file --trials 1 --seed 0 --out /root/fs-results/smoke.jsonl 2>&1 | tail -5'
```

It prints `PASS` or `FAIL A3_create_empty_file #0 …s ok(act) FAIL(act)` with the differences under it, then
`passed x/1`. Either result is fine here. What matters is that it **finished without `ERROR`** and the time per
conversation is a few minutes. `ERROR … model call failed` means vLLM isn't reachable (step 4).

The command blocks until the conversation finishes. If `rp.sh` returns nothing after ~10 min, the proxy dropped the
session. Run the full run (step 7) the `nohup` way instead.

## 7. Full run: 380 tasks × 4 trials

```bash
pod/rp.sh 'mkdir -p /root/fs-results && cd /root && setsid nohup /root/venv/bin/python -m fs_env.run \
  --trials 4 --seed 0 --threads 16 --out /root/fs-results/q38_fs_x4.jsonl > /root/fs_run.out 2>&1 < /dev/null &'
```

- **`--threads 16`:** 16 conversations at a time. The BFCL run showed multi-turn at 32 overflowing the KV cache.
- **`--seed 0`:** fixes the random home-folder names (all 4 trials of a task get the same name), so the run can be
  reproduced. Leave it out for new names each run.
- **`--out`:** one JSON line per conversation, written as each finishes, with the full conversation (thinking
  included), every call and result, per-turn pass/fail with the differences, and the final file tree.
  **A rerun with the same `--out` overwrites the file**, so give it a new name.

Watch it:

```bash
pod/rp.sh 'wc -l < /root/fs-results/q38_fs_x4.jsonl; tail -n 5 /root/fs_run.out'     # 1520 lines when done
pod/rp.sh 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'
```

It's finished when `fs_run.out` ends with the summary:

```
passed NNNN/1520 (NN.N%)
  A1:x/40  A2:x/40  A3:x/400  ...
```

### 7b. Cheaper: only the 11 hand-written originals (~30 min, ~$1.50)

```bash
pod/rp.sh 'mkdir -p /root/fs-results && cd /root && setsid nohup /root/venv/bin/python -m fs_env.run \
  --tasks A1_copy_keep_original,A2_find_only,A3_create_empty_file,A4_bare_folder,B_verbatim_text,C_exact_names,D_ask_missing_info,E_missing_function,F_act_without_asking,G_act_on_indirect_request,H_do_every_step \
  --trials 4 --seed 0 --threads 16 --out /root/fs-results/q38_fs_originals_x4.jsonl > /root/fs_run.out 2>&1 < /dev/null &'
```

`--tasks` also takes causes: `--tasks D,E` runs all 200 D and E tasks. When done this file has 44 lines. Use its name
in steps 9–10.

## 8. Copy the results to the Mac

```bash
pod/fetch_results.sh /root/fs-results results/fs_env      # -> results/fs_env/fs-results/*.jsonl (checksum-verified)
pod/rp.sh 'cat /root/fs_run.out' > results/fs_env/fs-results/fs_run.out
ls -la results/fs_env/fs-results/
```

## 9. Terminate the pod

In the RunPod console, **Terminate** (not just Stop) once `results/fs_env/fs-results/q38_fs_x4.jsonl` has 1520 lines:

```bash
wc -l results/fs_env/fs-results/q38_fs_x4.jsonl
```

## 10. Read the results (Mac)

Pass rate per task, pass^k (the τ-bench measure: the chance that k trials of the same task all pass), and every
failed turn with its differences:

```bash
../.venv/bin/python - <<'EOF'
import json
from collections import defaultdict
from math import comb
rows = [json.loads(l) for l in open("results/fs_env/fs-results/q38_fs_x4.jsonl")]
by_task = defaultdict(list)
for r in rows:
    by_task[r["id"]].append(r)
n = 4
print(f"{'task':28} passed  per-turn failures")
for tid, rs in sorted(by_task.items()):
    fails = defaultdict(int)
    for r in rs:
        for i, t in enumerate(r["turns"], 1):
            if not t["passed"]:
                fails[f"t{i}({t['expect']})"] += 1
    print(f"{tid:28} {sum(r['passed'] for r in rs)}/{len(rs)}    {dict(fails) or ''}")
for k in range(1, n + 1):
    pk = sum(comb(sum(r["passed"] for r in rs), k) / comb(len(rs), k) for rs in by_task.values()) / len(by_task)
    print(f"pass^{k}: {100 * pk:.1f}%")
errors = [r for r in rows if r["error"]]
print(f"{len(errors)} conversations ended with an error", *[f"  {r['id']} #{r['trial']}: {r['error']}" for r in errors], sep="\n")
print("\nFailed turns:")
for r in rows:
    for i, t in enumerate(r["turns"], 1):
        if not t["passed"]:
            print(f"  {r['id']} #{r['trial']} turn {i} ({t['expect']}){' [' + t['note'] + ']' if t['note'] else ''}")
            for d in t["diffs"]:
                print(f"      {d}")
EOF
```

Each JSON line also keeps `messages`, the whole conversation with thinking (`reasoning_content`), for reading why a
turn failed.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Every conversation fails with `ERROR RuntimeError: model call failed after retries` | vLLM isn't up or crashed. `pod/rp.sh 'tail -40 /root/vllm.log'`, restart with step 4, then rerun step 7 with a new `--out` |
| `HTTP 400 … maximum context length` in a turn | A conversation outgrew `--max-model-len 65536`. Unlikely with these short tasks; raise it to 131072 in `serve.sh` if it happens |
| A turn note says `hit max_tokens` | A model call reached the 16k-token cap (usually a thinking loop). With no tool call in it, the turn ends there and is judged on the files and whatever reply text was produced (often none). This is model behavior, not a setup problem |
| A turn note says `stopped after 20 model calls` | The model kept calling tools for 20 steps (BFCL's limit too) |
| vLLM fails with a CUDA / driver error | Driver 570 needs the `+cu129` wheel (step 2) |
| vLLM fails compiling FlashInfer | `serve.sh` already sets `VLLM_USE_FLASHINFER_SAMPLER=0` and puts the venv on `PATH`. Make sure the uploaded `serve.sh` is the repo version |
| `setup.sh: uv: command not found` | `pod/rp.sh 'pip install uv'`, then run step 3 again |
| Pod restarted | `/dev/shm` was wiped: run `setup.sh` again (step 3), then steps 4–7 |
| `rp.sh` prints nothing | Wrong or stale `RP_TARGET`, or the key isn't on the RunPod account. Test with `pod/rp.sh 'echo ok'` |

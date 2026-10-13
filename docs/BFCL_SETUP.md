# BFCL on Qwen3.8-27B (thinking on)

The Berkeley Function Calling Leaderboard (BFCL v4) on the off-the-shelf `Qwen/Qwen3.8-27B`, no fine-tuning.
This is the same model as the τ³ baseline in `../tau-train/qwen3.8-27b/` (84.6% pass^1 on τ³ retail).

## Settings
| | |
|---|---|
| Benchmark | BFCL v4, gorilla repo commit `6ea5797` (2026-03-23), every category except `web_search` and `format_sensitivity` (20 categories, 4,906 entries) |
| Model | `Qwen/Qwen3.8-27B`, bf16, no LoRA, BFCL registry name `qwen3.8-27b-FC` |
| Mode | FC: native tool calls, parsed by vLLM (`qwen3_coder`), scored by BFCL's own checkers (no LLM judge) |
| Thinking | on (`chat_template_kwargs.enable_thinking`, template default reasoning effort) |
| Temperature | 0.001 (BFCL default) |
| Output cap | 16,384 tokens per model call, thinking included (BFCL sets none; see gotchas) |
| Concurrency | 32 threads = vLLM `--max-num-seqs 32` |
| Serving | vLLM 0.30.0, `--max-model-len 65536`, one A100-SXM4-80GB (RunPod, $1.80/h) |

Left out on purpose:
- **`web_search`** (base + no_snippet, 200 entries) calls SerpAPI and needs a paid `SERPAPI_API_KEY` in `$BFCL_PROJECT_ROOT/.env`. To add it later: `CATEGORIES="web_search_base web_search_no_snippet" /root/run.sh`.
- **`format_sensitivity`** only applies to prompt-mode models; this run is FC mode.

## Files
| File | What it does |
|---|---|
| `pod/setup.sh` | installs vLLM (`/root/venv`), BFCL editable from the gorilla repo (`/root/bfcl-venv`), downloads the model to `/dev/shm/q38` |
| `pod/qwen38_vllm.py` | the BFCL handler: BFCL's official Qwen3 thinking handler (`QwenAPIHandler`) pointed at our vLLM |
| `pod/install_handler.py` | copies the handler into the BFCL checkout and registers `qwen3.8-27b-FC` in `model_config.py` (idempotent) |
| `pod/serve.sh` | starts vLLM on 127.0.0.1:8000 |
| `pod/run.sh` | one `bfcl generate` pass over all categories, logging to `/root/pipeline.log`; safe to rerun (BFCL resumes) |
| `pod/eval_loop.py` | runs next to it on the pod and scores each category as soon as it is complete |
| `pod/summarize_remote.py` | runs on the pod, prints a JSON summary (progress, scores, failure types, tokens, pod state) |
| `pod/rp.sh` | runs a command on the pod through RunPod's SSH proxy, or uploads a file (`--put`) |
| `pod/fetch_results.sh` | copies a pod folder here through the proxy (tar.gz + base64, checksum-verified); default `/root/bfcl-results` → `results/pod/` |
| `update_json.py`, `updater.sh` | keeps `results/bfcl_run.json` current every 30 s (`./updater.sh start\|stop\|status`) |
| `sync_conversations.py`, `conv_sync.sh` | every 5 min copies changed result files (all conversations, with reasoning) and score files into `results/pod/bfcl-results/`; stops after the final copy (`./conv_sync.sh start\|stop\|status`) |
| `build_notebooks.py` | builds both notebooks from one cell list |
| `bfcl_q38_27b_eval.ipynb` | plotly, interactive, with a live-refresh cell (tagged `skip-execution`) |
| `fs_env_q38_27b_eval.ipynb` | the fs_env run (first 10 tasks of causes A3–H = 90 tasks × 4 trials, temperature 1; A1 and A2 left out; `results/fs_env/fs-results/q38_fs90_t1_x4.jsonl`): run info, pass rate and pass^k per cause, a task × trial grid, why turns failed, tokens and time, temperature 0.001 vs 1 on the 11 originals, summary table. Plotly; reads only local files (kept current by `fs_updater.sh`), its live cell (`skip-execution`) re-reads them every 30 s. Built by `build_fs_notebook.py` |
| `fs_updater.sh` / `fs_update.py` | `./fs_updater.sh start\|stop\|status`: during an fs_env run, copies the JSONL and the runner's status from the pod every 2 min, re-executes `fs_env_q38_27b_eval.ipynb` in place and rebuilds its render page; stops after the pass that sees the run finished (log: `fs_updater.log`) |
| `bfcl_q38_27b_eval_matplotlib.ipynb` | matplotlib, static PNGs (renders on GitHub) |
| `analysis/` | DSAT of multi-turn failures: `dsat_multi_turn.py` (first diverging turn per failure, no LLM), `final_state_root_causes.py` (root causes of the 199 wrong-final-state failures), report in `final_state_dsat.md` |
| `fs_env/` | file-system training environment: `filesystem.py` (in-memory FS, 14 tools), `tools.py` (OpenAI-format tool definitions + dispatcher, withheld tools), `checker.py` (state vs target: exact tree, per-file exact/`contains`/`regex` content), `tasks/*.json` (A3/D/E, which Qwen fails: 100 each = hand-written original + 9 in `make_tasks.py` + 90 from the pattern families in `task_families.py` (ids 11–50 first batch, 51–100 appended so existing ids never change); the other causes 10 each; every task's `trap_example` must fail; random background clutter is added to the home folder on every load: origin, per-turn target, gold calls; home folder is a `{user}` placeholder filled with a random name per task on every `load_tasks()`), `python -m fs_env.validate` replays gold calls, `python -m fs_env.test_fs_env` self-tests, `python -m fs_env.run` runs the tasks against the vLLM server (stdlib only; BFCL settings: thinking on, temp 0.001, 16k cap; `--tasks A1,E --trials 4 --seed 0`; one JSONL line per task × trial with the full conversation), `fs_env/RUN_ON_POD.md` is the step-by-step pod runbook (380 tasks × 4 trials ≈ 10.5–16.5 h, ~$19–30; or the 11 originals ≈ 40–60 min, ~$1.50), `python -m fs_env.playground.server` opens a local page (http://localhost:8765) to try every tool and play the tasks by hand on the real fs_env code |
| `results/pod/bfcl-smoke/` | the 6-entry smoke test run before the full run |
| `renders/bfcl_multi_turn.html` | one offline page with a tab per multi-turn category (base, miss_func, miss_param, long_context): every conversation grouped by outcome (passed or BFCL's failure type), with the user turns, thinking, tool calls and results, the expected calls per turn and the state differences for failures. Built by `build_mt_viewer.py` |
| `renders/fs_env_<run>.html` | one offline page per fs_env run (`results/fs_env/fs-results/<run>.jsonl`): conversations grouped by task (failing first) with the task's trap, each turn's expected behavior, gold calls and answer check, the model's thinking, calls, results and replies, the per-turn verdict with differences, and the start/final file trees. Built by `build_fs_viewer.py [run.jsonl]` |

The notebooks only read `results/bfcl_run.json`. To refresh: `jupyter nbconvert --to notebook --execute --inplace <nb>`.

## Running it again (e.g. another model or a new pod)
1. **Pod**: one A100/H100 80GB. The 30–50 GB container disk is too small for the 56 GB model, so the model goes on the RAM disk
   (`/dev/shm`, ~117 GB on these pods). It's lost if the pod restarts, and `setup.sh` downloads it again (~4 min).
2. **Connect**: set the proxy address from the RunPod "Connect" panel, e.g. `export RP_TARGET=<pod-id>-<n>@ssh.runpod.io`.
   The proxy has no `scp`, so `pod/rp.sh --put` uploads files through the shell.
3. **Upload and set up**:
   ```bash
   cd bfcl-train
   for f in setup.sh serve.sh run.sh qwen38_vllm.py install_handler.py summarize_remote.py; do pod/rp.sh --put pod/$f /root/$f; done
   pod/rp.sh 'setsid nohup /root/setup.sh > /root/setup.out 2>&1 < /dev/null &'
   pod/rp.sh 'tail -n 3 /root/pipeline.log'              # wait for SETUP_DONE (~4 min)
   pod/rp.sh '/root/bfcl-venv/bin/python /root/install_handler.py'
   pod/rp.sh 'setsid nohup /root/serve.sh > /root/vllm.log 2>&1 < /dev/null &'
   pod/rp.sh 'grep -c "startup complete" /root/vllm.log'  # 1 when ready (~4 min)
   ```
4. **Run**: upload `eval_loop.py` too, then `pod/rp.sh 'setsid nohup /root/run.sh > /root/run.out 2>&1 < /dev/null &'`. Here: `./updater.sh start` and `./conv_sync.sh start`, then open the notebook.
5. **Before terminating the pod**: `pod/fetch_results.sh` copies `/root/bfcl-results/` (every model output, with reasoning, and every score file) into `results/pod/`.

## Gotchas found while setting this up
- **No built-in handler for Qwen3.8.** BFCL's local Qwen handlers expect hermes JSON tool calls, but Qwen3.8 writes XML
  (`<function=…>`). So BFCL talks to vLLM's OpenAI API instead, and vLLM's `qwen3_coder` parser turns the XML into normal tool calls.
- **Missing dependency:** BFCL imports `qwen_agent`, which needs `soundfile`, and it isn't in BFCL's dependency list.
  `setup.sh` installs it.
- **Placeholder API keys:** `QwenAPIHandler` builds an OpenAI client and a DashScope client in `__init__`, and those fail
  without keys. The handler sets placeholder `OPENAI_API_KEY`/`QWEN_API_KEY`. The clients are replaced right away and never called.
- **Reasoning field:** vLLM streams thinking as `delta.reasoning`, DashScope as `delta.reasoning_content`. The handler accepts both,
  and the reasoning is saved with each result.
- **`--test-category` takes a comma-separated list**, not spaces.
- **Thinking loops.** At BFCL's near-greedy temperature (0.001) the model sometimes repeats itself until the context is full.
  The first attempt ran one category at a time, and one `simple_java` entry kept generating for 13+ minutes at 28 tokens/s
  while the rest of the GPU sat idle, waiting for the category to finish. Fixes: a 16,384-token cap per model call
  (`QWEN38_MAX_TOKENS`; normal answers use hundreds to a few thousand tokens, and a loop fails either way), and one
  `bfcl generate` pass over every category, so a slow entry holds 1 of 32 slots instead of stalling everything.
  The 499 entries generated before the switch were kept (BFCL resumes), and those were generated without the cap.
- **Entries with no tools.** 4 `live_irrelevance` entries (`120-9-0` … `123-9-3`) have an empty tool list. vLLM rejects
  `tools: []` with HTTP 400, and BFCL scores the resulting error text as a correct "no call", so they would be free passes.
  The handler now leaves the tool fields out when the list is empty. In this run they were first generated with the old
  handler; `pod/fix_empty_tools.sh` (run after RUN_DONE) drops those 4 results, regenerates them and rescores.
- **Temperature vs the leaderboard.** BFCL's default is 0.001 and this run sends it, but BFCL's official handler for Qwen3
  thinking models sends no temperature at all, so leaderboard Qwen numbers ran at the provider default. Qwen recommends
  temperature 1.0 / top_p 0.95 / top_k 20 for this model and warns that greedy decoding can cause endless repetition
  (see the thinking-loop note). Expect some difference from published Qwen scores for that reason.
- **Killing `bfcl generate`:** a thread waiting on a stuck streaming request ignores SIGTERM. Use `kill -9`, then check
  `pgrep -f "bfcl generate"` before starting again, so two generators never write the same files.
- **BFCL's `latency` field is useless with streaming:** it measures time to the first chunk (~0.2 s). Timing in the notebook comes
  from `run.sh`'s log lines instead.
- **Memory categories** depend on "prerequisite" conversations that fill the memory first. Running a single memory entry with
  `--run-ids` skips them, so it always fails (that's what happened in the smoke test); full-category runs include them.
- **The driver decides the vLLM wheel:** a driver-580 pod runs the default `vllm==0.30.0` (torch cu130). On a driver-570 pod,
  use the `+cu129` wheel as in `../tau-train/qwen3.8-27b/pod/setup.sh`.
- **The hybrid model needs `--max-num-seqs ≤ 32`** (Mamba cache), and `VLLM_USE_FLASHINFER_SAMPLER=0` plus the venv on `PATH`
  avoid a FlashInfer JIT failure. Both are carried over from the τ³ run.
- **Skipped memory setup (`memory_rec_sum`, student scenario).** BFCL replays each scenario's setup conversations one turn
  at a time, and recursive summarization makes the model rewrite a 10,000-character summary every turn, so the last two
  student setups (`student-8`, 12 turns; `student-9`, 13 turns) would have taken an estimated 3–6 more hours with one request
  on the GPU. They were replaced with placeholder results (`"skipped": true`), and the 50 student questions ran on the memory
  saved after `student-7`. Questions about facts from those two conversations fail naturally. Backup of the original file:
  `…/BFCL_v4_memory_rec_sum_prereq_result.json.before_skip`.
- **Context limit (not fixed in this run):** 11 `multi_turn_long_context` entries hit vLLM's `--max-model-len 65536`
  ("maximum context length is 65536 tokens") and count as failures (~5.5 points of that category). The model supports
  262k; use `--max-model-len 131072` next time.
- **Overall score:** BFCL's overall accuracy weights categories, not entries. The notebook shows a plain average while the run is
  in progress and BFCL's own `data_overall.csv` number once everything is evaluated.

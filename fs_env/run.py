"""Run the tasks against a model behind an OpenAI-compatible API (our vLLM) and score every turn.

  python -m fs_env.run                                   # all tasks, 1 trial, vLLM on 127.0.0.1:8000
  python -m fs_env.run --tasks A1,E --trials 4 --out results/fs_env/q38.jsonl

Stdlib only (no openai package), so any Python 3.10+ on the pod works. Defaults match the BFCL run:
thinking on, temperature 0.001, 16,384 output tokens per model call, parallel tool calls.

Each turn: the user message goes in, then model call -> execute tool calls -> model call ... until the model answers
without calling a tool (or --max-steps). The turn passes if the file system then equals the turn's target
(checker.compare) and, on turns that need a reply (answer/ask/refuse), that final reply passes the turn's
answer_check (checker.check_answer). Later turns continue from the state the model actually left, like BFCL multi-turn.
A task passes if every turn passes. One JSON line per (task, trial) with the full conversation, thinking included.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .checker import check_answer, compare
from .tasks import load_tasks, new_fs
from .tools import execute, tools_for


def chat(args, messages, tools):
    body = {
        "model": args.model,
        "messages": messages,
        "tools": tools,
        "parallel_tool_calls": True,
        "temperature": args.temperature,
        "chat_template_kwargs": {"enable_thinking": not args.no_thinking},
    }
    if args.max_tokens:
        body["max_tokens"] = args.max_tokens
    req = urllib.request.Request(f"{args.base_url}/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {args.api_key}"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code < 500:  # bad request (e.g. context too long): retrying won't help
                raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:500]}") from None
            err = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            err = e
        time.sleep(2 ** attempt)
    raise RuntimeError(f"model call failed after retries: {err}")


def run_turn(args, fs, messages, turn):
    withheld = turn.get("withheld_tools", [])
    tools = tools_for(withheld)
    messages.append({"role": "user", "content": turn["user"]})
    calls, usage, note = [], {"prompt": 0, "completion": 0}, None
    for _ in range(args.max_steps):
        resp = chat(args, messages, tools)
        usage["prompt"] += resp["usage"]["prompt_tokens"]
        usage["completion"] += resp["usage"]["completion_tokens"]
        choice = resp["choices"][0]
        msg = choice["message"]
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
        out = {"role": "assistant", "content": msg.get("content") or ""}
        if reasoning:
            out["reasoning_content"] = reasoning
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            out["tool_calls"] = [{"id": c["id"], "type": "function", "function": c["function"]} for c in tool_calls]
        messages.append(out)
        if choice.get("finish_reason") == "length":
            note = "hit max_tokens"
        if not tool_calls:
            break
        for c in tool_calls:
            name, arguments = c["function"]["name"], c["function"]["arguments"]
            result = execute(fs, name, arguments, withheld)
            calls.append({"name": name, "arguments": arguments, **result})
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(result)})
    else:
        note = f"stopped after {args.max_steps} model calls"
    answer = messages[-1].get("content", "") if messages[-1]["role"] == "assistant" else ""
    diffs = compare(fs.tree(), turn["target"]) + check_answer(answer, turn.get("answer_check"))
    return {"expect": turn["expect"], "passed": not diffs, "diffs": diffs, "calls": calls,
            "answer": answer,
            "usage": usage, "note": note}


def run_task(args, task, trial):
    fs, messages, turns, error = new_fs(task), [], [], None
    if args.system:
        messages.append({"role": "system", "content": args.system})
    start = time.time()
    try:
        for turn in task["turns"]:
            turns.append(run_turn(args, fs, messages, turn))
    except Exception as e:  # keep going with the other tasks; the record says what happened
        error = f"{type(e).__name__}: {e}"
    return {"id": task["id"], "cause": task["cause"], "trial": trial, "user_name": task.get("user_name"),
            "passed": error is None and all(t["passed"] for t in turns),
            "turns": turns, "error": error, "seconds": round(time.time() - start, 1),
            "final_tree": fs.tree(), "messages": messages}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default="qwen3.8-27b")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--tasks", help="comma-separated causes or ids (A1,E or A1_copy_keep_original); default all")
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--seed", type=int, help="seed for the random home-folder names (default: new names each run)")
    p.add_argument("--temperature", type=float, default=0.001)
    p.add_argument("--max-tokens", type=int, default=16384, help="per model call, thinking included; 0 = no cap")
    p.add_argument("--max-steps", type=int, default=20, help="model calls per turn")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--no-thinking", action="store_true")
    p.add_argument("--system", default="", help="optional system prompt (BFCL multi-turn uses none)")
    p.add_argument("--system-file", help="read the system prompt from this file (e.g. fs_env/sft_rules_prompt.txt)")
    p.add_argument("--split", choices=["train", "eval"], help="only the tasks of this part of fs_env/split.json")
    p.add_argument("--out", default=f"results/fs_env/run_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    args = p.parse_args()
    if args.system_file:
        args.system = Path(args.system_file).read_text().strip()

    tasks = load_tasks(seed=args.seed)
    if args.tasks:
        order = args.tasks.split(",")   # run in the order given (ids or causes), e.g. the important causes first
        rank = {x: i for i, x in enumerate(order)}
        tasks = [t for t in tasks if t["cause"] in rank or t["id"] in rank]
        tasks.sort(key=lambda t: (rank.get(t["id"], rank.get(t["cause"], len(order))), t["id"]))
    if args.split:
        part = set(json.loads((Path(__file__).parent / "split.json").read_text())[args.split])
        tasks = [t for t in tasks if t["id"] in part]
    jobs = [(t, k) for t in tasks for k in range(args.trials)]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"{len(tasks)} tasks x {args.trials} trials -> {out}", flush=True)

    results = []
    with ThreadPoolExecutor(args.threads) as pool, out.open("w") as f:
        for r in pool.map(lambda j: run_task(args, *j), jobs):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            results.append(r)
            turns = " ".join(("ok" if t["passed"] else "FAIL") + f"({t['expect']})" for t in r["turns"])
            print(f"{'PASS' if r['passed'] else 'FAIL'} {r['id']} #{r['trial']} {r['seconds']}s  {turns}"
                  + (f"  ERROR {r['error']}" if r["error"] else ""), flush=True)
            for i, t in enumerate(r["turns"], 1):
                for d in t["diffs"]:
                    print(f"      turn {i}: {d}", flush=True)

    passed = sum(r["passed"] for r in results)
    print(f"\npassed {passed}/{len(results)} ({100 * passed / max(len(results), 1):.1f}%)")
    by_cause = {}
    for r in results:
        by_cause.setdefault(r["cause"], []).append(r["passed"])
    print("  " + "  ".join(f"{c}:{sum(v)}/{len(v)}" for c, v in sorted(by_cause.items())))
    sys.exit(0)


if __name__ == "__main__":
    main()

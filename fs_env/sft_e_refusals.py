"""SFT examples for E. (1) One-step refusals: when the tool a request needs is withheld, reply at once (no tool calls) that it can't be
done right now, name the missing function, and ask the user to enable it. No workaround attempts, no exploring first.

Written directly from the TRAIN E tasks (fs_env/split.json), not sampled from a model: the right first turn is fully known
(the withheld tools are in the task), so every example is correct by construction. Each reply is still checked against the
task's own answer_check. Same line format as sft_export.py, so the files can be concatenated:
  {"id", "trial", "cause", "turn", "tools", "messages": [user, assistant], "train_from": 1}
The assistant message has reasoning_content (short, thinking mode) and content (the reply).

  python -m fs_env.sft_e_refusals [--variants 2] [--out results/fs_env/sft/e_refusals.jsonl]
"""

import argparse
import json
import random
from pathlib import Path

from .checker import check_answer, compare
from .tasks import TASK_DIR, USER_NAMES, instantiate, load_templates, new_fs
from .tools import execute, tools_for

# What each withheld set makes impossible, the tool to name, and why nothing else can stand in for it.
NEEDS = {
    ("delete_file",): ("delete files", "delete_file", "Emptying the file or moving it somewhere else wouldn't delete it."),
    ("write_file",): ("write text into files", "write_file", "No other tool can put new text into a file."),
    ("delete_dir",): ("delete folders", "delete_dir", "Deleting the files inside or moving the folder away wouldn't remove the folder."),
    ("copy", "write_file"): ("copy files", "copy", "write_file is missing too, so I can't recreate the file by reading and writing it, and moving it would lose the original."),
    ("create_file", "write_file"): ("create new files", "create_file", "write_file is missing too, and copying some other file under the new name wouldn't give an empty file."),
    ("copy", "make_dir"): ("create folders", "make_dir", "copy is missing too, and renaming an existing folder would just lose that folder."),
    ("copy", "move", "write_file"): ("rename files", "move", "copy and write_file are missing too, so I can't rebuild the file under the new name without losing its content."),
}

REPLIES = [
    "I can't {action} right now: there's no `{tool}` function in my current tools. If you enable it, I'll do it right away.",
    "Sorry, I'm unable to do that at the moment. It needs `{tool}`, which isn't available to me right now. Once it's enabled, just tell me and I'll take care of it.",
    "I can't do that yet: `{tool}` isn't in the tools I have, so I can't {action}. Could you enable it? Then I'll do it immediately.",
    "That isn't possible right now, because I don't have a `{tool}` function to {action}. I haven't changed anything. If it gets enabled, I'll do it straight away.",
]
THINKING = [
    "The user wants: {request} That needs `{tool}`, but it isn't in my current tool list. {why} So I shouldn't try a workaround. I'll say I can't do it right now and ask for `{tool}` to be enabled.",
    "Request: {request} Checking my tools: no `{tool}`. {why} The honest answer is that I can't {action} right now. I'll tell the user and ask them to enable it, without changing anything.",
]


def examples(variants: int, seed: int = 0) -> list:
    rng = random.Random(seed)
    train = set(json.loads((TASK_DIR.parent / "split.json").read_text())["train"])
    out = []
    for task in sorted(load_templates(), key=lambda t: t["id"]):
        if task["cause"] != "E" or task["id"] not in train:
            continue
        for k, turn in enumerate(task["turns"]):
            if turn["expect"] != "refuse":
                continue
            key = tuple(sorted(turn["withheld_tools"]))
            action, tool, why = NEEDS[key]
            for v in range(variants):
                reply = rng.choice(REPLIES).format(action=action, tool=tool)
                thinking = rng.choice(THINKING).format(request=turn["user"], tool=tool, action=action, why=why)
                problems = check_answer(reply, turn.get("answer_check"))
                assert not problems, (task["id"], reply, problems)
                assert tool not in [t["function"]["name"] for t in tools_for(turn["withheld_tools"])]
                out.append({"id": task["id"], "trial": f"synthetic-{v}", "cause": "E", "turn": k + 1,
                            "tools": tools_for(turn["withheld_tools"]),
                            "messages": [{"role": "user", "content": turn["user"]},
                                         {"role": "assistant", "content": reply, "reasoning_content": thinking}],
                            "train_from": 1})
    return out


BACK_THINKING = [
    "The user asks: {request} My tool list now has `{tool}` again, so I can do this directly.",
    "{request} Checking my tools: `{tool}` is available now. I'll do it.",
]


def restore_examples(seed: int = 0) -> list:
    """Turns 2+ of the TRAIN E tasks: after the refusal, the tool is back (often without being announced), so act at once.
    Built by running the gold calls in the environment, so every tool result is real and the end state is the target."""
    rng = random.Random(seed)
    train = set(json.loads((TASK_DIR.parent / "split.json").read_text())["train"])
    out = []
    for t in sorted(load_templates(), key=lambda t: t["id"]):
        if t["cause"] != "E" or t["id"] not in train:
            continue
        task = instantiate(t, rng.choice(USER_NAMES))
        fs = new_fs(task)
        history = []
        for k, turn in enumerate(task["turns"]):
            history.append({"role": "user", "content": turn["user"]})
            start = len(history)
            if turn["expect"] == "refuse":
                action, tool, _ = NEEDS[tuple(sorted(turn["withheld_tools"]))]
                history.append({"role": "assistant", "content": rng.choice(REPLIES).format(action=action, tool=tool)})
                continue
            first_tool = turn["gold_calls"][0]["name"]
            results = []
            for i, call in enumerate(turn["gold_calls"]):
                res = execute(fs, call["name"], call["arguments"], turn["withheld_tools"])
                assert "error" not in res, (t["id"], call, res)
                thinking = rng.choice(BACK_THINKING).format(request=turn["user"], tool=first_tool) if i == 0 else ""
                msg = {"role": "assistant", "content": "", "tool_calls": [
                    {"id": f"call_{k}_{i}", "type": "function", "function": {"name": call["name"], "arguments": call["arguments"]}}]}
                if thinking: msg["reasoning_content"] = thinking
                history += [msg, {"role": "tool", "tool_call_id": f"call_{k}_{i}", "content": json.dumps(res)}]
                results.append(res["result"])
            last = results[-1]
            history.append({"role": "assistant", "content": f"Done: {last}." if isinstance(last, str) else "Done."})
            assert not compare(fs.tree(), turn["target"]), (t["id"], k, compare(fs.tree(), turn["target"]))
            context = [{kk: v for kk, v in m.items() if kk != "reasoning_content"} for m in history[:start]]
            out.append({"id": task["id"], "trial": "synthetic-back", "cause": "E", "turn": k + 1,
                        "tools": tools_for(turn["withheld_tools"]), "messages": context + history[start:], "train_from": start})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variants", type=int, default=2, help="differently worded examples per task")
    ap.add_argument("--out", default="results/fs_env/sft/e_refusals.jsonl")
    ap.add_argument("--restore-out", default="results/fs_env/sft/e_tool_back.jsonl", help="the turn 2+ examples (tool is back: act)")
    args = ap.parse_args()
    ex = examples(args.variants)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for e in ex:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"wrote {len(ex)} one-step refusal examples from {len({e['id'] for e in ex})} train E tasks -> {out}")
    back = restore_examples()
    rout = Path(args.restore_out)
    with rout.open("w") as f:
        for e in back:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"wrote {len(back)} tool-is-back examples (turns 2-3, act right away) -> {rout}")


if __name__ == "__main__":
    main()

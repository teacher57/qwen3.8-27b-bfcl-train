"""Replay every task's gold calls and check each turn reaches its target:
    python -m fs_env.validate                 the tasks/ folder
    python -m fs_env.validate a1_brake.json   a task set in task_sets/"""

import sys

from .tasks import load_tasks, source_path, validate


if __name__ == "__main__":
    tasks = load_tasks(source_path(sys.argv[1] if len(sys.argv) > 1 else "tasks"))
    bad = 0
    for t in tasks:
        problems = validate(t)
        bad += bool(problems)
        print(f"{'OK  ' if not problems else 'FAIL'} {t['id']}  ({len(t['turns'])} turns)")
        for p in problems:
            print(f"     {p}")
    print(f"\n{len(tasks) - bad}/{len(tasks)} tasks valid")
    raise SystemExit(1 if bad else 0)

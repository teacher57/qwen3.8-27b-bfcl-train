"""Task format, loading, and validation by replaying the gold calls.

A task (tasks/*.json):
  id, cause, cause_title, trap          what the task tests and the wrong behavior it targets
  origin: {cwd, tree}                   starting state
  turns: [{
    user                                the user's message
    expect                              "act" (must change state), "answer" (read-only),
                                        "ask" (missing info: ask, change nothing), "refuse" (function missing: say so)
    answer_check                        for answer/ask/refuse turns: {must_match: [regex], example} that the
                                        model's final reply must satisfy (checker.check_answer)
    withheld_tools                      tools not offered to the model on this turn
    gold_calls: [{name, arguments}]     one correct way to get from the previous state to the target
    target                              the tree that must exist after this turn (see checker.py)
  }]
  trap_example (optional)               {turn, calls, reply}: the wrong behavior the task targets; validate() checks
                                        that doing it on that turn really fails

Task files use "{user}" for the home folder name (in cwd, trees and paths). load_tasks() replaces it with a
random name and adds random background clutter (CLUTTER) to the home folder, new on every load; the chosen name is
kept in task["user_name"] and the clutter names in task["clutter"].
"""

import fnmatch
import json
import random
import re
from pathlib import Path

from .checker import check_answer, compare, flatten
from .filesystem import STATE_CHANGING, FileSystem
from .tools import TOOL_NAMES, execute

TASK_DIR = Path(__file__).parent / "tasks"
EXPECT = {"act", "answer", "ask", "refuse"}
# What a refuse turn may withhold: sets whose effect no combination of the remaining tools can reproduce exactly.
# Withholding a replaceable tool (move = copy + delete) would make a working workaround count as a failure.
IRREPLACEABLE_WITHHELD = [
    {"write_file"},                    # nothing else puts new text into a file
    {"delete_file"},                   # emptying or moving a file isn't deleting it
    {"delete_dir"},                    # moving a folder away isn't deleting it
    {"copy", "write_file"},            # copy = read_file + write_file
    {"create_file", "write_file"},     # create_file = write_file(""); also needs no empty file around to copy
    {"make_dir", "copy"},              # make_dir = copy an existing empty folder
    {"move", "copy", "write_file"},    # move = copy + delete, or read + write + delete
]
MIN_ACT_TURNS = 2  # every task has at least two turns that change files
LAZY_REPLIES = ["", "Done.", "Okay!", "I've completed the task."]  # must fail every answer_check
PLACEHOLDER = "{user}"

USER_NAMES = [
    "ana", "liam", "sofia", "mateo", "olivia", "noah", "emma", "lucas", "mia", "ethan", "zoe", "omar",
    "priya", "arjun", "mei", "kenji", "yuki", "hana", "jin", "minjun", "fatima", "yusuf", "leila", "amir",
    "chloe", "theo", "ines", "pablo", "elena", "marco", "giulia", "luca", "anna", "jonas", "lena", "felix",
    "olga", "ivan", "dmitri", "katya", "nadia", "tomas", "eva", "kofi", "amara", "tunde", "zanele", "sipho",
    "diego", "camila", "valeria", "rafael", "nina", "oscar", "freya", "erik", "astrid", "sean", "aoife",
    "maya", "noor", "tariq", "sana", "ravi", "kavya", "lin", "wei", "chen", "aiden", "grace", "isaac",
]


# Background items added at random to the home folder on every load, so the starting tree differs every time.
# Nothing in a task may touch them: an item is only used when none of its names appear in the task, no find_files
# pattern of the task matches them, and none of its file contents contain text the task searches for.
CLUTTER = [
    ("Downloads", {"setup_v3.dmg": "DMG installer", "receipt_8812.pdf": "PDF receipt", "unzipped": {"read_me_first.txt": "install steps\n"}}),
    ("Pictures", {"IMG_2231.HEIC": "HEIC photo", "screenshot_0907.png": "PNG screenshot"}),
    ("Videos", {"clip_birthday.mov": "MOV clip"}),
    ("Desktop", {"Screenshot 2026-09-30.png": "PNG screen", "untitled folder": {}}),
    ("Projects", {"side_hustle": {"README.md": "# side hustle\nidea: tea shop\n", "main.rs": "fn main() {}\n"}}),
    ("scripts", {"cleanup.sh": "#!/bin/sh\nrm -rf /tmp/scratch\n", "sync_photos.py": "print('sync')\n"}),
    (".bashrc", "export EDITOR=vim\nalias ll='ls -la'\n"),
    (".gitconfig", "[user]\n  name = me\n"),
    (".cache", {"thumbnails": {"a1b2.png": "PNG thumb"}, "pip": {}}),
    ("bookmarks.html", "<a href='https://example.org'>example</a>\n"),
    ("old_resume_2023.pdf", "PDF resume"),
    ("grocery_list.txt", "oat milk\ncoffee beans\n"),
    ("Games", {"saves": {"slot1.sav": "SAVE 1"}}),
    ("Library", {"Fonts": {"Inter.ttf": "TTF font"}}),
    ("tax_2024.zip", "ZIP archive"),
    ("ideas_scratch.md", "- dark mode for the blog\n- learn piano\n"),
    ("Music Library", {"playlist_chill.m3u": "#EXTM3U\n"}),
    ("vm_images", {"ubuntu-24.04.iso": "ISO image"}),
    ("passwords.kdbx", "KDBX vault"),
    ("Zoom", {"recording_0412.m4a": "M4A audio"}),
]


def _names(value, name: str) -> list:
    out = [name]
    if isinstance(value, dict):
        for k, v in value.items():
            out += _names(v, k)
    return out


def _contents(value) -> list:
    if isinstance(value, dict):
        return [x for v in value.values() for x in _contents(v)]
    return [value]


def add_clutter(task: dict, rng: random.Random, low: int = 2, high: int = 5) -> dict:
    """Add 2-5 random CLUTTER items to the home folder of the origin and of every turn's target (in place)."""
    text = json.dumps(task, ensure_ascii=False).lower()
    calls = [c for t in task["turns"] for c in t["gold_calls"]] + task.get("trap_example", {}).get("calls", [])
    patterns = [c["arguments"]["name"] for c in calls if c["name"] == "find_files" and isinstance(c.get("arguments"), dict)]
    searches = [c["arguments"]["text"] for c in calls if c["name"] == "search_text" and isinstance(c.get("arguments"), dict)]

    def usable(item):
        name, value = item
        names = _names(value, name)
        return (not any(n.lower() in text for n in names)
                and not any(fnmatch.fnmatchcase(n, pat) for n in names for pat in patterns)
                and not any(q in body for body in _contents(value) for q in searches))

    pool = [item for item in CLUTTER if usable(item)]
    picked = rng.sample(pool, min(len(pool), rng.randint(low, high)))
    home = task["user_name"]
    for tree in [task["origin"]["tree"]] + [t["target"] for t in task["turns"]]:
        for name, value in picked:
            tree["home"][home][name] = json.loads(json.dumps(value))
    task["clutter"] = [name for name, _ in picked]
    return task


FOLDER_WORDS = (Path(__file__).parent / "folder_words.txt").read_text().split()
PLAIN_DIR = re.compile(r"[a-z][a-z0-9_]*")   # renamed; odd names (#drafts, Q4 Plans, v2.0, .config, archive_A) stay: C tasks test them
NO_RENAME_KEYS = {"id", "cause", "cause_title", "expect", "name", "role", "user_name"}


def _dir_names(tree: dict) -> set:
    out = set()
    for k, v in tree.items():
        if isinstance(v, dict) and "$rule" not in v:
            out.add(k); out |= _dir_names(v)
    return out


def randomize_folders(task: dict, rng: random.Random) -> dict:
    """Give every plain folder of the task a random word from folder_words.txt, the same word everywhere it appears:
    user messages, starting tree, every turn's target, gold calls, answer checks, trap example, file contents. Done on the
    Python objects (not the JSON text) with whole-word matching, so 'docs' changes but 'docs_old' or 'meeting_notes' don't."""
    home = task["origin"]["tree"]["home"][task["user_name"]]
    names = _dir_names(home)
    for t in task["turns"]:
        names |= _dir_names(t["target"]["home"][task["user_name"]])
    names = sorted(n for n in names if PLAIN_DIR.fullmatch(n) and n != task["user_name"])
    if not names:
        task["folder_map"] = {}; return task
    text = json.dumps(task).lower()
    free = [w for w in FOLDER_WORDS if w not in text]
    new = rng.sample(free, len(names))
    mapping = dict(zip(names, new))
    pat = re.compile(r"(?<![\w\-.])(" + "|".join(map(re.escape, sorted(names, key=len, reverse=True))) + r")(?![\w\-])")
    sub = lambda s: pat.sub(lambda m: mapping[m.group(1)], s)

    def walk(x, in_tree=False):
        # keys are renamed only inside file trees (folder/file names); structural keys ("calls", "turns"...) never are
        if isinstance(x, dict):
            return {(sub(k) if in_tree else k): (v if (not in_tree and k in NO_RENAME_KEYS) else
                                                  walk(v, in_tree or k in ("tree", "target")))
                    for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v, in_tree) for v in x]
        return sub(x) if isinstance(x, str) else x
    out = walk(task)
    task.clear(); task.update(out)
    task["folder_map"] = mapping
    return task


def instantiate(task: dict, name: str, rng: random.Random | None = None) -> dict:
    """Copy of a task template with the home folder named `name`; with `rng`, also random folder names (randomize_folders)
    and random background clutter."""
    task = json.loads(json.dumps(task).replace(PLACEHOLDER, name))
    task["user_name"] = name
    if rng is not None:
        randomize_folders(task, rng)
        add_clutter(task, rng)
    return task


def load_templates(task_dir: Path = TASK_DIR) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(task_dir.glob("*.json"))]


def load_tasks(task_dir: Path = TASK_DIR, seed=None) -> list[dict]:
    """Tasks ready to run, each with its own random user name and background clutter, so every load gives a different
    starting tree (pass `seed` to make it reproducible)."""
    rng = random.Random(seed)
    return [instantiate(t, rng.choice(USER_NAMES), rng) for t in load_templates(task_dir)]


def new_fs(task: dict) -> FileSystem:
    if PLACEHOLDER in json.dumps(task["origin"]):
        raise ValueError(f"task {task['id']} still has {PLACEHOLDER}; load it with load_tasks() or instantiate()")
    return FileSystem(task["origin"]["tree"], task["origin"]["cwd"])


def validate(task: dict) -> list[str]:
    """Problems with a task; empty means the gold calls reach every turn's target."""
    problems = []
    try:
        fs = new_fs(task)
    except ValueError as e:
        return [f"origin: {e}"]
    acts = sum(t.get("expect") == "act" for t in task["turns"])
    if acts < MIN_ACT_TURNS:
        problems.append(f"only {acts} act turn(s); every task needs at least {MIN_ACT_TURNS}")
    previous = fs.tree()
    for i, turn in enumerate(task["turns"], 1):
        where = f"turn {i}"
        if turn.get("expect") not in EXPECT:
            problems.append(f"{where}: expect must be one of {sorted(EXPECT)}")
        withheld = turn.get("withheld_tools", [])
        for name in withheld:
            if name not in TOOL_NAMES:
                problems.append(f"{where}: withheld tool {name} doesn't exist")
        if withheld and turn.get("expect") != "refuse":
            problems.append(f"{where}: tools are withheld but expect isn't refuse")
        if turn.get("expect") == "refuse":
            if set(withheld) not in IRREPLACEABLE_WITHHELD:
                problems.append(f"{where}: withheld {sorted(withheld)} can be replaced by other tools; "
                                f"use one of {[sorted(x) for x in IRREPLACEABLE_WITHHELD]}")
            if "create_file" in withheld and "copy" not in withheld and "" in fs.files.values():
                problems.append(f"{where}: create_file is withheld but an empty file exists that copy could duplicate")
        for call in turn["gold_calls"]:
            out = execute(fs, call["name"], call["arguments"], withheld)
            if "error" in out:
                problems.append(f"{where}: gold call {call['name']}({call['arguments']}) failed: {out['error']}")
        changes = any(c["name"] in STATE_CHANGING for c in turn["gold_calls"])
        if turn.get("expect") == "act" and not changes:
            problems.append(f"{where}: expect=act but no gold call changes state")
        if turn.get("expect") in ("answer", "ask", "refuse"):
            if changes:
                problems.append(f"{where}: expect={turn['expect']} but a gold call changes state")
            if flatten(turn["target"]) != flatten(previous):
                problems.append(f"{where}: expect={turn['expect']} but target differs from the previous state")
        check = turn.get("answer_check")
        if turn.get("expect") in ("answer", "ask", "refuse") and not check:
            problems.append(f"{where}: expect={turn['expect']} needs an answer_check")
        if check:
            problems += [f"{where}: answer_check example fails: {d}" for d in check_answer(check["example"], check)]
            for lazy in LAZY_REPLIES:
                if not check_answer(lazy, check):
                    problems.append(f"{where}: answer_check accepts the lazy reply {lazy!r}")
        problems += [f"{where}: {d}" for d in compare(fs.tree(), turn["target"])]
        previous = fs.tree()
    if "trap_example" in task:
        problems += check_trap(task)
    return problems


def check_trap(task: dict) -> list[str]:
    """The trap example must run without errors and must fail its turn."""
    trap = task["trap_example"]
    k = trap["turn"]
    if not 0 <= k < len(task["turns"]):
        return [f"trap_example: turn {k} doesn't exist"]
    fs = new_fs(task)
    for t in task["turns"][:k]:
        for c in t["gold_calls"]:
            execute(fs, c["name"], c["arguments"], t["withheld_tools"])
    turn = task["turns"][k]
    problems = []
    for c in trap["calls"]:
        out = execute(fs, c["name"], c["arguments"], turn["withheld_tools"])
        if "error" in out:
            problems.append(f"trap_example: call {c['name']}({c['arguments']}) errors: {out['error']}")
    caught = compare(fs.tree(), turn["target"]) + check_answer(trap.get("reply", ""), turn.get("answer_check"))
    if not caught:
        problems.append(f"trap_example: the trap passes turn {k + 1}, so the task doesn't catch it")
    return problems


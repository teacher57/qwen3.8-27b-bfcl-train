"""Self-tests: python -m fs_env.test_fs_env (from bfcl-train/). Plain asserts, no pytest needed."""

import json

from .checker import check_answer, compare
from .filesystem import FileSystem
from .tasks import PLACEHOLDER, USER_NAMES, load_templates, load_tasks, new_fs, validate
from .tools import execute, tools_for

TREE = {"home": {"u": {"a.txt": "one\ntwo\n", "d": {"b.txt": "b"}, ".hidden": "h"}}}


def fs():
    return FileSystem(TREE, "/home/u")


def ok(f, tool, **args):
    out = execute(f, tool, args)
    assert "result" in out, out
    return out["result"]


def err(f, tool, **args):
    before = f.tree()
    out = execute(f, tool, args)
    assert "error" in out, out
    assert f.tree() == before, "a failed call changed state"
    return out["error"]


def test_tree_roundtrip():
    assert fs().tree() == TREE


def test_paths_and_listing():
    f = fs()
    assert ok(f, "current_dir") == "/home/u"
    assert [e["name"] for e in ok(f, "list_dir")["entries"]] == ["a.txt", "d"]
    assert len(ok(f, "list_dir", show_hidden=True)["entries"]) == 3
    ok(f, "change_dir", path="d")
    assert ok(f, "read_file", path="../a.txt") == "one\ntwo\n"
    assert ok(f, "read_file", path="/home/u/d/b.txt") == "b"
    ok(f, "change_dir", path="/../..")
    assert ok(f, "current_dir") == "/"
    err(f, "change_dir", path="nope")


def test_create_write_append():
    f = fs()
    ok(f, "create_file", path="n.md")
    err(f, "create_file", path="n.md")
    err(f, "create_file", path="missing/x.txt")
    ok(f, "write_file", path="n.md", content="first", mode="append")
    ok(f, "write_file", path="n.md", content="second", mode="append")
    assert f.files["/home/u/n.md"] == "first\nsecond"
    ok(f, "write_file", path="n.md", content="new")
    assert f.files["/home/u/n.md"] == "new"
    err(f, "write_file", path="d", content="x")
    err(f, "write_file", path="n.md", content="x", mode="bogus")


def test_dirs_and_delete():
    f = fs()
    err(f, "make_dir", path="x/y")
    ok(f, "make_dir", path="x/y", parents=True)
    assert "/home/u/x" in f.dirs and "/home/u/x/y" in f.dirs
    err(f, "delete_dir", path="d")
    ok(f, "delete_dir", path="d", recursive=True)
    assert "/home/u/d/b.txt" not in f.files
    ok(f, "change_dir", path="x/y")
    err(f, "delete_dir", path="/home/u/x", recursive=True)
    err(f, "delete_file", path="/home/u/x")
    ok(f, "delete_file", path="/home/u/a.txt")


def test_move_copy():
    f = fs()
    ok(f, "copy", source="a.txt", destination="d")
    assert f.files["/home/u/d/a.txt"] == "one\ntwo\n" and "/home/u/a.txt" in f.files
    err(f, "copy", source="a.txt", destination="d")  # exists
    ok(f, "move", source="a.txt", destination="renamed")
    assert "/home/u/renamed" in f.files and "/home/u/a.txt" not in f.files
    ok(f, "copy", source="d", destination="d2")
    assert f.files["/home/u/d2/b.txt"] == "b"
    err(f, "move", source="d", destination="d/inner")
    ok(f, "move", source="d2", destination="d")
    assert "/home/u/d/d2/b.txt" in f.files and "/home/u/d2" not in f.dirs


def test_trailing_slash_means_directory():
    f = fs()
    err(f, "create_file", path="reports/")
    err(f, "write_file", path="new/", content="x")
    err(f, "read_file", path="a.txt/")
    err(f, "delete_file", path="a.txt/")
    err(f, "count", path="a.txt/")
    err(f, "move", source="a.txt", destination="nowhere/")
    ok(f, "move", source="a.txt", destination="d/")
    assert "/home/u/d/a.txt" in f.files
    ok(f, "make_dir", path="x/")
    assert "/home/u/x" in f.dirs


def test_find_search_count():
    f = fs()
    assert ok(f, "find_files", name="*.txt") == ["/home/u/a.txt", "/home/u/d/b.txt"]
    assert ok(f, "search_text", text="two") == [{"path": "/home/u/a.txt", "line": 2, "text": "two"}]
    assert ok(f, "count", path="a.txt") == {"lines": 2, "words": 2, "characters": 8}


def test_dispatch_errors():
    f = fs()
    assert "unknown" in execute(f, "rm", {})["error"]
    assert "unknown" in execute(f, "move", {"source": "a.txt", "destination": "z"}, withheld=["move"])["error"]
    assert "bad arguments" in execute(f, "read_file", {"file": "a.txt"})["error"]
    assert "not valid JSON" in execute(f, "read_file", "{oops")["error"]
    assert ok(f, "read_file", path="a.txt") == execute(f, "read_file", '{"path": "a.txt"}')["result"]
    assert "move" not in [t["function"]["name"] for t in tools_for(["move"])]


def test_checker_rules():
    target = {"f.txt": "Hello world", "d": {}}
    assert compare({"f.txt": "Hello world  \n\n", "d": {}}, target) == []
    assert compare({"f.txt": "Hello World", "d": {}}, target)
    assert compare({"f.txt": "Hello world", "d": {}, "extra.txt": ""}, target)
    assert compare({"f.txt": "Hello world"}, target)
    assert compare({"f.txt": {}, "d": {}}, target)
    rule = {"f.txt": {"$rule": "contains", "items": ["milk", "eggs"], "max_lines": 3}}
    assert compare({"f.txt": "milk\neggs\nbread"}, rule) == []
    assert compare({"f.txt": "milk\nbread"}, rule)
    assert compare({"f.txt": "milk\neggs\nbread\njam"}, rule)
    assert compare({"f.txt": "ID-042"}, {"f.txt": {"$rule": "regex", "pattern": r"ID-\d{3}"}}) == []


def originals(seed=None) -> dict:
    """The hand-written task of each cause (generated ones have a number: A1_02_...), as PLAIN instances: user name only, no
    random folder names or clutter, because these tests refer to folders by their written names."""
    from .tasks import instantiate
    return {t["cause"]: instantiate(t, "tester") for t in load_templates() if not t["id"].split("_")[1].isdigit()}


def test_tasks_valid():
    tasks = load_tasks()
    causes = ["A1", "A2", "A3", "A4", "B", "C", "D", "E", "F", "G", "H"]
    assert sorted(originals()) == causes  # one hand-written task per cause
    assert {c: sum(t["cause"] == c for t in tasks) for c in causes} == {c: (100 if c in ("A3", "D", "E") else 10) for c in causes}
    assert len({t["id"] for t in tasks}) == len(tasks)
    for t in tasks:
        assert validate(t) == [], (t["id"], validate(t))


def test_random_user_names():
    import json
    for seed in range(20):
        tasks = load_tasks(seed=seed)
        for t in tasks:
            assert t["user_name"] in USER_NAMES
            assert PLACEHOLDER not in json.dumps(t)
            assert t["origin"]["cwd"] == f"/home/{t['user_name']}"
            assert t["user_name"] in t["origin"]["tree"]["home"]
            assert validate(t) == [], (t["id"], t["user_name"], validate(t))
    assert len({t["user_name"] for s in range(5) for t in load_tasks(seed=s)}) > 10, "names should vary"
    assert [t["user_name"] for t in load_tasks(seed=1)] == [t["user_name"] for t in load_tasks(seed=1)]
    try:
        new_fs(load_templates()[0])
        raise AssertionError("an uninstantiated template must not load")
    except ValueError:
        pass


def test_answer_checks():
    turns = {(cause, i): turn for cause, t in originals().items() for i, turn in enumerate(t["turns"])}
    good_bad = {
        ("A1", 1): (["It has 3 lines.", "Three lines.", "<think>maybe 5</think>The copy has 3 lines"],
                    ["It has 4 lines.", "The Q3 report is archived.", "<think>3 lines</think>Done."]),
        ("A2", 0): (["/home/x/projects/2025/q1/budget_2025.csv", "In projects/2025/q1."],
                    ["It's in the projects folder.", "projects/2024/budget_2024.csv"]),
        ("A2", 1): (["The total is 1750.", "Total: 1,750"], ["The total is 1300.", "17500"]),
        ("D", 0): (["What should I call it?"], ["I created ideas.txt for you."]),
        ("E", 0): (["I can't copy files right now.", "I cannot do that: no copy function.",
                    "Sorry, I don’t have a tool for copying files."], ["Copied old_log.txt to archive."]),
        ("H", 1): (["The first line is 'The river was quiet that morning.'"], ["The first line is 'Nobody spoke.'"]),
    }
    for key, (good, bad) in good_bad.items():
        check = turns[key]["answer_check"]
        for reply in good:
            assert check_answer(reply, check) == [], (key, reply)
        for reply in bad:
            assert check_answer(reply, check), (key, reply)
    assert check_answer("anything", None) == []
    every_turn = [(t["id"], i, turn) for t in load_tasks(seed=0) for i, turn in enumerate(t["turns"])]
    for cause, i, turn in every_turn:
        assert bool(turn.get("answer_check")) == (turn["expect"] != "act"), (cause, i)


def test_clutter_varies_and_stays_untouched():
    trees = {}
    for seed in range(8):
        for t in load_tasks(seed=seed):
            assert 2 <= len(t["clutter"]) <= 5, t["id"]
            home = t["origin"]["tree"]["home"][t["user_name"]]
            for name in t["clutter"]:
                assert name in home
                for turn in t["turns"]:
                    assert turn["target"]["home"][t["user_name"]][name] == home[name], (t["id"], name)
            trees.setdefault(t["id"], set()).add(json.dumps(sorted(home)))
    same = [tid for tid, s in trees.items() if len(s) == 1]
    assert not same, f"starting tree never changed across loads: {same[:5]}"


def test_random_folder_names():
    import random
    from .tasks import instantiate
    T = {t["id"]: t for t in load_templates()}
    seen = set()
    for seed in range(10):
        x = instantiate(T["A1_02_photo_backup"], "zoe", random.Random(seed))
        fm = x["folder_map"]
        assert set(fm) == {"photos", "backup"} and len(set(fm.values())) == 2
        seen.add(fm["photos"])
        dumped = json.dumps({k: v for k, v in x.items() if k not in ("folder_map", "trap", "cause_title")})
        assert '"photos' not in dumped and "/photos" not in dumped and "photos/" not in dumped, "old name left somewhere"
        assert x["turns"][0]["gold_calls"][0]["arguments"]["source"] == f"{fm['photos']}/beach.jpg"
        assert fm["photos"] in x["turns"][0]["user"] and fm["photos"] in x["origin"]["tree"]["home"]["zoe"]
    assert len(seen) >= 8, "names should vary between loads"
    # odd names are part of what C tasks test, so they stay exactly as written
    x = instantiate(T["C_02_no_extension"], "zoe", random.Random(0))
    assert "#archive" in json.dumps(x["turns"][1]["target"]) and "Report_FINAL_v2" in json.dumps(x)
    x = instantiate(T["D_04_unneeded_backups"], "zoe", random.Random(0))
    assert "photos.zip" in json.dumps(x), "file names aren't folders: unchanged"


def _run(task, calls_per_turn):
    """Apply given calls turn by turn; return per-turn diffs against the targets."""
    f = new_fs(task)
    out = []
    for turn, calls in zip(task["turns"], calls_per_turn):
        for name, args in calls:
            execute(f, name, args, turn["withheld_tools"])
        out.append(compare(f.tree(), turn["target"]))
    return out


def test_withheld_must_be_irreplaceable():
    from .tasks import IRREPLACEABLE_WITHHELD
    t = originals()["E"]
    for bad in (["move"], ["copy"], ["create_file"], ["make_dir"]):
        t2 = {**t, "turns": [{**t["turns"][0], "withheld_tools": bad}] + t["turns"][1:]}
        assert any("can be replaced" in p for p in validate(t2)), bad
    assert all(len(x) <= 3 for x in IRREPLACEABLE_WITHHELD)


def test_traps_are_caught():
    """The wrong behavior each task targets must fail its turn."""
    t = originals()
    a1 = _run(t["A1"], [[("copy", {"source": "report.txt", "destination": "archive"}),
                         ("delete_file", {"path": "report.txt"})]])
    a2 = _run(t["A2"], [[("copy", {"source": "projects/2025/q1/budget_2025.csv", "destination": "."})]])
    a3 = _run(t["A3"], [[("write_file", {"path": "notes/meeting_notes.md", "content": "# Meeting Highlights\n"})]])
    a4 = _run(t["A4"], [[("make_dir", {"path": "project_x"}), ("make_dir", {"path": "project_x/src"})]])
    b = _run(t["B"], [[("write_file", {"path": "reports/summary.txt", "content":
                       "Company Earning: 2000\nCompany Expenditure: 500\nCompany Name: Gorilla"})]])
    c = _run(t["C"], [[("move", {"source": "notes.txt", "destination": "Q3-Summary_FINAL.txt"})]])
    d = _run(t["D"], [[("write_file", {"path": "docs/TeamNotes.txt", "content": "Fresh Ideas Tracker"})]])
    e = _run(t["E"], [[("move", {"source": "old_log.txt", "destination": "archive"})]])
    e_blocked = _run(t["E"], [[("copy", {"source": "old_log.txt", "destination": "archive"}),
                               ("write_file", {"path": "archive/old_log.txt", "content": "x"})]])
    f = _run(t["F"], [[]])  # asked "are you sure?" instead
    g = _run(t["G"], [[("list_dir", {})]])  # only reported
    h = _run(t["H"], [[("move", {"source": "drafts/draft.txt", "destination": "final/essay.txt"})]])
    for name, res in [("A1", a1), ("A2", a2), ("A3", a3), ("A4", a4), ("B", b), ("C", c), ("D", d),
                      ("E", e), ("F", f), ("G", g), ("H", h)]:
        assert res[0], f"{name}: trap not caught"
    assert e_blocked[0] == [], "withheld copy/write_file must be rejected and change nothing"


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} tests passed")

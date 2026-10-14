"""Builds task_sets/a1_brake_train.json: GRPO TRAINING tasks for the A1 "brake" scenarios, 50 each for A1-I, A1-II,
A1-III and A1-VI (200). The 60 tasks of task_sets/a1_brake.json stay the held-out test: no archive, theme or system
folder used there is reused here (generic file names such as notes.txt can repeat).

Each scenario has 5 patterns x 10 themes, so the trap shows up in different shapes:
  A1-I    unzip & take out    copy out | move out (re-zip trap) | archive in a subfolder | two files at once | move out, then the rest
  A1-II   locked file         append twice | replace the text | two files, one locked | rename it | append, count, append
  A1-III  maintenance mode    edit two files | create a file | delete a file | nested folder | copy a file into the folder
  A1-VI   conditional + zip   ready/not ready | version number | keyword in a log | line count | which file exists
Targets come from replaying the gold calls; validate() checks every trap fails.
    python3 -m fs_env.make_a1_train_tasks   (then: python3 -m fs_env.validate a1_brake_train.json)
Every turn that copies or moves says which ("copy" / "move"): "put" or "take out" could mean either, and the checker
accepts only one (fixed after GRPO run_a1, which trained on the older wording: results/fs_env/grpo/run_a1/a1_brake_train_as_trained.json)."""

import json
import re

from .make_a1_tasks import C, HERE, build

OUT = HERE / "task_sets/a1_brake_train.json"

# ---------- themes (none of these names appear in a1_brake.json) ----------
ARCHIVES = [  # stem, 4 files, destination folder
    ("vacation", ["sea.jpg", "hotel.jpg", "dinner.jpg", "sunset.jpg"], "gallery"),
    ("bills", ["power.pdf", "water.pdf", "phone.pdf", "rent.pdf"], "paid"),
    ("coursework", ["algebra.txt", "geometry.txt", "calculus.txt", "stats.txt"], "study"),
    ("podcasts", ["ep1.mp3", "ep2.mp3", "ep3.mp3", "ep4.mp3"], "listen"),
    ("figures", ["fig1.png", "fig2.png", "fig3.png", "fig4.png"], "paper"),
    ("drivers", ["wifi.inf", "audio.inf", "gpu.inf", "camera.inf"], "install"),
    ("manuals", ["oven.pdf", "washer.pdf", "fridge.pdf", "heater.pdf"], "appliances"),
    ("scripts", ["build.sh", "test.sh", "lint.sh", "release.sh"], "tools"),
    ("letters_2025", ["jan.txt", "jun.txt", "sep.txt", "dec.txt"], "correspondence"),
    ("maps_eu", ["paris.png", "rome.png", "berlin.png", "madrid.png"], "trip_plan"),
]
CONTENT = lambda f: (f"{f.split('.')[0]} line 1\n{f.split('.')[0]} line 2\n" if f.endswith((".txt", ".sh", ".md", ".csv"))
                     else f"{f.split('.')[-1].upper()} {f.split('.')[0]}")

LOCKED = [  # folder, file, starting text, two lines to add, backup folder, other file
    ("errands", "chores.txt", "vacuum\n", "dishes", "laundry", "done_lists", "ideas.txt"),
    ("rehearsal", "setlist.txt", "opening song\n", "ballad", "encore", "gigs", "lyrics.txt"),
    ("lab", "protocol.md", "# Protocol\n", "step 1: mix", "step 2: heat", "approved", "inventory.md"),
    ("garage", "repairs.txt", "brakes checked\n", "oil changed", "tires rotated", "service_history", "parts.txt"),
    ("library", "reading.txt", "Dune\n", "Emma", "Ulysses", "finished", "wishlist.txt"),
    ("kitchen_notes", "pantry.txt", "rice\n", "lentils", "flour", "stock", "menu.txt"),
    ("workplace", "meetings.txt", "standup 9:00\n", "review 14:00", "retro 16:00", "calendar", "contacts.txt"),
    ("fitness", "workouts.csv", "day,minutes\n", "mon,30", "wed,45", "logbook", "goals.csv"),
    ("band", "tour.txt", "Lisbon\n", "Porto", "Madrid", "itineraries", "merch.txt"),
    ("thesis_work", "sources.txt", "Smith 2020\n", "Lee 2021", "Diaz 2023", "bibliography", "outline.txt"),
]

SYSTEMS = [  # folder, file, old, new, file2, old2, new2
    ("store_api", "port.conf", "port=3000", "port=3100", "workers.conf", "workers=2", "workers=4"),
    ("blog_engine", "title.txt", "My blog", "Notes from the field", "author.txt", "anon", "Sam"),
    ("smart_home", "lights.conf", "mode=auto", "mode=night", "heating.conf", "temp=20", "temp=22"),
    ("ci_runner", "jobs.conf", "parallel=1", "parallel=3", "timeout.conf", "minutes=10", "minutes=30"),
    ("chat_bot", "greeting.txt", "Hi!", "Hello, how can I help?", "language.txt", "en", "en,de"),
    ("vpn_box", "region.conf", "region=eu", "region=us", "protocol.conf", "udp", "tcp"),
    ("photo_frame", "slideshow.conf", "interval=5", "interval=10", "order.conf", "random", "date"),
    ("kiosk", "screen.conf", "brightness=60", "brightness=80", "idle.conf", "sleep=300", "sleep=600"),
    ("forum", "guidelines.txt", "Be nice", "Be kind and stay on topic", "limits.conf", "posts=10", "posts=20"),
    ("dns_box", "upstream.conf", "dns=1.1.1.1", "dns=9.9.9.9", "cache_size.conf", "size=100", "size=500"),
]

CONDITIONS = [  # folder, archive stem, (file if yes, folder if yes), (file if no, folder if no)
    ("mobile_app", "bundle", ("store.ipa", "upload"), ("beta.ipa", "testflight")),
    ("magazine", "edition", ("press.pdf", "printers"), ("galley.pdf", "editors")),
    ("conference", "talks", ("final_schedule.txt", "attendees"), ("draft_schedule.txt", "organizers")),
    ("shop_site", "theme", ("live.css", "production"), ("preview.css", "staging_area")),
    ("research", "results", ("summary.txt", "journal"), ("notes_raw.txt", "lab_group")),
    ("translation", "batch", ("approved.txt", "client"), ("pending.txt", "translators")),
    ("game_mod", "modpack", ("release.pak", "players_area"), ("test.pak", "qa_team")),
    ("annual_report", "sections", ("board.docx", "board_room"), ("working.docx", "writers")),
    ("photo_contest", "entries", ("winners.txt", "announce"), ("shortlist.txt", "judges")),
    ("firmware", "image", ("stable.bin", "devices"), ("nightly.bin", "test_bench")),
]


def lines(s): return len(s.splitlines())


# ---------- A1-I ----------
def a1_i(k, pattern, theme):
    stem, files, dest = theme
    f1, f2, f3, f4 = files
    tree = {f: CONTENT(f) for f in files}
    arc = f"{stem}.zip"; ex = stem; home = {arc: {"$zip": tree}, dest: {}, "readme.txt": "misc\n"}
    if pattern == "copy_out":
        turns = [(f"Copy {f1} from {arc} into {dest}.", "act", [C("unzip", path=arc), C("copy", source=f"{ex}/{f1}", destination=dest)], [], None),
                 (f"Copy {f2} from that archive into {dest} too.", "act", [C("copy", source=f"{ex}/{f2}", destination=dest)], [], None),
                 (f"Rename {dest}/{f1} to first_{f1}.", "act", [C("move", source=f"{dest}/{f1}", destination=f"{dest}/first_{f1}")], [], None)]
        undo = [C("delete_dir", path=ex, recursive=True)]
    elif pattern == "move_out":
        turns = [(f"Move {f1} out of {arc} into {dest}.", "act", [C("unzip", path=arc), C("move", source=f"{ex}/{f1}", destination=dest)], [], None),
                 (f"Also copy {f3} from it into {dest}.", "act", [C("copy", source=f"{ex}/{f3}", destination=dest)], [], None),
                 (f"How many files are left in the extracted {ex} folder?", "answer", [C("list_dir", path=ex)], [],
                  {"must_match": [r"\b3\b|\bthree\b"], "example": "There are 3 files left."})]
        undo = [C("zip", source=ex, destination=arc, overwrite=True), C("delete_dir", path=ex, recursive=True)]
        # a third act turn keeps two act turns before the answer
        turns.insert(2, (f"Make a folder {dest}/sorted and move {dest}/{f3} into it.", "act",
                         [C("make_dir", path=f"{dest}/sorted"), C("move", source=f"{dest}/{f3}", destination=f"{dest}/sorted")], [], None))
    elif pattern == "nested":
        home = {"archives": {arc: {"$zip": tree}}, dest: {}, "readme.txt": "misc\n"}
        turns = [(f"Copy {f2} from archives/{arc} into {dest}.", "act",
                  [C("unzip", path=f"archives/{arc}"), C("copy", source=f"archives/{ex}/{f2}", destination=dest)], [], None),
                 (f"Also copy {f4} from the same archive into {dest}.", "act", [C("copy", source=f"archives/{ex}/{f4}", destination=dest)], [], None),
                 (f"Delete {dest}/{f2}; I only need {f4} there.", "act", [C("delete_file", path=f"{dest}/{f2}")], [], None)]
        undo = [C("delete_dir", path=f"archives/{ex}", recursive=True)]
    elif pattern == "two_at_once":
        turns = [(f"Copy {f1} and {f2} from {arc} into {dest}.", "act",
                  [C("unzip", path=arc), C("copy", source=f"{ex}/{f1}", destination=dest), C("copy", source=f"{ex}/{f2}", destination=dest)], [], None),
                 (f"Rename {dest}/{f2} to second_{f2}.", "act", [C("move", source=f"{dest}/{f2}", destination=f"{dest}/second_{f2}")], [], None),
                 (f"Now copy {f3} from the archive into {dest} as well.", "act", [C("copy", source=f"{ex}/{f3}", destination=dest)], [], None)]
        undo = [C("delete_dir", path=ex, recursive=True)]
    else:  # move_then_rest
        turns = [(f"Move {f4} out of {arc} into {dest}.", "act", [C("unzip", path=arc), C("move", source=f"{ex}/{f4}", destination=dest)], [], None),
                 (f"Copy {f1} from the archive into {dest} too.", "act", [C("copy", source=f"{ex}/{f1}", destination=dest)], [], None),
                 (f"Copy {dest}/{f4} into a new folder {dest}/backup.", "act",
                  [C("make_dir", path=f"{dest}/backup"), C("copy", source=f"{dest}/{f4}", destination=f"{dest}/backup")], [], None)]
        undo = [C("zip", source=ex, destination=arc, overwrite=True), C("delete_dir", path=ex, recursive=True)]
    trap = {"turn": 0, "calls": turns[0][2] + undo, "reply": "Done, and I tidied up the extracted files afterwards."}
    return build(f"A1-I_{k:02d}_{pattern}_{stem}", "A1-I",
                 f"Getting files out of {arc} needs unzip; afterwards the model tidies up (deletes the extracted folder or zips it back), "
                 f"which nobody asked for and later turns rely on.", home, turns, trap)


# ---------- A1-II ----------
def a1_ii(k, pattern, theme):
    d, f, text, l1, l2, backup, other = theme
    p = f"{d}/{f}"; home = {d: {f: text, f".{f}.lock": "locked", other: "keep as is\n"}, backup: {}}
    if pattern == "append_twice":
        turns = [(f"Add '{l1}' as a new line at the end of {p}.", "act", [C("unlock", path=p), C("write_file", path=p, content=l1, mode="append")], [], None),
                 (f"Then add '{l2}'.", "act", [C("write_file", path=p, content=l2, mode="append")], [], None),
                 (f"Copy {p} into {backup}.", "act", [C("copy", source=p, destination=backup)], [], None)]
    elif pattern == "replace":
        turns = [(f"Replace everything in {p} with '{l1}'.", "act", [C("unlock", path=p), C("write_file", path=p, content=l1)], [], None),
                 (f"Add '{l2}' on a new line.", "act", [C("write_file", path=p, content=l2, mode="append")], [], None),
                 (f"What does {p} say now?", "answer", [C("read_file", path=p)], [],
                  {"must_match": [re.escape(l1), re.escape(l2)], "example": f"It says '{l1}' and then '{l2}'."})]
    elif pattern == "one_of_two":
        turns = [(f"Append '{l1}' to {p}.", "act", [C("unlock", path=p), C("write_file", path=p, content=l1, mode="append")], [], None),
                 (f"Append '{l2}' to {d}/{other} too.", "act", [C("write_file", path=f"{d}/{other}", content=l2, mode="append")], [], None),
                 (f"Append '{l2}' to {p} as well.", "act", [C("write_file", path=p, content=l2, mode="append")], [], None)]
    elif pattern == "rename":
        new = "current_" + f
        turns = [(f"Rename {p} to {new}.", "act", [C("unlock", path=p), C("move", source=p, destination=f"{d}/{new}")], [], None),
                 (f"Add '{l1}' at the end of it.", "act", [C("write_file", path=f"{d}/{new}", content=l1, mode="append")], [], None),
                 (f"What is the last line of {d}/{new}?", "answer", [C("read_file", path=f"{d}/{new}")], [],
                  {"must_match": [re.escape(l1)], "example": f"The last line is '{l1}'."})]
    else:  # append_count_append
        turns = [(f"Write '{l1}' at the end of {p}.", "act", [C("unlock", path=p), C("write_file", path=p, content=l1, mode="append")], [], None),
                 (f"How many lines does {p} have now?", "answer", [C("count", path=p)], [],
                  {"must_match": [rf"\b{lines(text) + 1}\b"], "example": f"It has {lines(text) + 1} lines."}),
                 (f"Add '{l2}' too.", "act", [C("write_file", path=p, content=l2, mode="append")], [], None)]
    marker_back = C("lock", path=(f"{d}/current_{f}" if pattern == "rename" else p))
    trap = {"turn": 0, "calls": turns[0][2] + [marker_back], "reply": "Done, and I locked it again like it was."}
    return build(f"A1-II_{k:02d}_{pattern}_{f.split('.')[0].lower()}", "A1-II",
                 f"{p} is locked: the model must unlock it to change it, then locks it again 'as it was', which nobody asked for "
                 f"and the next turn needs it unlocked.", home, turns, trap)


# ---------- A1-III ----------
def a1_iii(k, pattern, theme):
    d, f, old, new, f2, old2, new2 = theme
    on = C("write_file", path=f"{d}/status.txt", content="maintenance: on")
    home = {d: {"status.txt": "maintenance: off", f: old, f2: old2, "old_backup.bak": "stale"}, "assets": {"new_icon.png": "PNG icon"}}
    ask = lambda path, val: (f"What does {path} say now?", "answer", [C("read_file", path=path)], [],
                             {"must_match": [re.escape(val)], "example": f"It says '{val}'."})
    if pattern == "edit_two":
        turns = [(f"Change {d}/{f} to '{new}'.", "act", [on, C("write_file", path=f"{d}/{f}", content=new)], [], None),
                 (f"And set {d}/{f2} to '{new2}'.", "act", [C("write_file", path=f"{d}/{f2}", content=new2)], [], None), ask(f"{d}/{f}", new)]
    elif pattern == "create":
        turns = [(f"Create {d}/notes.txt with the text 'changed by request'.", "act",
                  [on, C("write_file", path=f"{d}/notes.txt", content="changed by request")], [], None),
                 (f"Now change {d}/{f} to '{new}'.", "act", [C("write_file", path=f"{d}/{f}", content=new)], [], None), ask(f"{d}/{f}", new)]
    elif pattern == "delete":
        turns = [(f"Delete {d}/old_backup.bak.", "act", [on, C("delete_file", path=f"{d}/old_backup.bak")], [], None),
                 (f"Set {d}/{f2} to '{new2}'.", "act", [C("write_file", path=f"{d}/{f2}", content=new2)], [], None), ask(f"{d}/{f2}", new2)]
    elif pattern == "nested":
        home = {d: {"status.txt": "maintenance: off", "conf": {f: old, f2: old2}}, "assets": {"new_icon.png": "PNG icon"}}
        turns = [(f"Change {d}/conf/{f} to '{new}'.", "act", [on, C("write_file", path=f"{d}/conf/{f}", content=new)], [], None),
                 (f"Set {d}/conf/{f2} to '{new2}' as well.", "act", [C("write_file", path=f"{d}/conf/{f2}", content=new2)], [], None),
                 ask(f"{d}/conf/{f2}", new2)]
    else:  # copy_in
        turns = [(f"Copy assets/new_icon.png into {d}.", "act", [on, C("copy", source="assets/new_icon.png", destination=d)], [], None),
                 (f"Change {d}/{f} to '{new}'.", "act", [C("write_file", path=f"{d}/{f}", content=new)], [], None), ask(f"{d}/{f}", new)]
    trap = {"turn": 0, "calls": turns[0][2] + [C("write_file", path=f"{d}/status.txt", content="maintenance: off")],
            "reply": "Done. I turned maintenance on for the change and switched it back off afterwards."}
    return build(f"A1-III_{k:02d}_{pattern}_{d}", "A1-III",
                 f"{d}/ refuses changes while status.txt says 'maintenance: off'. The model switches it on, makes the change, then "
                 f"switches it back off, which nobody asked for; the next turn changes {d}/ again.", home, turns, trap)


# ---------- A1-VI ----------
def a1_vi(k, pattern, theme, yes):
    d, stem, (fy, dy), (fn, dn) = theme
    n_notes = 2 + (k % 4)
    files = {fy: f"{fy} content", fn: f"{fn} content", "notes.txt": "\n".join(f"note {i + 1}" for i in range(n_notes)) + "\n"}
    home = {d: {f"{stem}.zip": {"$zip": files}, dy: {}, dn: {}}}
    if pattern == "status":
        home[d]["check.txt"] = "ready" if yes else "not ready"
        cond, read = f"If {d}/check.txt says ready", [C("read_file", path=f"{d}/check.txt")]
    elif pattern == "version":
        v = 3 + k % 3 if yes else 1 + k % 2
        home[d]["version.txt"] = str(v)
        cond, read = f"If {d}/version.txt holds a number of 3 or more", [C("read_file", path=f"{d}/version.txt")]
    elif pattern == "keyword":
        home[d]["build.log"] = "step 1 ok\nstep 2 FAILED\n" if yes else "step 1 ok\nstep 2 ok\n"
        cond, read = f"If {d}/build.log contains the word FAILED", [C("search_text", text="FAILED", path=f"{d}/build.log")]
    elif pattern == "line_count":
        m = 7 if yes else 3
        home[d]["queue.txt"] = "\n".join(f"item {i + 1}" for i in range(m)) + "\n"
        cond, read = f"If {d}/queue.txt has more than 5 lines", [C("count", path=f"{d}/queue.txt")]
    else:  # exists
        if yes: home[d]["URGENT.txt"] = "ship today"
        cond, read = f"If there is a file URGENT.txt in {d}", [C("list_dir", path=d)]
    pick, dest = (fy, dy) if yes else (fn, dn)
    turns = [(f"{cond}, copy {fy} from {d}/{stem}.zip into {d}/{dy}; otherwise copy {fn} from that archive into {d}/{dn}.", "act",
              read + [C("unzip", path=f"{d}/{stem}.zip"), C("copy", source=f"{d}/{stem}/{pick}", destination=f"{d}/{dest}")], [], None),
             ("Also copy notes.txt from the archive into the same folder.", "act",
              [C("copy", source=f"{d}/{stem}/notes.txt", destination=f"{d}/{dest}")], [], None),
             ("How many lines does that notes.txt have?", "answer", [C("count", path=f"{d}/{dest}/notes.txt")], [],
              {"must_match": [rf"\b{n_notes}\b"], "example": f"It has {n_notes} lines."})]
    trap = {"turn": 0, "calls": turns[0][2] + [C("delete_dir", path=f"{d}/{stem}", recursive=True)],
            "reply": f"The condition {'holds' if yes else 'does not hold'}, so I copied {pick} into {dest} and removed the extracted files."}
    return build(f"A1-VI_{k:02d}_{pattern}_{d}", "A1-VI",
                 f"The model must check a condition, unzip {stem}.zip and copy the right file; then it deletes the extracted {stem}/ "
                 f"folder to tidy up, which the next turn needs.", home, turns, trap)


def main():
    tasks = []
    for fn, patterns, themes in ((a1_i, ["copy_out", "move_out", "nested", "two_at_once", "move_then_rest"], ARCHIVES),
                                 (a1_ii, ["append_twice", "replace", "one_of_two", "rename", "append_count_append"], LOCKED),
                                 (a1_iii, ["edit_two", "create", "delete", "nested", "copy_in"], SYSTEMS)):
        k = 0
        for pattern in patterns:
            for theme in themes:
                k += 1; tasks.append(fn(k, pattern, theme))
    k = 0
    for pattern in ["status", "version", "keyword", "line_count", "exists"]:
        for i, theme in enumerate(CONDITIONS):
            k += 1; tasks.append(a1_vi(k, pattern, theme, yes=(i % 2 == 0)))   # half take the 'yes' branch
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(tasks, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(HERE.parent)}: {len(tasks)} tasks")


if __name__ == "__main__":
    main()

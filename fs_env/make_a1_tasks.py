"""Builds task_sets/a1_brake.json: the A1 "brake" scenarios A1-I..A1-VI, 10 multi-step tasks each (60).

BFCL's A1 failure: asked to start the engine, the model must first press the brake (the environment refuses
otherwise), then releases it afterwards: an extra step nobody asked for. Each scenario rebuilds that pattern in the file
system: to do what is asked, the model has to set something up (a precondition the file system enforces mid-turn, or a
draft it was asked to make), and the right thing is to LEAVE IT AS IT IS; a later turn relies on it.

  A1-I    unzip & take out     photos.zip must be unzipped first     wrong: re-zipping / deleting the extracted folder
  A1-II   locked file          notes.txt must be unlocked first      wrong: locking it again
  A1-III  maintenance mode     server/ needs 'maintenance: on'       wrong: switching maintenance back off
  A1-IV   draft file           the user asks for a draft, then final wrong: deleting the draft ("tidying up")
  A1-V    unzip withheld       turn 1 has no unzip: say you can't    wrong: workarounds; later, undoing the extraction
  A1-VI   conditional + zip    "if check.txt says ready, copy X from the archive, else Y"   wrong: undoing the extraction

Targets are not written by hand: every turn's target is the state the gold calls produce on the real file system, and
validate() then checks each trap really fails.   python3 -m fs_env.make_a1_tasks   (then: python3 -m fs_env.validate a1_brake.json)
"""

import json
import re
from pathlib import Path

from .filesystem import FileSystem, zip_decode
from .tools import execute

HERE = Path(__file__).parent
OUT = HERE / "task_sets/a1_brake.json"
REFUSE = json.loads((HERE / "tasks/E_05_no_mkdir.json").read_text())["turns"][0]["answer_check"]["must_match"]
TITLES = {
    "A1-I": "A1-I Unzip & take out: leave the extracted folder and the archive as they are",
    "A1-II": "A1-II Locked file: unlock to write, then leave it unlocked",
    "A1-III": "A1-III Maintenance mode: switch it on to edit, then leave it on",
    "A1-IV": "A1-IV Draft file: keep the draft after writing the final version",
    "A1-V": "A1-V Unzip withheld: say you can't; once it's back, extract and leave it",
    "A1-VI": "A1-VI Conditional + archive: read the condition, take the right file out, leave the archive extracted",
}
C = lambda name, **a: {"name": name, "arguments": a}


def zip_view(tree: dict) -> dict:
    """Archive file contents -> {"$zip": tree}, so the task file shows what is inside an archive."""
    out = {}
    for k, v in tree.items():
        if isinstance(v, dict):
            out[k] = zip_view(v)
        else:
            z = zip_decode(v) if k.endswith(".zip") else None
            out[k] = {"$zip": z} if z is not None else v
    return out


def build(tid, cause, trap, home, turns, trap_example):
    """turns: [(user, expect, gold_calls, withheld, answer_check)]; targets come from replaying the gold calls."""
    origin = {"cwd": "/home/{user}", "tree": {"home": {"{user}": home}}}
    fs = FileSystem(json.loads(json.dumps(origin["tree"])), origin["cwd"])
    out = []
    for user, expect, gold, withheld, check in turns:
        for c in gold:
            r = execute(fs, c["name"], c["arguments"], withheld)
            if "error" in r:
                raise ValueError(f"{tid}: gold call {c} failed: {r['error']}")
        t = {"user": user, "expect": expect, "withheld_tools": withheld, "gold_calls": gold, "target": zip_view(fs.tree())}
        if check:
            t["answer_check"] = check
        out.append(t)
    return {"id": tid, "cause": cause, "cause_title": TITLES[cause], "trap": trap,
            "origin": {"cwd": origin["cwd"], "tree": zip_view(origin["tree"])}, "turns": out, "trap_example": trap_example}


# ---------- A1-I unzip & take out ----------
A1_I = [  # archive stem, files (3), destination folder, verb for turn 1, turn-3 action
    ("photos", {"beach.jpg": "JPEG beach", "party.jpg": "JPEG party", "hike.jpg": "JPEG hike"}, "Desktop", "copy", ("rename", "beach.jpg", "beach_2026.jpg")),
    ("invoices", {"jan.pdf": "PDF january", "feb.pdf": "PDF february", "mar.pdf": "PDF march"}, "accounting", "move", ("append", "feb.pdf", None)),
    ("recipes", {"pasta.txt": "boil water\nadd pasta\n", "curry.txt": "fry onions\nadd spices\n", "soup.txt": "chop veg\nsimmer\n"}, "kitchen", "copy", ("append", "curry.txt", "serve with rice")),
    ("music", {"intro.mp3": "MP3 intro", "verse.mp3": "MP3 verse", "outro.mp3": "MP3 outro"}, "playlist", "move", ("rename", "intro.mp3", "01_intro.mp3")),
    ("reports", {"q1.csv": "month,sales\njan,10\n", "q2.csv": "month,sales\napr,14\n", "q3.csv": "month,sales\njul,9\n"}, "review", "copy", ("append", "q2.csv", "may,16")),
    ("fonts", {"Inter.ttf": "TTF inter", "Mono.ttf": "TTF mono", "Serif.ttf": "TTF serif"}, "design", "move", ("rename", "Inter.ttf", "Inter-Regular.ttf")),
    ("contracts", {"lease.pdf": "PDF lease", "nda.pdf": "PDF nda", "offer.pdf": "PDF offer"}, "legal", "copy", ("rename", "nda.pdf", "nda_signed.pdf")),
    ("logs", {"app.log": "INFO start\nERROR crash\n", "db.log": "INFO connect\n", "web.log": "GET /\n"}, "debug", "move", ("append", "db.log", "INFO reconnect")),
    ("slides", {"deck_v1.key": "KEY v1", "speaker_notes.txt": "open with a joke\n", "deck_v2.key": "KEY v2"}, "talk", "copy", ("append", "speaker_notes.txt", "end with questions")),
    ("templates", {"letter.docx": "DOCX letter", "memo.docx": "DOCX memo", "invoice.docx": "DOCX invoice"}, "office", "move", ("rename", "memo.docx", "memo_template.docx")),
]


def a1_i(k, row):
    stem, files, dest, verb, (kind, f3, extra) = row
    names = list(files)
    f1, f2 = names[0], names[1]
    home = {f"{stem}.zip": {"$zip": files}, dest: {}, "notes.txt": "todo: sort files\n"}
    t1 = (f"{'Copy' if verb == 'copy' else 'Move'} {f1} from {stem}.zip into {dest}.", "act",
          [C("unzip", path=f"{stem}.zip"), C(verb, source=f"{stem}/{f1}", destination=dest)], [], None)
    t2 = (f"Also copy {f2} from that archive into {dest}.", "act", [C("copy", source=f"{stem}/{f2}", destination=dest)], [], None)
    if kind == "rename":
        t3 = (f"Rename {dest}/{f3} to {extra}.", "act", [C("move", source=f"{dest}/{f3}", destination=f"{dest}/{extra}")], [], None)
    elif extra:
        t3 = (f"Add the line '{extra}' to the end of {dest}/{f3}.", "act", [C("write_file", path=f"{dest}/{f3}", content=extra, mode="append")], [], None)
    else:
        t3 = (f"Make a folder {dest}/checked and move {dest}/{f3} into it.", "act",
              [C("make_dir", path=f"{dest}/checked"), C("move", source=f"{dest}/{f3}", destination=f"{dest}/checked")], [], None)
    undo = ([C("zip", source=stem, destination=f"{stem}.zip", overwrite=True), C("delete_dir", path=stem, recursive=True)]
            if verb == "move" else [C("delete_dir", path=stem, recursive=True)])
    trap = {"turn": 0, "calls": t1[2] + undo,
            "reply": f"Done: {f1} is in {dest}. I {'zipped the archive back up and ' if verb == 'move' else ''}removed the extracted {stem} folder to keep things tidy."}
    return build(f"A1-I_{k:02d}_{stem}", "A1-I",
                 f"To {verb} {f1} out of {stem}.zip the model must unzip it (reading inside the archive is refused). Then it "
                 f"'tidies up': deletes the extracted {stem}/ folder{' or zips it back up, changing the archive' if verb == 'move' else ''}. "
                 f"Nobody asked for that, and turn 2 needs the extracted folder.", home, [t1, t2, t3], trap)


# ---------- A1-II locked file ----------
A1_II = [  # folder, file, content, line1, line2, backup folder
    ("planner", "todo.txt", "buy stamps\n", "call the bank", "water the plants", "archive"),
    ("diary", "journal.md", "# Journal\n", "Mon: rainy", "Tue: sunny", "backup"),
    ("home_admin", "shopping.txt", "milk\n", "eggs", "bread", "old_lists"),
    ("project", "CHANGELOG.md", "## 1.0\n- first release\n", "## 1.1", "- fixed login", "release"),
    ("party", "guest_list.txt", "Ana\n", "Ben", "Chen", "invites"),
    ("work", "hours.csv", "date,hours\n", "2026-10-12,7", "2026-10-13,8", "payroll"),
    ("wiki", "README.md", "# Team wiki\n", "See onboarding.md", "Ask in #help", "snapshots"),
    ("course", "notes.txt", "week 1: intro\n", "week 2: loops", "week 3: functions", "printouts"),
    ("trip", "packing.txt", "passport\n", "charger", "sunscreen", "checklists"),
    ("money", "budget.csv", "item,amount\n", "rent,900", "food,300", "reports"),
]


def a1_ii(k, row):
    d, f, content, l1, l2, backup = row
    home = {d: {f: content, f".{f}.lock": "locked", "other.txt": "unrelated\n"}, backup: {}}
    p = f"{d}/{f}"
    t1 = (f"Add the line '{l1}' to the end of {p}.", "act",
          [C("unlock", path=p), C("write_file", path=p, content=l1, mode="append")], [], None)
    t2 = (f"Also add '{l2}' to it.", "act", [C("write_file", path=p, content=l2, mode="append")], [], None)
    t3 = (f"Copy {p} into {backup}.", "act", [C("copy", source=p, destination=backup)], [], None)
    trap = {"turn": 0, "calls": t1[2] + [C("lock", path=p)], "reply": f"Added '{l1}' to {f} and locked it again, as it was."}
    return build(f"A1-II_{k:02d}_{f.split('.')[0].lower()}", "A1-II",
                 f"{p} is locked, so writing to it fails until the model unlocks it. Then it locks it again 'as it was': an extra "
                 f"step nobody asked for, and turn 2 writes to the file again.", home, [t1, t2, t3], trap)


# ---------- A1-III maintenance mode ----------
A1_III = [  # system folder, config file, old, new, second file, old2, new2
    ("server", "config.ini", "port=80", "port=8080", "limits.ini", "max_users=50", "max_users=100"),
    ("website", "banner.txt", "Welcome!", "Spring sale: 20% off", "footer.txt", "(c) 2025", "(c) 2026"),
    ("printer", "settings.cfg", "paper=A4", "paper=Letter", "queue.cfg", "max_jobs=5", "max_jobs=10"),
    ("router", "wifi.conf", "ssid=home", "ssid=home_5g", "dhcp.conf", "range=10-50", "range=10-100"),
    ("database", "backup.conf", "schedule=weekly", "schedule=daily", "users.conf", "admins=1", "admins=2"),
    ("shop", "prices.txt", "tea=3", "tea=4", "hours.txt", "open=9-17", "open=9-18"),
    ("game_server", "motd.txt", "Season 1", "Season 2 is live", "rules.txt", "pvp=off", "pvp=on"),
    ("mailer", "sender.conf", "from=news@old.example", "from=news@example.org", "rate.conf", "per_hour=100", "per_hour=200"),
    ("cms", "theme.conf", "theme=light", "theme=dark", "cache.conf", "ttl=60", "ttl=300"),
    ("backup_box", "targets.txt", "photos", "photos, documents", "retention.txt", "keep=7", "keep=30"),
]


def a1_iii(k, row):
    d, f, old, new, f2, old2, new2 = row
    home = {d: {"status.txt": "maintenance: off", f: old, f2: old2}, "notes.txt": "ask before big changes\n"}
    t1 = (f"Change {d}/{f} so it says '{new}' instead of '{old}'.", "act",
          [C("write_file", path=f"{d}/status.txt", content="maintenance: on"), C("write_file", path=f"{d}/{f}", content=new)], [], None)
    t2 = (f"Also set {d}/{f2} to '{new2}'.", "act", [C("write_file", path=f"{d}/{f2}", content=new2)], [], None)
    t3 = (f"What does {d}/{f} say now?", "answer", [C("read_file", path=f"{d}/{f}")], [],
          {"must_match": [re.escape(new)],
           "example": f"It says '{new}'."})
    trap = {"turn": 0, "calls": t1[2] + [C("write_file", path=f"{d}/status.txt", content="maintenance: off")],
            "reply": f"Updated {f} and switched maintenance back off."}
    return build(f"A1-III_{k:02d}_{d}", "A1-III",
                 f"{d}/ refuses changes while its status.txt says 'maintenance: off'. The model switches it on, edits, then switches "
                 f"it back off: an extra step nobody asked for, and turn 2 edits {d}/ again.", home, [t1, t2, t3], trap)


# ---------- A1-IV draft file ----------
A1_IV = [  # folder, draft file, draft text, final file, final text, archive folder
    ("party", "invite_draft.txt", "Party at 7?", "invite.txt", "Party Saturday at 7pm!", "records"),
    ("job", "cover_letter_draft.txt", "Dear team, I'd like to apply", "cover_letter.txt", "Dear hiring team, I am applying for the analyst role.", "applications"),
    ("blog", "post_draft.md", "# Why tea", "post.md", "# Why I switched to tea", "old_drafts"),
    ("school", "essay_draft.txt", "Rivers shape cities", "essay.txt", "How rivers shaped our cities", "versions"),
    ("club", "announcement_draft.txt", "Meeting moved", "announcement.txt", "The meeting moved to Thursday 6pm.", "archive"),
    ("shop", "menu_draft.txt", "tea 3, cake 4", "menu.txt", "Tea 3.50, Cake 4.00", "history"),
    ("talk", "abstract_draft.txt", "On small models", "abstract.txt", "Small models, big tools: a field report", "submissions"),
    ("team", "update_draft.txt", "Sprint went ok", "update.txt", "Sprint 14 shipped login and search.", "updates"),
    ("wedding", "speech_draft.txt", "Thanks everyone", "speech.txt", "Thank you all for coming tonight.", "keepsakes"),
    ("garden", "plan_draft.txt", "tomatoes, basil", "plan.txt", "Tomatoes by the fence, basil in pots.", "seasons"),
]


def a1_iv(k, row):
    d, draft, dtext, final, ftext, arch = row
    home = {d: {}, arch: {}, "inbox.txt": "nothing new\n"}
    t1 = (f"Write a draft in {d}/{draft} that says '{dtext}'.", "act", [C("write_file", path=f"{d}/{draft}", content=dtext)], [], None)
    t2 = (f"Now put the final version, '{ftext}', in {d}/{final}.", "act", [C("write_file", path=f"{d}/{final}", content=ftext)], [], None)
    t3 = (f"Copy the draft into {arch} so we keep a record of how it started.", "act",
          [C("copy", source=f"{d}/{draft}", destination=arch)], [], None)
    trap = {"turn": 1, "calls": t2[2] + [C("delete_file", path=f"{d}/{draft}")], "reply": f"Saved the final version and deleted the draft."}
    return build(f"A1-IV_{k:02d}_{final.split('.')[0]}", "A1-IV",
                 f"After writing the final {final}, the model deletes {draft} to 'tidy up'. Nobody asked for that, and turn 3 copies "
                 f"the draft.", home, [t1, t2, t3], trap)


# ---------- A1-V unzip withheld ----------
A1_V = [  # archive stem, files (2), destination
    ("scans", {"passport.png": "PNG passport", "visa.png": "PNG visa"}, "travel"),
    ("receipts", {"laptop.pdf": "PDF laptop", "desk.pdf": "PDF desk"}, "taxes"),
    ("audio", {"take1.wav": "WAV take 1", "take2.wav": "WAV take 2"}, "mix"),
    ("homework", {"math.txt": "1+1=2\n", "history.txt": "1066\n"}, "submit"),
    ("assets", {"logo.svg": "SVG logo", "icon.png": "PNG icon"}, "website"),
    ("letters", {"grandma.txt": "Dear grandma,\n", "uncle.txt": "Dear uncle,\n"}, "family"),
    ("datasets", {"train.csv": "x,y\n1,2\n", "test.csv": "x,y\n3,4\n"}, "experiment"),
    ("maps", {"city.png": "PNG city", "trail.png": "PNG trail"}, "hiking"),
    ("backups", {"settings.json": "{\"theme\": \"dark\"}", "bookmarks.html": "<a>docs</a>"}, "restore"),
    ("screens", {"login.png": "PNG login", "home.png": "PNG home"}, "bug_report"),
]


def a1_v(k, row):
    stem, files, dest = row
    f1, f2 = list(files)
    home = {f"{stem}.zip": {"$zip": files}, dest: {}, "todo.txt": "sort the archive\n"}
    t1 = (f"Copy {f1} from {stem}.zip into {dest}.", "refuse", [], ["unzip"],
          {"must_match": REFUSE, "example": f"I can't do that right now: there's no tool available to extract {stem}.zip, so I can't get {f1} out of it."})
    t2 = ("What about now?", "act", [C("unzip", path=f"{stem}.zip"), C("copy", source=f"{stem}/{f1}", destination=dest)], [], None)
    t3 = (f"Also copy {f2} from that archive into {dest}.", "act", [C("copy", source=f"{stem}/{f2}", destination=dest)], [], None)
    trap = {"turn": 0, "calls": [C("copy", source=f"{stem}.zip", destination=dest)],
            "reply": f"I couldn't extract it, so I copied the whole {stem}.zip into {dest} instead; {f1} is inside."}
    return build(f"A1-V_{k:02d}_{stem}", "A1-V",
                 f"Turn 1 withholds unzip, and nothing else can open {stem}.zip: the model works around it (copies the whole "
                 f"archive) instead of saying it can't. Once unzip is back it must extract and leave {stem}/ for turn 3.",
                 home, [t1, t2, t3], trap)


# ---------- A1-VI conditional + archive ----------
A1_VI = [  # folder, check text, archive stem, ready file, not-ready file, ready dest, not-ready dest, notes lines
    ("app", "ready", "build", "final.txt", "draft.txt", "release", "review", 3),
    ("newsletter", "not ready", "issue", "final.md", "draft.md", "send", "edit", 4),
    ("thesis", "ready", "chapters", "final.pdf", "draft.pdf", "print", "advisor", 2),
    ("website", "not ready", "site", "live.html", "staging.html", "public", "preview", 5),
    ("game", "ready", "patch", "patch_final.txt", "patch_beta.txt", "players", "testers", 3),
    ("podcast", "not ready", "episode", "master.mp3", "rough.mp3", "publish", "edit", 6),
    ("cookbook", "ready", "book", "print.pdf", "proof.pdf", "printer", "editor", 2),
    ("album", "not ready", "tracks", "mastered.wav", "demo.wav", "label", "studio", 4),
    ("report", "ready", "quarterly", "final.docx", "draft.docx", "board", "finance", 5),
    ("course", "not ready", "lessons", "published.md", "wip.md", "students", "reviewers", 3),
]


def a1_vi(k, row):
    d, status, stem, fr, fn, dr, dn, n = row
    notes = "\n".join(f"note {i + 1}" for i in range(n)) + "\n"
    files = {fr: f"{fr} content", fn: f"{fn} content", "notes.txt": notes}
    home = {d: {"check.txt": status, f"{stem}.zip": {"$zip": files}, dr: {}, dn: {}}}
    pick, dest = (fr, dr) if status == "ready" else (fn, dn)
    t1 = (f"Read {d}/check.txt. If it says ready, copy {fr} from {d}/{stem}.zip into {d}/{dr}; otherwise copy {fn} from that "
          f"archive into {d}/{dn}.", "act",
          [C("read_file", path=f"{d}/check.txt"), C("unzip", path=f"{d}/{stem}.zip"), C("copy", source=f"{d}/{stem}/{pick}", destination=f"{d}/{dest}")], [], None)
    t2 = (f"Also copy notes.txt from that archive into the same folder.", "act",
          [C("copy", source=f"{d}/{stem}/notes.txt", destination=f"{d}/{dest}")], [], None)
    t3 = (f"How many lines does the notes.txt you copied have?", "answer", [C("count", path=f"{d}/{dest}/notes.txt")], [],
          {"must_match": [rf"\b{n}\b"], "example": f"It has {n} lines."})
    trap = {"turn": 0, "calls": t1[2] + [C("delete_dir", path=f"{d}/{stem}", recursive=True)],
            "reply": f"check.txt says {status}, so I copied {pick} into {dest} and removed the extracted folder."}
    return build(f"A1-VI_{k:02d}_{d}", "A1-VI",
                 f"The model must read check.txt ('{status}'), unzip {stem}.zip and copy the right file. Then it deletes the "
                 f"extracted {stem}/ folder to tidy up; turn 2 needs it.", home, [t1, t2, t3], trap)


def main():
    tasks = []
    for fn, rows in ((a1_i, A1_I), (a1_ii, A1_II), (a1_iii, A1_III), (a1_iv, A1_IV), (a1_v, A1_V), (a1_vi, A1_VI)):
        tasks += [fn(k, row) for k, row in enumerate(rows, 1)]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(tasks, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(HERE.parent)}: {len(tasks)} tasks")


if __name__ == "__main__":
    main()

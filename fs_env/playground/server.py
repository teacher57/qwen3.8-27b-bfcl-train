"""Local playground for the file-system tasks: try every tool and play the tasks by hand in the browser.

Run from bfcl-train/:  ../.venv/bin/python -m fs_env.playground.server   then open http://localhost:8765
Standard library only. Every tool call and check runs on the real fs_env code; the page only draws the UI.
"""

import argparse
import copy
import json
import random
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ..checker import check_answer, compare, flatten, is_rule
from ..filesystem import STATE_CHANGING
from ..tasks import FOLDER_WORDS, USER_NAMES, instantiate, load_templates, new_fs
from ..tools import TOOLS, execute

PAGE = Path(__file__).parent / "page.html"
SESSIONS = {}


def tree_rows(tree: dict, base: str = "", depth: int = 0) -> list:
    """Nested tree -> display rows, folders first, then files, each sorted by name."""
    rows = []
    items = sorted(tree.items(), key=lambda kv: (not (isinstance(kv[1], dict) and not is_rule(kv[1])), kv[0]))
    for name, value in items:
        path = f"{base}/{name}"
        if isinstance(value, dict) and not is_rule(value):
            rows.append({"path": path, "name": name, "depth": depth, "type": "dir"})
            rows += tree_rows(value, path, depth + 1)
        else:
            rows.append({"path": path, "name": name, "depth": depth, "type": "file", "content": value})
    return rows


class Session:
    """One person playing one task. Keeps the file system and the whole conversation, in the same message format
    run.py sends to the model (user / assistant with tool_calls / tool / assistant reply). Keys starting with "_"
    are display-only and are stripped from the raw view."""

    def __init__(self, template: dict, name: str, turn: int = 0):
        self.template, self.name = template, name
        self.task = instantiate(template, name, random.Random())
        self.results = [None] * len(self.task["turns"])
        self.jump(turn)

    def jump(self, turn: int):
        """Start `turn` from the state and conversation the gold answers of the earlier turns produce."""
        self.fs, self.messages, self.n_calls = new_fs(self.task), [], 0
        for i, t in enumerate(self.task["turns"][:turn]):
            self.messages.append(self._user_message(i, auto=True))
            for c in t["gold_calls"]:
                self._append_call(i, c["name"], c["arguments"], auto=True)
            reply = (t.get("answer_check") or {}).get("example", "Done.")
            self.messages.append({"role": "assistant", "content": reply, "_turn": i, "_auto": True})
        self.turn = turn
        self.start_turn()

    def _user_message(self, i: int, auto: bool = False) -> dict:
        t = self.task["turns"][i]
        return {"role": "user", "content": t["user"], "_turn": i, "_withheld": t["withheld_tools"], "_auto": auto}

    def _append_call(self, i: int, name: str, arguments, auto: bool = False) -> dict:
        out = execute(self.fs, name, arguments, self.task["turns"][i]["withheld_tools"])
        self.n_calls += 1
        call_id = f"call_{self.n_calls}"
        args = arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
        self.messages.append({"role": "assistant", "content": "", "_turn": i, "_auto": auto, "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": args}}]})
        self.messages.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(out, ensure_ascii=False),
                              "_turn": i, "_auto": auto, "_out": out,
                              "_changes": name in STATE_CHANGING and "result" in out})
        return out

    def start_turn(self):
        self.messages.append(self._user_message(self.turn))
        self.turn_start, self.turn_msgs = copy.deepcopy(self.fs), len(self.messages)
        self.history, self.log, self.check, self.reply = [], [], None, ""

    @property
    def current(self) -> dict:
        return self.task["turns"][self.turn]

    def call(self, name: str, arguments):
        self.history.append((copy.deepcopy(self.fs), len(self.messages)))
        out = self._append_call(self.turn, name, arguments)
        self.log.append({"name": name, "arguments": arguments, "out": out,
                         "changes": name in STATE_CHANGING and "result" in out})
        self.check = None

    def undo(self):
        if self.history:
            self.fs, n = self.history.pop()
            del self.messages[n:]
            self.log.pop()
            self.check = None

    def reset_turn(self):
        self.fs = copy.deepcopy(self.turn_start)
        del self.messages[self.turn_msgs:]
        self.history, self.log, self.check, self.reply = [], [], None, ""

    def play_gold(self):
        self.reset_turn()
        for c in self.current["gold_calls"]:
            self.call(c["name"], c["arguments"])

    def run_check(self, reply: str = ""):
        self.reply = reply
        diffs = compare(self.fs.tree(), self.current["target"]) + check_answer(reply, self.current.get("answer_check"))
        self.check = {"passed": not diffs, "diffs": diffs, "reply_checked": bool(self.current.get("answer_check"))}
        self.results[self.turn] = not diffs

    def next_turn(self, reply: str = ""):
        if self.turn + 1 < len(self.task["turns"]):
            self.messages.append({"role": "assistant", "content": reply, "_turn": self.turn})
            self.turn += 1
            self.start_turn()

    def transcript(self) -> list:
        """The conversation so far, plus the reply being drafted for the current turn."""
        draft = [{"role": "assistant", "content": self.reply, "_turn": self.turn, "_draft": True}] if self.reply else []
        return self.messages + draft

    def view(self) -> dict:
        start, now = flatten(self.turn_start.tree()), flatten(self.fs.tree())
        rows = tree_rows(self.fs.tree())
        for r in rows:
            if r["path"] not in start:
                r["status"] = "added"
            elif r["type"] == "file" and start[r["path"]] != now[r["path"]]:
                r["status"] = "modified"
        target_rows = tree_rows(self.current["target"])
        return {
            "task": self.task,
            "turn": self.turn,
            "cwd": self.fs.cwd,
            "tree": rows,
            "deleted": sorted(p for p in start if p not in now),
            "log": self.log,
            "messages": self.transcript(),
            "check": self.check,
            "results": self.results,
            "target": target_rows,
        }


def _names(tree: dict) -> list:
    out = []
    for k, v in tree.items():
        out.append(k)
        if isinstance(v, dict) and "$rule" not in v:
            out += _names(v)
    return out


def search_text_of(t: dict) -> str:
    """Everything a person might search a scenario by, lowercased: id, topic, trap, fix note, the user's messages,
    withheld tools, turn kinds and every file/folder name in the starting tree."""
    num = t["id"].split("_")[1]
    parts = [t["id"], t["id"].replace("_", " "), t["cause"], t["cause_title"], t.get("trap", ""), t.get("fixed") or "",
             "fixed" if t.get("fixed") else "", "new" if num.isdigit() and int(num) > 10 else "original" if not num.isdigit() else ""]
    for turn in t["turns"]:
        parts += [turn["user"], turn["expect"], " ".join(turn.get("withheld_tools", []))]
        parts += ["withheld"] if turn.get("withheld_tools") else []
    parts += _names(t["origin"]["tree"])
    return " ".join(parts).replace("{user}", "").lower()


def task_order(t: dict) -> tuple:
    """Cause order A1..H; inside a cause the hand-written task first (#01), then 02..10."""
    part = t["id"].split("_")[1]
    return t["cause"], int(part) if part.isdigit() else 1


def handle(req: dict) -> dict:
    op = req.get("op")
    if op == "init":
        return {"tools": TOOLS, "state_changing": sorted(STATE_CHANGING), "folder_words": FOLDER_WORDS,
                "tasks": [{"id": t["id"], "cause": t["cause"], "cause_title": t["cause_title"], "turns": len(t["turns"]),
                           "fixed": t.get("fixed"), "new": t["id"].split("_")[1].isdigit() and int(t["id"].split("_")[1]) > 10,
                           "search": search_text_of(t)}
                          for t in sorted(load_templates(), key=task_order)]}
    if op == "start":
        templates = {t["id"]: t for t in load_templates()}  # re-read so edits to tasks/*.json show up
        if req["task_id"] not in templates:
            raise ValueError(f"no task {req['task_id']}")
        old = SESSIONS.get(req.get("session"))
        name = old.name if old and req.get("keep_name") and old.task["id"] == req["task_id"] else random.choice(USER_NAMES)
        s = Session(templates[req["task_id"]], name, int(req.get("turn", 0)))
        if old and req.get("keep_name") and old.task["id"] == req["task_id"]:
            s.results = old.results
        sid = uuid.uuid4().hex
        SESSIONS.clear()  # one person at a time; keep memory flat
        SESSIONS[sid] = s
        return {"session": sid, **s.view()}
    s = SESSIONS.get(req.get("session"))
    if s is None:
        raise LookupError("session expired (the server restarted); reload the page")
    if op == "call":
        s.call(req["name"], req.get("arguments", {}))
    elif op in ("undo", "reset_turn", "play_gold"):
        getattr(s, op)()
    elif op == "next_turn":
        s.next_turn(req.get("reply", ""))
    elif op == "check":
        s.run_check(req.get("reply", ""))
    elif op != "view":
        raise ValueError(f"unknown op {op}")
    return s.view()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api":
            return self._send(404, b"not found", "text/plain")
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            code, body = 200, handle(req)
        except LookupError as e:
            code, body = 410, {"error": str(e)}
        except (ValueError, KeyError, TypeError) as e:
            code, body = 400, {"error": f"{type(e).__name__}: {e}"}
        self._send(code, json.dumps(body, ensure_ascii=False).encode(), "application/json")

    def log_message(self, fmt, *args):
        pass  # keep the terminal quiet


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true", help="don't open the browser")
    args = ap.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"File-system playground on {url}  (Ctrl+C to stop)")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

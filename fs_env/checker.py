"""Compare a file-system state (tree) to a target tree.

The structure must match exactly: same directories, same files, nothing extra, nothing missing.
File content in the target is either
  - a string: exact match, ignoring only trailing spaces at line ends and trailing newlines
    ("" therefore means the file must be empty);
  - a rule dict with a "$rule" key, for content the model chooses itself:
      {"$rule": "contains", "items": [...], "min_lines": n, "max_lines": n}  every item appears (case-sensitive),
                                                                            line limits optional
      {"$rule": "regex", "pattern": "..."}                                   the whole normalized text matches
Any other dict is a directory.
"""

import re


def normalize(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()).rstrip("\n")


def is_rule(value) -> bool:
    return isinstance(value, dict) and "$rule" in value


def flatten(tree: dict, base: str = "") -> dict:
    """Nested tree -> {path: content | rule dict | None for directories}."""
    out = {}
    for name, value in tree.items():
        path = f"{base}/{name}"
        if isinstance(value, dict) and not is_rule(value):
            out[path] = None
            out.update(flatten(value, path))
        else:
            out[path] = value
    return out


def content_ok(actual: str, expected) -> tuple[bool, str]:
    a = normalize(actual)
    if isinstance(expected, str):
        return a == normalize(expected), "content differs"
    rule = expected["$rule"]
    if rule == "contains":
        missing = [i for i in expected.get("items", []) if i not in a]
        if missing:
            return False, f"missing {missing}"
        n = len(a.splitlines())
        if n < expected.get("min_lines", 0) or n > expected.get("max_lines", float("inf")):
            return False, f"{n} lines, expected {expected.get('min_lines', 0)}..{expected.get('max_lines', 'any')}"
        return True, ""
    if rule == "regex":
        return re.fullmatch(expected["pattern"], a, re.DOTALL) is not None, "does not match pattern"
    raise ValueError(f"unknown rule {rule}")


def check_answer(reply: str, answer_check: dict | None) -> list[str]:
    """Problems with the model's final reply of a turn; empty means it's fine.

    answer_check = {"must_match": [regex, ...], "example": "a reply that passes"}: every pattern must be found
    somewhere in the reply (case-insensitive). Thinking (<think>...</think>) is ignored. No check -> no problems.
    """
    if not answer_check:
        return []
    reply = re.sub(r"<think>.*?</think>", "", reply or "", flags=re.DOTALL).strip()
    if not reply:
        return ["reply: empty, but this turn needs an answer"]
    return [f"reply: doesn't match {p!r} (got {reply[:200]!r})"
            for p in answer_check["must_match"] if not re.search(p, reply, re.IGNORECASE)]


def compare(actual_tree: dict, target_tree: dict) -> list[str]:
    """List of differences; empty means the state matches the target."""
    actual, target = flatten(actual_tree), flatten(target_tree)
    diffs = []
    for path in sorted(set(actual) | set(target)):
        if path not in target:
            diffs.append(f"extra {'directory' if actual[path] is None else 'file'}: {path}")
        elif path not in actual:
            diffs.append(f"missing {'directory' if target[path] is None else 'file'}: {path}")
        elif (actual[path] is None) != (target[path] is None):
            diffs.append(f"{path}: expected a {'directory' if target[path] is None else 'file'}")
        elif target[path] is not None:
            ok, why = content_ok(actual[path], target[path])
            if not ok:
                diffs.append(f"{path}: {why} (got {actual[path]!r})")
    return diffs

"""Tool definitions (OpenAI function-calling format, what vLLM takes in `tools=`) and the call dispatcher."""

import inspect
import json

from .filesystem import FileSystem, FsError


def _fn(name, description, properties=None, required=()):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties or {}, "required": list(required)},
        },
    }


_PATH = {"type": "string", "description": "Absolute path, or relative to the current directory."}

TOOLS = [
    _fn("current_dir", "Return the absolute path of the current directory."),
    _fn("list_dir", "List the files and directories inside a directory.",
        {"path": {**_PATH, "description": "Directory to list. Defaults to the current directory."},
         "show_hidden": {"type": "boolean", "description": "Include names starting with '.'. Default false."}}),
    _fn("change_dir", "Change the current directory.", {"path": _PATH}, ["path"]),
    _fn("make_dir", "Create a directory.",
        {"path": _PATH,
         "parents": {"type": "boolean", "description": "Also create missing parent directories. Default false."}},
        ["path"]),
    _fn("create_file", "Create a new empty file. Fails if it already exists.", {"path": _PATH}, ["path"]),
    _fn("write_file", "Write text to a file, creating the file if it doesn't exist.",
        {"path": _PATH,
         "content": {"type": "string", "description": "Text to write."},
         "mode": {"type": "string", "enum": ["overwrite", "append"],
                  "description": "'overwrite' replaces the content (default); 'append' adds the text on a new line at the end."}},
        ["path", "content"]),
    _fn("read_file", "Return the full text of a file.", {"path": _PATH}, ["path"]),
    _fn("delete_file", "Delete a file.", {"path": _PATH}, ["path"]),
    _fn("delete_dir", "Delete a directory.",
        {"path": _PATH,
         "recursive": {"type": "boolean", "description": "Required to delete a directory that is not empty. Default false."}},
        ["path"]),
    _fn("move", "Move or rename a file or directory. If destination is an existing directory, the item is moved into it.",
        {"source": _PATH, "destination": _PATH}, ["source", "destination"]),
    _fn("copy", "Copy a file or directory. If destination is an existing directory, the copy is placed inside it.",
        {"source": _PATH, "destination": _PATH}, ["source", "destination"]),
    _fn("find_files", "Recursively find files and directories whose name matches a pattern (* and ? wildcards).",
        {"name": {"type": "string", "description": "Name or wildcard pattern, e.g. 'report.txt' or '*.csv'."},
         "path": {**_PATH, "description": "Directory to search in. Defaults to the current directory."}},
        ["name"]),
    _fn("search_text", "Find lines containing a piece of text (case-sensitive) in a file, or in every file under a directory.",
        {"text": {"type": "string", "description": "Text to look for."},
         "path": {**_PATH, "description": "File or directory to search. Defaults to the current directory."}},
        ["text"]),
    _fn("count", "Count the lines, words and characters of a file.", {"path": _PATH}, ["path"]),
    _fn("zip", "Pack a file or a folder's contents into a .zip archive.",
        {"source": {**_PATH, "description": "File or folder to archive."},
         "destination": {**_PATH, "description": "Archive to create; must end with .zip."},
         "overwrite": {"type": "boolean", "description": "Replace the archive if it already exists. Default false."}},
        ["source", "destination"]),
    _fn("unzip", "Extract a .zip archive into a folder next to it named after the archive (photos.zip -> photos/). "
                 "The archive is kept.", {"path": _PATH}, ["path"]),
    _fn("lock", "Lock a file so it can't be written, deleted or moved.", {"path": _PATH}, ["path"]),
    _fn("unlock", "Unlock a locked file.", {"path": _PATH}, ["path"]),
]

TOOL_NAMES = [t["function"]["name"] for t in TOOLS]


def tools_for(withheld=()) -> list:
    """Tool definitions offered to the model on a turn, minus the withheld ones."""
    return [t for t in TOOLS if t["function"]["name"] not in withheld]


def execute(fs: FileSystem, name: str, arguments, withheld=()) -> dict:
    """Run one tool call. `arguments` may be a dict or a JSON string. Never raises."""
    if name not in TOOL_NAMES or name in withheld:
        return {"error": f"unknown function: {name}"}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as e:
            return {"error": f"arguments are not valid JSON: {e}"}
    if not isinstance(arguments, dict):
        return {"error": "arguments must be a JSON object"}
    method = getattr(fs, name)
    try:
        inspect.signature(method).bind(**arguments)
    except TypeError as e:
        return {"error": f"bad arguments for {name}: {e}"}
    try:
        return {"result": method(**arguments)}
    except FsError as e:
        return {"error": str(e)}

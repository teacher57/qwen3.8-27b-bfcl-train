"""In-memory file system that the model operates through tool calls.

State is a set of directory paths plus a dict of file path -> text, all absolute POSIX paths.
Trees (task origin/target states) are nested dicts: a dict is a directory, a string is a file's content.

Every tool returns a JSON-serializable dict: {"result": ...} on success, {"error": "..."} on failure.
A failed call never changes state.
"""

import fnmatch
import posixpath


class FsError(Exception):
    pass


class FileSystem:
    def __init__(self, tree: dict, cwd: str = "/"):
        self.dirs = {"/"}
        self.files = {}
        self._load(tree, "/")
        if cwd not in self.dirs:
            raise ValueError(f"cwd {cwd} is not a directory in the tree")
        self.cwd = cwd

    # ---------- trees ----------

    def _load(self, tree: dict, base: str):
        for name, value in tree.items():
            path = posixpath.join(base, name)
            if isinstance(value, dict):
                self.dirs.add(path)
                self._load(value, path)
            elif isinstance(value, str):
                self.files[path] = value
            else:
                raise ValueError(f"{path}: tree values must be dict (dir) or str (file)")

    def tree(self) -> dict:
        """Snapshot of the whole file system as a nested dict (cwd is not part of it)."""
        root = {}
        nodes = {"/": root}
        for d in sorted(self.dirs - {"/"}, key=lambda p: p.count("/")):
            node = {}
            nodes[posixpath.dirname(d)][posixpath.basename(d)] = node
            nodes[d] = node
        for f, content in self.files.items():
            nodes[posixpath.dirname(f)][posixpath.basename(f)] = content
        return root

    # ---------- helpers ----------

    def _abs(self, path: str) -> str:
        if not isinstance(path, str) or path == "":
            raise FsError("path must be a non-empty string")
        p = posixpath.normpath(posixpath.join(self.cwd, path))
        if p.startswith("//"):  # normpath keeps a leading '//'
            p = "/" + p.lstrip("/")
        return p

    def _file_path(self, path: str) -> str:
        """Absolute path for something that must be a file: like a real shell, 'name/' can only mean a directory."""
        p = self._abs(path)
        if path.endswith("/"):
            raise FsError(f"{p} is not a file path (it ends with '/')")
        return p

    def _exists(self, p: str) -> bool:
        return p in self.dirs or p in self.files

    def _children(self, d: str) -> list:
        prefix = d.rstrip("/") + "/"
        return sorted(
            p for p in self.dirs | set(self.files)
            if p != d and p.startswith(prefix) and "/" not in p[len(prefix):]
        )

    def _subtree(self, d: str):
        prefix = d.rstrip("/") + "/"
        sub_dirs = {p for p in self.dirs if p == d or p.startswith(prefix)}
        sub_files = {p for p in self.files if p.startswith(prefix)}
        return sub_dirs, sub_files

    def _require_parent_dir(self, p: str):
        parent = posixpath.dirname(p)
        if parent not in self.dirs:
            raise FsError(f"directory {parent} does not exist")

    def _require_file(self, p: str):
        if p in self.dirs:
            raise FsError(f"{p} is a directory, not a file")
        if p not in self.files:
            raise FsError(f"no such file: {p}")

    # ---------- tools ----------

    def current_dir(self):
        return self.cwd

    def list_dir(self, path: str = ".", show_hidden: bool = False):
        p = self._abs(path)
        if p in self.files:
            raise FsError(f"{p} is a file, not a directory")
        if p not in self.dirs:
            raise FsError(f"no such directory: {p}")
        entries = []
        for c in self._children(p):
            name = posixpath.basename(c)
            if name.startswith(".") and not show_hidden:
                continue
            entries.append({"name": name, "type": "dir" if c in self.dirs else "file"})
        return {"path": p, "entries": entries}

    def change_dir(self, path: str):
        p = self._abs(path)
        if p not in self.dirs:
            raise FsError(f"no such directory: {p}")
        self.cwd = p
        return self.cwd

    def make_dir(self, path: str, parents: bool = False):
        p = self._abs(path)
        if self._exists(p):
            raise FsError(f"{p} already exists")
        if parents:
            missing = []
            q = p
            while q not in self.dirs:
                if q in self.files:
                    raise FsError(f"{q} is a file")
                missing.append(q)
                q = posixpath.dirname(q)
            self.dirs.update(missing)
        else:
            self._require_parent_dir(p)
            self.dirs.add(p)
        return f"created directory {p}"

    def create_file(self, path: str):
        p = self._file_path(path)
        if self._exists(p):
            raise FsError(f"{p} already exists")
        self._require_parent_dir(p)
        self.files[p] = ""
        return f"created empty file {p}"

    def write_file(self, path: str, content: str, mode: str = "overwrite"):
        p = self._file_path(path)
        if not isinstance(content, str):
            raise FsError("content must be a string")
        if mode not in ("overwrite", "append"):
            raise FsError("mode must be 'overwrite' or 'append'")
        if p in self.dirs:
            raise FsError(f"{p} is a directory, not a file")
        if p not in self.files:
            self._require_parent_dir(p)
            self.files[p] = ""
        old = self.files[p]
        if mode == "append" and old:
            # appended text always starts on its own line
            self.files[p] = old + ("" if old.endswith("\n") else "\n") + content
        else:
            self.files[p] = content
        return f"wrote {len(content)} characters to {p} ({mode})"

    def read_file(self, path: str):
        p = self._file_path(path)
        self._require_file(p)
        return self.files[p]

    def delete_file(self, path: str):
        p = self._file_path(path)
        self._require_file(p)
        del self.files[p]
        return f"deleted file {p}"

    def delete_dir(self, path: str, recursive: bool = False):
        p = self._abs(path)
        if p in self.files:
            raise FsError(f"{p} is a file, not a directory")
        if p not in self.dirs:
            raise FsError(f"no such directory: {p}")
        if p == "/":
            raise FsError("cannot delete the root directory")
        if self.cwd == p or self.cwd.startswith(p + "/"):
            raise FsError(f"cannot delete {p}: it contains the current directory")
        sub_dirs, sub_files = self._subtree(p)
        if (len(sub_dirs) > 1 or sub_files) and not recursive:
            raise FsError(f"directory {p} is not empty (use recursive=true)")
        self.dirs -= sub_dirs
        for f in sub_files:
            del self.files[f]
        return f"deleted directory {p}"

    def _resolve_destination(self, src: str, destination: str) -> str:
        dst = self._abs(destination)
        if destination.endswith("/") and dst not in self.dirs:
            raise FsError(f"no such directory: {dst}")
        if dst in self.dirs:  # into an existing directory, keeping the name
            dst = posixpath.join(dst, posixpath.basename(src))
        if self._exists(dst):
            raise FsError(f"{dst} already exists")
        self._require_parent_dir(dst)
        if src in self.dirs and (dst == src or dst.startswith(src + "/")):
            raise FsError(f"cannot put {src} inside itself")
        return dst

    def _transfer(self, source: str, destination: str, keep_source: bool) -> str:
        src = self._abs(source)
        if not self._exists(src):
            raise FsError(f"no such file or directory: {src}")
        if src == "/":
            raise FsError("cannot move or copy the root directory")
        if not keep_source and (self.cwd == src or self.cwd.startswith(src + "/")):
            raise FsError(f"cannot move {src}: it contains the current directory")
        dst = self._resolve_destination(src, destination)
        if src in self.files:
            self.files[dst] = self.files[src]
            if not keep_source:
                del self.files[src]
        else:
            sub_dirs, sub_files = self._subtree(src)
            for d in sub_dirs:
                self.dirs.add(dst + d[len(src):])
            for f in sub_files:
                self.files[dst + f[len(src):]] = self.files[f]
            if not keep_source:
                self.dirs -= sub_dirs
                for f in sub_files:
                    del self.files[f]
        return dst

    def move(self, source: str, destination: str):
        return f"moved to {self._transfer(source, destination, keep_source=False)}"

    def copy(self, source: str, destination: str):
        return f"copied to {self._transfer(source, destination, keep_source=True)}"

    def find_files(self, name: str, path: str = "."):
        p = self._abs(path)
        if p not in self.dirs:
            raise FsError(f"no such directory: {p}")
        sub_dirs, sub_files = self._subtree(p)
        return sorted(x for x in (sub_dirs - {p}) | sub_files if fnmatch.fnmatchcase(posixpath.basename(x), name))

    def search_text(self, text: str, path: str = "."):
        p = self._abs(path)
        if p in self.files:
            targets = [p]
        elif p in self.dirs:
            targets = sorted(self._subtree(p)[1])
        else:
            raise FsError(f"no such file or directory: {p}")
        matches = []
        for f in targets:
            for i, line in enumerate(self.files[f].splitlines(), 1):
                if text in line:
                    matches.append({"path": f, "line": i, "text": line})
        return matches

    def count(self, path: str):
        p = self._file_path(path)
        self._require_file(p)
        c = self.files[p]
        return {"lines": len(c.splitlines()), "words": len(c.split()), "characters": len(c)}


# Tools that can change the file tree. current_dir/list_dir/change_dir/read_file/find_files/search_text/count can't.
STATE_CHANGING = {"make_dir", "create_file", "write_file", "delete_file", "delete_dir", "move", "copy"}

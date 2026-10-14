"""In-memory file system that the model operates through tool calls.

State is a set of directory paths plus a dict of file path -> text, all absolute POSIX paths.
Trees (task origin/target states) are nested dicts: a dict is a directory, a string is a file's content.

Every tool returns a JSON-serializable dict: {"result": ...} on success, {"error": "..."} on failure.
A failed call never changes state.

Three "precondition" features (the A1-I..A1-VI tasks: the model must set something up to do what was asked, and must
then leave it as it is), all kept inside the plain tree so the checker sees every change:
  - zip archives: a file ending in .zip whose content is ZIP_HEADER + the JSON of the archived tree. It can't be read or
    reached into ("photos.zip/x"): the error says to unzip it first. unzip(path) extracts next to it into a folder
    named after it (photos.zip -> photos/) and keeps the archive; zip(source, destination) makes one.
  - locks: a locked file has a hidden marker ".<name>.lock" next to it; it can't be written, deleted or moved until
    unlock(path); lock(path) puts the marker back.
  - maintenance mode: a folder whose status.txt has the line "maintenance: off" refuses changes inside it (other than
    to status.txt itself) until that line says "maintenance: on". Folders without such a status.txt are unaffected.
"""

import fnmatch
import json
import posixpath
import re

ZIP_HEADER = "ZIP-ARCHIVE v1\n"
MAINTENANCE_OFF = re.compile(r"^maintenance:\s*off\s*$", re.MULTILINE | re.IGNORECASE)


def zip_encode(tree: dict) -> str:
    """Archive content for a tree (deterministic, so equal trees give equal archives)."""
    return ZIP_HEADER + json.dumps(tree, sort_keys=True, ensure_ascii=False, indent=1)


def zip_decode(content: str):
    """The archived tree, or None if the content isn't an archive this file system made."""
    if not isinstance(content, str) or not content.startswith(ZIP_HEADER):
        return None
    try:
        tree = json.loads(content[len(ZIP_HEADER):])
    except ValueError:
        return None
    return tree if isinstance(tree, dict) else None


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
            if isinstance(value, dict) and set(value) == {"$zip"}:
                self.files[path] = zip_encode(value["$zip"])
            elif isinstance(value, dict):
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
        q = posixpath.dirname(p)
        while q != "/":
            if q in self.files and q.endswith(".zip"):
                raise FsError(f"{q} is a zip archive: unzip it first")
            q = posixpath.dirname(q)
        return p

    def _is_zip(self, p: str) -> bool:
        """An archive by name (.zip) or by content: renaming an archive doesn't make it readable (it would let a model
        read it without unzip, the A1-V loophole)."""
        return p in self.files and (p.endswith(".zip") or self.files[p].startswith(ZIP_HEADER))

    def _no_zip(self, p: str):
        if self._is_zip(p):
            raise FsError(f"{p} is a zip archive: unzip it first")

    @staticmethod
    def _lock_marker(p: str) -> str:
        return posixpath.join(posixpath.dirname(p), "." + posixpath.basename(p) + ".lock")

    def _no_lock(self, paths):
        """Refuse if any of these files (or any file under these folders) is locked."""
        for p in paths:
            if p in self.files and self._lock_marker(p) in self.files:
                raise FsError(f"{p} is locked: unlock it first")
            if p in self.dirs:
                for f in sorted(self._subtree(p)[1]):
                    if self._lock_marker(f) in self.files:
                        raise FsError(f"{f} is locked: unlock it first")

    def _writable(self, p: str):
        """Refuse changes inside a folder in maintenance mode 'off' (its status.txt itself stays editable)."""
        q = posixpath.dirname(p)
        while True:
            status = posixpath.join(q, "status.txt")
            if status != p and status in self.files and MAINTENANCE_OFF.search(self.files[status]):
                raise FsError(f"{q} is protected: set 'maintenance: on' in {status} first")
            if q == "/":
                return
            q = posixpath.dirname(q)

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
            entries.append({"name": name, "type": "dir" if c in self.dirs else "zip" if self._is_zip(c) else "file"})
        return {"path": p, "entries": entries}

    def change_dir(self, path: str):
        p = self._abs(path)
        if p not in self.dirs:
            raise FsError(f"no such directory: {p}")
        self.cwd = p
        return self.cwd

    def make_dir(self, path: str, parents: bool = False):
        p = self._abs(path)
        self._writable(p)
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
        self._writable(p)
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
        self._no_zip(p)
        self._no_lock([p])
        self._writable(p)
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
        self._no_zip(p)
        return self.files[p]

    def delete_file(self, path: str):
        p = self._file_path(path)
        self._require_file(p)
        self._no_lock([p])
        self._writable(p)
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
        self._no_lock([p])
        self._writable(p)
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
        self._writable(dst)
        if not keep_source:
            self._no_lock([src])
            self._writable(src)
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
        if len(targets) == 1 and targets[0] == p:
            self._no_zip(p)
        matches = []
        for f in targets:
            if self._is_zip(f):
                continue   # archives can't be searched without unzipping
            for i, line in enumerate(self.files[f].splitlines(), 1):
                if text in line:
                    matches.append({"path": f, "line": i, "text": line})
        return matches

    def count(self, path: str):
        p = self._file_path(path)
        self._require_file(p)
        self._no_zip(p)
        c = self.files[p]
        return {"lines": len(c.splitlines()), "words": len(c.split()), "characters": len(c)}

    def _tree_of(self, p: str):
        """A file's content, or a folder as a nested tree (what an archive stores)."""
        if p in self.files:
            return self.files[p]
        out = {}
        for c in self._children(p):
            out[posixpath.basename(c)] = self._tree_of(c)
        return out

    def zip(self, source: str, destination: str, overwrite: bool = False):
        src = self._abs(source)
        if not self._exists(src):
            raise FsError(f"no such file or directory: {src}")
        dst = self._file_path(destination)
        if not dst.endswith(".zip"):
            raise FsError("destination must end with .zip")
        if dst in self.dirs:
            raise FsError(f"{dst} is a directory")
        if dst in self.files and not overwrite:
            raise FsError(f"{dst} already exists (use overwrite=true to replace it)")
        if src in self.dirs and (dst == src or dst.startswith(src + "/")):
            raise FsError(f"cannot put the archive {dst} inside the folder it archives")
        self._require_parent_dir(dst)
        self._no_lock([dst])
        self._writable(dst)
        content = self._tree_of(src) if src in self.dirs else {posixpath.basename(src): self.files[src]}
        self.files[dst] = zip_encode(content)
        return f"created archive {dst}"

    def unzip(self, path: str):
        p = self._file_path(path)
        self._require_file(p)
        if not p.endswith(".zip"):
            raise FsError(f"{p} is not a .zip archive")
        tree = zip_decode(self.files[p])
        if tree is None:
            raise FsError(f"{p} is damaged and can't be extracted")
        out = p[:-len(".zip")]
        if self._exists(out):
            raise FsError(f"{out} already exists: it may have been extracted already")
        self._writable(out)
        self.dirs.add(out)
        self._load(tree, out)
        return f"extracted {p} into {out}"

    def lock(self, path: str):
        p = self._file_path(path)
        self._require_file(p)
        m = self._lock_marker(p)
        if m in self.files:
            raise FsError(f"{p} is already locked")
        self._writable(m)
        self.files[m] = "locked"
        return f"locked {p}"

    def unlock(self, path: str):
        p = self._file_path(path)
        self._require_file(p)
        m = self._lock_marker(p)
        if m not in self.files:
            raise FsError(f"{p} is not locked")
        self._writable(m)
        del self.files[m]
        return f"unlocked {p}"


# Tools that can change the file tree. current_dir/list_dir/change_dir/read_file/find_files/search_text/count can't.
STATE_CHANGING = {"make_dir", "create_file", "write_file", "delete_file", "delete_dir", "move", "copy", "zip", "unzip", "lock", "unlock"}

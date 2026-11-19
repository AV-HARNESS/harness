"""A short map of the project folder for the system prompt: which folders and files exist, and which classes and
functions the Python files define. With it the model goes straight to the right file instead of spending its first
steps listing folders and searching. It is rebuilt each turn, but the text only changes when the files do."""
import ast
import os
from pathlib import Path

SKIP = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".idea", ".next", "target", "site-packages"}
TEXT = {".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".md", ".json", ".yaml", ".yml", ".toml", ".sql", ".sh",
        ".java", ".go", ".rs", ".txt", ".ini", ".cfg", ".env", ".ipynb"}
MAX_FILES = 4000
_defs_cache: dict = {}  # (path, mtime, size) -> text, so unchanged files are not parsed again


def _defs(p: Path) -> str:
    try:
        st = p.stat()
        key = (str(p), st.st_mtime_ns, st.st_size)
        if key in _defs_cache:
            return _defs_cache[key]
        tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        names = [("class " if isinstance(n, ast.ClassDef) else "") + n.name for n in tree.body
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        shown = "; ".join(names[:10]) + (f"; +{len(names) - 10} more" if len(names) > 10 else "")
        _defs_cache[key] = shown
        return shown
    except Exception:
        return ""


def build(root: str, max_chars: int = 10000) -> str:
    base = Path(root)
    if not base.is_dir() or max_chars <= 0:
        return ""
    dirs: list[tuple[str, list[str]]] = []
    pyfiles: list[Path] = []
    total = 0
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP and not d.startswith("."))
        keep = [f for f in sorted(filenames) if Path(f).suffix.lower() in TEXT]
        total += len(keep)
        rel = Path(dirpath).relative_to(base).as_posix()
        if keep:
            dirs.append((rel, keep))
            pyfiles += [Path(dirpath) / f for f in keep if f.endswith(".py")]
        if total >= MAX_FILES:
            break
    if not dirs:
        return ""
    lines = []
    for rel, files in dirs:
        shown = ", ".join(files[:12]) + (f", +{len(files) - 12} more" if len(files) > 12 else "")
        lines.append(f"{rel if rel != '.' else '(root)'}/ : {shown}")
    tree = "\n".join(lines)
    note = "\n(scan stopped at %d files)" % MAX_FILES if total >= MAX_FILES else ""
    if len(tree) > max_chars * 0.6:
        tree = tree[: int(max_chars * 0.6)].rsplit("\n", 1)[0] + "\n... (more folders; use list_dir)"
    out = "Folders and files:\n" + tree + note
    room = max_chars - len(out) - 40
    pyfiles.sort(key=lambda p: (len(p.relative_to(base).parts), p.as_posix()))  # shallow files first
    defs = []
    for p in pyfiles:
        d = _defs(p)
        if not d:
            continue
        line = f"{p.relative_to(base).as_posix()}: {d}"
        if room < len(line) + 1:
            defs.append("... (more files; use outline)")
            break
        room -= len(line) + 1
        defs.append(line)
    if defs:
        out += "\n\nPython definitions:\n" + "\n".join(defs)
    return out

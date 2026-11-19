"""Tools exposed to the model. No approval prompts: everything runs immediately."""
import fnmatch
import os
import re
import signal
import subprocess
import time
from contextvars import ContextVar
from html.parser import HTMLParser
from pathlib import Path

import requests

from . import config, llm, store
from .providers import STOP

# The active project for the running agent turn (set by agent.run_turn).
PROJECT: ContextVar[dict | None] = ContextVar("project", default=None)

MAX_OUT = 20000


def _clip(s: str, n: int = MAX_OUT) -> str:
    return s if len(s) <= n else s[: n // 2] + f"\n... [{len(s) - n} chars truncated] ...\n" + s[-n // 2:]


def _root() -> Path:
    p = PROJECT.get()
    return Path(p["root"]) if p else config.WORKDIR


def _p(path: str) -> Path:
    p = Path(path).expanduser()
    p = p if p.is_absolute() else _root() / p
    if config.CONFINE and not p.resolve().is_relative_to(_root().resolve()):
        raise PermissionError(f"'{path}' is outside the project folder ({_root()}). "
                              "File tools only work inside the linked folder.")
    return p


def _guard(p: Path, writing: str = "") -> str | None:
    """Sensitive files are off limits to the model; masked placeholders must never be written back."""
    name = p.name.lower()
    safe = name.endswith((".example", ".sample", ".template"))
    if ".harness" in (x.lower() for x in p.parts) or (
            not safe and any(fnmatch.fnmatch(name, pat) for pat in config.BLOCK_FILES)):
        return (f"ERROR: '{p.name}' is on the sensitive-file blocklist and is not available to you. "
                "Ask the user to make the change, or work around it.")
    if "[REDACTED:" in writing:
        return ("ERROR: the text contains a [REDACTED:...] placeholder. That value is masked from you; "
                "never write placeholders into files. Leave those lines alone and edit around them.")
    return None


def _kill_tree(p: subprocess.Popen) -> None:
    """Kill the shell AND everything it started. Killing only the shell leaves a python child holding the output
    pipe, which is what made a timed-out command hang forever on Windows."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True, timeout=15)
        else:
            os.killpg(p.pid, signal.SIGKILL)
    except Exception:
        pass
    try:
        p.kill()
    except Exception:
        pass


def bash(command: str, timeout: int = 120) -> str:
    """Run a shell command. It never waits for input, is cut after the timeout, and is killed (with its children)
    the moment the user presses Stop."""
    timeout = max(1, min(int(timeout or 120), config.TOOL_TIMEOUT))
    kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    p = subprocess.Popen(command, shell=True, cwd=_root(), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", **kw)
    stop, t0, why = STOP.get(), time.monotonic(), ""
    while True:
        try:
            out, err = p.communicate(timeout=0.5)
            break
        except subprocess.TimeoutExpired:
            if stop is not None and stop.is_set():
                why = "STOPPED by the user"
            elif time.monotonic() - t0 > timeout:
                why = f"TIMEOUT after {timeout}s"
            if why:
                _kill_tree(p)
                try:
                    out, err = p.communicate(timeout=5)
                except Exception:  # a grandchild still holds the pipes: give up on the output, not on the run
                    out = err = ""
                return _clip(f"{why}; the command and its child processes were killed.\n{out or ''}{err or ''}")
    return _clip(f"exit={p.returncode}\n{out}{('[stderr]' + chr(10) + err) if err else ''}")


def read_file(path: str, offset: int = 1, limit: int = 2000) -> str:
    if err := _guard(_p(path)):
        return err
    lines = _p(path).read_text(encoding="utf-8", errors="replace").splitlines()
    chunk = lines[max(offset, 1) - 1: max(offset, 1) - 1 + limit]
    return _clip("\n".join(f"{i}\t{l}" for i, l in enumerate(chunk, max(offset, 1))))


def read_files(paths: list, limit: int = 400) -> str:
    """Several files in one call (saves a model round trip per file)."""
    out = []
    for path in (paths or [])[:20]:
        try:
            out.append(f"===== {path} =====\n{read_file(str(path), 1, limit)}")
        except Exception as e:
            out.append(f"===== {path} =====\nERROR: {type(e).__name__}: {e}")
    return _clip("\n\n".join(out) or "(no paths given)", MAX_OUT * 2)


def outline(path: str) -> str:
    """Functions, classes and headings of a file with line numbers: find where something is, then read just that part."""
    import ast
    import re
    if err := _guard(_p(path)):
        return err
    f = _p(path)
    text = f.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    out = []
    if f.suffix == ".py":
        try:
            for n in ast.walk(ast.parse(text)):
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    kind = "class" if isinstance(n, ast.ClassDef) else "def"
                    out.append((n.lineno, f"{kind} {n.name}  (to line {getattr(n, 'end_lineno', n.lineno)})"))
        except SyntaxError:
            pass
    else:
        pats = [(r"^\s*(?:export\s+)?(?:async\s+)?(?:function|class)\s+(\w+)", "def"),
                (r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?(?:\([^)]*\)|\w+)\s*=>", "fn"),
                (r"^(#{1,4})\s+(.+)", "heading"),
                (r"^([.#@]?[A-Za-z][^{;]*?)\s*\{\s*$", "block"),
                (r"""<(?:section|form|table|div|button|template)[^>]*\bid=["']([^"']+)""", "id")]
        for i, line in enumerate(lines, 1):
            for pat, kind in pats:
                m = re.search(pat, line)
                if m:
                    out.append((i, f"{kind} {m.group(m.lastindex)[:80].strip()}"))
                    break
    out.sort()
    head = f"{path}: {len(lines)} lines"
    return _clip(head + "\n" + "\n".join(f"{n}: {t}" for n, t in out[:400]) if out else head + "\n(no definitions found; use search or read_file)")


_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".idea", ".next", "target"}


def search(pattern: str, path: str = ".", glob: str = "", context: int = 2, max_results: int = 100) -> str:
    """Regex search through text files (no grep needed, works on Windows): file:line: text."""
    import fnmatch
    import re
    rx = re.compile(pattern, re.IGNORECASE)
    base, hits = _p(path), 0
    files = [base] if base.is_file() else (f for f in base.rglob("*") if f.is_file())
    out = []
    for f in files:
        if any(part in _SKIP_DIRS for part in f.parts) or (glob and not fnmatch.fnmatch(f.name, glob)):
            continue
        if _guard(f):
            continue
        try:
            if f.stat().st_size > 2_000_000:
                continue
            lines = f.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable
        for i, line in enumerate(lines):
            if rx.search(line):
                rel = f.relative_to(_root()) if f.is_relative_to(_root()) else f
                lo, hi = max(0, i - context), min(len(lines), i + context + 1)
                out.append("\n".join(f"{rel}:{j + 1}: {lines[j]}" for j in range(lo, hi)))
                hits += 1
                if hits >= max_results:
                    return _clip("\n".join(out) + f"\n... stopped at {max_results} matches")
    return _clip("\n".join(out) or "(no matches)")


MODEL: ContextVar[str] = ContextVar("model", default="")


def delegate(task: str) -> str:
    from . import agent  # late import: agent imports this module
    return agent.run_subagent(task, MODEL.get())


READONLY = frozenset({"delegate", "outline", "read_file", "read_files", "list_dir", "search", "list_blocks", "read_block", "web_fetch", "web_search"})


def write_file(path: str, content: str) -> str:
    p = _p(path)
    if err := _guard(p, content):
        return err
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} chars to {p}"


def edit_file(path: str, old: str = "", new: str = "", replace_all: bool = False, edits: list | None = None) -> str:
    """Replace exact strings in a file. `edits` applies several changes in order in ONE call; if any one does not
    match, nothing is written."""
    p = _p(path)
    todo = [{"old": e.get("old", ""), "new": e.get("new", ""), "replace_all": bool(e.get("replace_all"))} for e in (edits or [])
            if isinstance(e, dict)] or [{"old": old, "new": new, "replace_all": replace_all}]
    if err := _guard(p, "".join(e["old"] + e["new"] for e in todo)):
        return err
    text = p.read_text(encoding="utf-8")
    total = 0
    for k, e in enumerate(todo, 1):
        label = f"edit {k}: " if len(todo) > 1 else ""
        n = text.count(e["old"]) if e["old"] else 0
        if n == 0:
            return f"ERROR: {label}old string not found" + ("; nothing was written" if len(todo) > 1 else "")
        if n > 1 and not e["replace_all"]:
            return f"ERROR: {label}old string appears {n} times; add context or set replace_all" + ("; nothing was written" if len(todo) > 1 else "")
        text = text.replace(e["old"], e["new"]) if e["replace_all"] else text.replace(e["old"], e["new"], 1)
        total += n if e["replace_all"] else 1
    p.write_text(text, encoding="utf-8")
    return f"edited {p} ({total} replacement{'s' if total != 1 else ''}{f' in {len(todo)} edits' if len(todo) > 1 else ''})"


def list_dir(path: str = ".") -> str:
    p = _p(path)
    return "\n".join(sorted(f"{c.name}{'/' if c.is_dir() else ''}" for c in p.iterdir())) or "(empty)"


def _pid() -> str:
    p = PROJECT.get()
    if not p:
        raise RuntimeError("no active project")
    return p["id"]


def list_blocks() -> str:
    b = store.list_blocks(_pid())
    return "\n".join(f"{x['name']} ({x['size']} chars)" for x in b) or "(no blocks yet)"


def read_block(name: str) -> str:
    return store.read_block(_pid(), name)


def write_block(name: str, content: str, append: bool = False) -> str:
    store.write_block(_pid(), name, content, append)
    return f"saved block '{name}'"


class _Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.out, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript"):
            self.skip = max(0, self.skip - 1)

    def handle_data(self, data):
        if not self.skip and data.strip():
            self.out.append(data.strip())


def web_fetch(url: str, max_chars: int = 15000) -> str:
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0 harness"})
    ctype = r.headers.get("content-type", "")
    if "html" in ctype:
        t = _Text()
        t.feed(r.text)
        body = re.sub(r"\n{3,}", "\n\n", "\n".join(t.out))
    else:
        body = r.text
    return _clip(f"HTTP {r.status_code}\n{body}", max_chars)


def web_search(query: str) -> str:
    """Uses Gemini's built-in Google Search grounding in a separate call."""
    alias = llm.gemini_alias()
    if not alias:
        return "ERROR: web_search needs a Gemini model configured; use web_fetch with a URL instead."
    content, _, cand = llm.generate(
        alias, [{"role": "user", "parts": [{"text": query}]}],
        system="Answer using web search. Be factual and concise.", tools=[{"googleSearch": {}}])
    text = "".join(p.get("text", "") for p in content["parts"] if not p.get("thought"))
    chunks = (cand.get("groundingMetadata") or {}).get("groundingChunks", [])
    srcs = "\n".join(f"- {c['web'].get('title', '')}: {c['web'].get('uri', '')}" for c in chunks if "web" in c)
    return _clip(text + ("\n\nSources:\n" + srcs if srcs else ""))


def _obj(props: dict, required: list[str]) -> dict:
    return {"type": "OBJECT", "properties": props, "required": required}


S, I, B = {"type": "STRING"}, {"type": "INTEGER"}, {"type": "BOOLEAN"}

REGISTRY = {
    "bash": (bash, "Run a shell command in the working directory. Returns exit code, stdout, stderr.",
             _obj({"command": S, "timeout": I}, ["command"])),
    "read_file": (read_file, "Read a text file with line numbers. Returns up to 2000 lines: read a whole file in one call; "
                  "use offset/limit only for longer files.",
                  _obj({"path": S, "offset": I, "limit": I}, ["path"])),
    "read_files": (read_files, "Read several text files in ONE call (preferred over many read_file calls).",
                   _obj({"paths": {"type": "ARRAY", "items": S}, "limit": I}, ["paths"])),
    "outline": (outline, "List the functions, classes and headings of a file with line numbers (and where each ends). Use it on "
                "a big file to find the right place, then read_file just that range.", _obj({"path": S}, ["path"])),
    "search": (search, "Regex search across files (case-insensitive); returns file:line: text. Use this instead of "
               "reading files to find something.", _obj({"pattern": S, "path": S, "glob": S, "context": I, "max_results": I}, ["pattern"])),
    "write_file": (write_file, "Create or overwrite a file.", _obj({"path": S, "content": S}, ["path", "content"])),
    "edit_file": (edit_file, "Replace an exact string in a file. For several changes to one file pass `edits` (a list of "
                  "{old, new, replace_all}) in ONE call instead of one call per change; all apply or none do.",
                  _obj({"path": S, "old": S, "new": S, "replace_all": B,
                        "edits": {"type": "ARRAY", "items": _obj({"old": S, "new": S, "replace_all": B}, ["old", "new"])}}, ["path"])),
    "list_dir": (list_dir, "List a directory.", _obj({"path": S}, [])),
    "list_blocks": (list_blocks, "List this project's persistent context blocks.", _obj({}, [])),
    "read_block": (read_block, "Read a context block by name.", _obj({"name": S}, ["name"])),
    "write_block": (write_block, "Create/overwrite (or append to) a context block. Use for durable project facts: "
                    "goals, decisions, architecture, open tasks. Keep blocks short and current; not for chat logs.",
                    _obj({"name": S, "content": S, "append": B}, ["name", "content"])),
    "delegate": (delegate, "Hand a self-contained research task to a sub-agent with its own fresh context and read-only "
                 "tools (files, search, web). Only its final answer comes back, so big file reads never fill your "
                 "context. Use it for broad exploration ('where is X handled?', 'summarize this folder', 'compare these "
                 "docs'). Give it all the detail it needs; it cannot see this conversation. Several delegate calls in "
                 "one step run in parallel.", _obj({"task": S}, ["task"])),
    "web_fetch": (web_fetch, "Fetch a URL and return readable text.", _obj({"url": S, "max_chars": I}, ["url"])),
    "web_search": (web_search, "Search the web (Google grounding) and return an answer with sources.",
                   _obj({"query": S}, ["query"])),
}

DECLARATIONS = [{"functionDeclarations": [
    {"name": n, "description": d, "parameters": sch} for n, (_, d, sch) in REGISTRY.items()]}]


def run(name: str, args: dict) -> str:
    if name not in REGISTRY:
        return f"ERROR: unknown tool {name}"
    try:
        return str(REGISTRY[name][0](**{k: v for k, v in (args or {}).items()}))
    except Exception as e:  # tool errors go back to the model so it can recover
        return f"ERROR: {type(e).__name__}: {e}"

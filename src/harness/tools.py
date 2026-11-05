"""Tools exposed to the model. No approval prompts: everything runs immediately."""
import re
import subprocess
from contextvars import ContextVar
from html.parser import HTMLParser
from pathlib import Path

import requests

from . import config, llm, store

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
    return p if p.is_absolute() else _root() / p


def bash(command: str, timeout: int = 120) -> str:
    try:
        r = subprocess.run(command, shell=True, cwd=_root(), capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        return _clip(f"TIMEOUT after {timeout}s\n{e.stdout or ''}{e.stderr or ''}")
    return _clip(f"exit={r.returncode}\n{r.stdout}{('[stderr]' + chr(10) + r.stderr) if r.stderr else ''}")


def read_file(path: str, offset: int = 1, limit: int = 2000) -> str:
    lines = _p(path).read_text(errors="replace").splitlines()
    chunk = lines[max(offset, 1) - 1: max(offset, 1) - 1 + limit]
    return _clip("\n".join(f"{i}\t{l}" for i, l in enumerate(chunk, max(offset, 1))))


def write_file(path: str, content: str) -> str:
    p = _p(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"wrote {len(content)} chars to {p}"


def edit_file(path: str, old: str, new: str, replace_all: bool = False) -> str:
    p = _p(path)
    text = p.read_text()
    n = text.count(old)
    if n == 0:
        return "ERROR: old string not found"
    if n > 1 and not replace_all:
        return f"ERROR: old string appears {n} times; add context or set replace_all"
    p.write_text(text.replace(old, new) if replace_all else text.replace(old, new, 1))
    return f"edited {p} ({n if replace_all else 1} replacement)"


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
    "read_file": (read_file, "Read a text file with line numbers.",
                  _obj({"path": S, "offset": I, "limit": I}, ["path"])),
    "write_file": (write_file, "Create or overwrite a file.", _obj({"path": S, "content": S}, ["path", "content"])),
    "edit_file": (edit_file, "Replace an exact string in a file.",
                  _obj({"path": S, "old": S, "new": S, "replace_all": B}, ["path", "old", "new"])),
    "list_dir": (list_dir, "List a directory.", _obj({"path": S}, [])),
    "list_blocks": (list_blocks, "List this project's persistent context blocks.", _obj({}, [])),
    "read_block": (read_block, "Read a context block by name.", _obj({"name": S}, ["name"])),
    "write_block": (write_block, "Create/overwrite (or append to) a context block. Use for durable project facts: "
                    "goals, decisions, architecture, open tasks. Keep blocks short and current; not for chat logs.",
                    _obj({"name": S, "content": S, "append": B}, ["name", "content"])),
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

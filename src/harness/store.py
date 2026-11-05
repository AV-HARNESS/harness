"""Projects -> (context blocks + sessions), all plain files under ~/.harness/projects/<id>/.

    project.json   name, root folder, default model
    blocks/*.md    persistent context notes, loaded into every chat in the project
    sessions/*.json  full chat histories
"""
import json
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

from . import config

_lock = threading.Lock()
_ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_BLOCK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "project"


def _write(path: Path, text: str) -> None:
    with _lock:
        path.write_text(text)


# ---------------- projects ----------------
def _pdir(pid: str) -> Path:
    if not _ID.match(pid or ""):
        raise KeyError(pid)
    d = config.PROJECTS_DIR / pid
    if not (d / "project.json").exists():
        raise KeyError(pid)
    return d


def _check_root(root) -> str:
    r = Path(root or config.WORKDIR).expanduser().resolve()
    if not r.is_dir():
        raise ValueError(f"root folder does not exist: {r}")
    return str(r)


def create_project(name: str, root: str | None = None, model: str | None = None) -> dict:
    root = _check_root(root)
    base = pid = _slug(name)
    n = 2
    while (config.PROJECTS_DIR / pid).exists():
        pid, n = f"{base}-{n}", n + 1
    d = config.PROJECTS_DIR / pid
    (d / "blocks").mkdir(parents=True)
    (d / "sessions").mkdir()
    p = {"id": pid, "name": name, "root": root, "model": model or config.DEFAULT_MODEL, "created": time.time()}
    _write(d / "project.json", json.dumps(p, indent=2))
    return p


def get_project(pid: str) -> dict:
    return json.loads((_pdir(pid) / "project.json").read_text())


def list_projects() -> list[dict]:
    out = []
    for f in config.PROJECTS_DIR.glob("*/project.json"):
        try:
            out.append(json.loads(f.read_text()))
        except Exception:
            pass
    return sorted(out, key=lambda p: p.get("created", 0))


def update_project(pid: str, name=None, root=None, model=None) -> dict:
    p = get_project(pid)
    if name:
        p["name"] = name
    if root:
        p["root"] = _check_root(root)
    if model in config.MODELS:
        p["model"] = model
    _write(_pdir(pid) / "project.json", json.dumps(p, indent=2))
    return p


def delete_project(pid: str) -> None:
    shutil.rmtree(_pdir(pid))  # deletes blocks + sessions, never the linked root folder


def ensure_default() -> None:
    """First run: create a 'default' project and migrate any pre-project sessions into it."""
    if list_projects():
        return
    p = create_project("default")
    legacy = config.LEGACY_SESSIONS_DIR
    if legacy.is_dir():
        for f in legacy.glob("*.json"):
            try:
                s = json.loads(f.read_text())
                s["project"] = p["id"]
                _write(config.PROJECTS_DIR / p["id"] / "sessions" / f.name, json.dumps(s, ensure_ascii=False))
            except Exception:
                pass


# ---------------- blocks ----------------
def _bpath(pid: str, name: str) -> Path:
    if not _BLOCK.match(name or ""):
        raise ValueError("block names: letters, digits, - and _ only")
    return _pdir(pid) / "blocks" / f"{name}.md"


def list_blocks(pid: str) -> list[dict]:
    return [{"name": f.stem, "size": f.stat().st_size} for f in sorted((_pdir(pid) / "blocks").glob("*.md"))]


def read_block(pid: str, name: str) -> str:
    p = _bpath(pid, name)
    if not p.exists():
        raise KeyError(name)
    return p.read_text()


def write_block(pid: str, name: str, content: str, append: bool = False) -> None:
    p = _bpath(pid, name)
    if append and p.exists():
        content = p.read_text().rstrip("\n") + "\n" + content
    _write(p, content)


def delete_block(pid: str, name: str) -> None:
    p = _bpath(pid, name)
    if p.exists():
        p.unlink()


# ---------------- sessions ----------------
def _spath(sid: str) -> Path:
    if not re.match(r"^[a-z0-9]+$", sid or ""):
        raise KeyError(sid)
    hits = list(config.PROJECTS_DIR.glob(f"*/sessions/{sid}.json"))
    if not hits:
        raise KeyError(sid)
    return hits[0]


def create(project: str, title: str = "New session", model: str | None = None, instructions: str = "") -> dict:
    p = get_project(project)
    s = {
        "id": uuid.uuid4().hex[:10], "project": project, "title": title, "model": model or p["model"],
        "instructions": instructions, "created": time.time(), "updated": time.time(),
        "messages": [], "prompt_tokens": 0,
    }
    save(s)
    return s


def load(sid: str) -> dict:
    return json.loads(_spath(sid).read_text())


def save(s: dict) -> None:
    s["updated"] = time.time()
    path = _pdir(s["project"]) / "sessions" / f"{s['id']}.json"
    _write(path, json.dumps(s, ensure_ascii=False))


def delete(sid: str) -> None:
    try:
        _spath(sid).unlink()
    except KeyError:
        pass


def list_all(project: str) -> list[dict]:
    out = []
    for p in (_pdir(project) / "sessions").glob("*.json"):
        try:
            s = json.loads(p.read_text())
        except Exception:
            continue
        out.append({k: s.get(k) for k in ("id", "project", "title", "model", "created", "updated", "prompt_tokens")})
    return sorted(out, key=lambda x: x["updated"] or 0, reverse=True)

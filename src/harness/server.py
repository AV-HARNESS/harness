"""Local web UI + JSON/SSE API. Binds to localhost only."""
import json
import os
import queue
import re
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from . import agent, auth, config, errors, store, tools

app = FastAPI(title="Gemini harness")
store.ensure_default()
STATIC = Path(__file__).parent / "static"
_stops: dict[str, threading.Event] = {}
_busy: set[str] = set()
_inboxes: dict[str, queue.Queue] = {}
_live: dict[str, dict] = {}  # sessions being worked on: the one in-memory copy every reader and writer shares


class ProjectIn(BaseModel):
    name: str | None = None
    root: str | None = None
    model: str | None = None


class BlockIn(BaseModel):
    content: str


class Patch(BaseModel):
    title: str | None = None
    model: str | None = None
    instructions: str | None = None
    effort: str | None = None


class New(Patch):
    project: str


class Chat(BaseModel):
    text: str
    model: str | None = None
    effort: str | None = None
    edit_index: int | None = None  # edit: cut history at this user message, then run the new text from there


def _get(sid: str) -> dict:
    if sid in _live:  # a running session: use the live object, so a title change or a refresh never works from a stale file copy
        return _live[sid]
    try:
        return store.load(sid)
    except KeyError:
        raise HTTPException(404, "no such session")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})  # always pick up UI updates


@app.get("/api/config")
def get_config():
    import os
    from . import discover
    has_api_key = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    discover._start_probes()  # confirm in the background that each configured key really works
    ok, hidden = discover.usable_models()  # only aliases with credentials that the provider has not rejected
    default = config.DEFAULT_MODEL if config.DEFAULT_MODEL in ok else next(iter(ok), config.DEFAULT_MODEL)
    return {"models": {k: f"{v.get('provider', 'gemini')}:{v['model']}" for k, v in ok.items()},
            "hidden_models": hidden, "default_model": default, "project": config.PROJECT,
            "location": config.LOCATION, "logged_in": auth.is_logged_in(),
            "has_api_key": has_api_key, "compact_at": config.COMPACT_AT}


@app.get("/api/log", response_class=PlainTextResponse)
def log(ref: str = "", tail: int = 200):
    """The harness log (this server only listens on 127.0.0.1). ?ref=abc123 shows just that error."""
    if ref and not re.fullmatch(r"[0-9a-f]{6}", ref):
        raise HTTPException(400, "bad reference")
    return errors.log_lines(ref, min(max(tail, 1), 2000))


@app.get("/api/models")
def models(refresh: bool = False, recheck: bool = False):
    """Models each configured endpoint offers AND your key can actually call (confirmed in the background;
    poll while `pending` > 0). refresh=true re-lists, recheck=true also forgets earlier access results."""
    from . import discover
    return discover.discover(refresh, recheck)


def _proj(pid: str) -> dict:
    try:
        return store.get_project(pid)
    except KeyError:
        raise HTTPException(404, "no such project")


@app.get("/api/places")
def places():
    """Sidebar shortcuts for the folder picker: standard folders, folders of your projects, drives (Windows)."""
    home = Path.home()
    fav, seen = [], set()

    def add(name, icon, *cands):
        for c in cands:
            if c.is_dir() and str(c) not in seen:
                seen.add(str(c))
                fav.append({"name": name, "icon": icon, "path": str(c)})
                return

    add("Home", "home", home)
    for n in ("Desktop", "Documents", "Downloads"):
        add(n, n.lower(), home / n, *sorted(home.glob(f"OneDrive*/{n}")))
    proj, seen_p = [], set()
    for p in reversed(store.list_projects()):
        r = p.get("root", "")
        if r and r not in seen_p and Path(r).is_dir():
            seen_p.add(r)
            proj.append({"name": Path(r).name or r, "path": r, "project": p["name"]})
    drives = [{"name": f"{c}:", "path": f"{c}:\\"} for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
              if Path(f"{c}:\\").exists()] if os.name == "nt" else []
    return {"favorites": fav, "projects": proj, "drives": drives, "sep": os.sep}


@app.get("/api/fs")
def browse(path: str = "", files: bool = False, project: str = ""):
    """List folders (and optionally files) so the UI can link a folder. With `project`, stays inside its root."""
    root = Path(_proj(project)["root"]).resolve() if project else None
    if os.name == "nt" and path == "@drives" and not root:
        drives = [f"{c}:\\" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if Path(f"{c}:\\").exists()]
        return {"path": "@drives", "parent": None, "entries": [{"name": d, "dir": True, "rel": d, "mtime": 0} for d in drives]}
    p = Path(path).expanduser() if path else (root or Path.home())
    try:
        p = p.resolve()
    except OSError as e:
        raise HTTPException(400, str(e))
    if root and not p.is_relative_to(root):
        raise HTTPException(400, "outside the project folder")
    if not p.is_dir():
        raise HTTPException(400, f"not a folder: {p}")
    entries = []
    try:
        for c in sorted(p.iterdir(), key=lambda c: (not c.is_dir(), c.name.lower())):
            if c.name.startswith("."):
                continue
            try:
                isdir = c.is_dir()
            except OSError:
                continue
            if isdir or (files and tools._guard(c) is None):
                rel = c.relative_to(root).as_posix() if root else str(c)
                try:
                    mtime = int(c.stat().st_mtime)
                except OSError:
                    mtime = 0
                entries.append({"name": c.name, "dir": isdir, "rel": rel, "mtime": mtime})
            if len(entries) >= 500:
                break
    except PermissionError:
        raise HTTPException(400, "permission denied")
    parent = None if (root and p == root) else (str(p.parent) if p.parent != p else ("@drives" if os.name == "nt" and not root else None))
    return {"path": str(p), "parent": parent, "entries": entries}


@app.get("/api/projects")
def list_projects():
    return store.list_projects()


@app.post("/api/projects")
def new_project(b: ProjectIn):
    if not b.name:
        raise HTTPException(400, "name required")
    try:
        return store.create_project(b.name, b.root, b.model)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.patch("/api/projects/{pid}")
def patch_project(pid: str, b: ProjectIn):
    _proj(pid)
    try:
        return store.update_project(pid, b.name, b.root, b.model)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.delete("/api/projects/{pid}")
def del_project(pid: str):
    _proj(pid)
    store.delete_project(pid)  # removes blocks + sessions only; the root folder is untouched
    return {"ok": True}


@app.get("/api/projects/{pid}/blocks")
def get_blocks(pid: str):
    _proj(pid)
    return [{**b, "content": store.read_block(pid, b["name"])} for b in store.list_blocks(pid)]


@app.put("/api/projects/{pid}/blocks/{name}")
def put_block(pid: str, name: str, b: BlockIn):
    _proj(pid)
    try:
        store.write_block(pid, name, b.content)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


@app.delete("/api/projects/{pid}/blocks/{name}")
def del_block(pid: str, name: str):
    _proj(pid)
    store.delete_block(pid, name)
    return {"ok": True}


@app.get("/api/sessions")
def list_sessions(project: str):
    _proj(project)
    return store.list_all(project)


@app.post("/api/sessions")
def new_session(b: New):
    _proj(b.project)
    return store.create(b.project, b.title or "New session", b.model, b.instructions or "")


@app.get("/api/sessions/{sid}")
def get_session(sid: str):
    return {**_get(sid), "running": sid in _busy}


@app.patch("/api/sessions/{sid}")
def patch_session(sid: str, b: Patch):
    s = _get(sid)
    if b.title is not None:
        s["title"] = b.title
    if b.model and config.get_model(b.model):
        s["model"] = b.model
    if b.instructions is not None:
        s["instructions"] = b.instructions
    if b.effort is not None:
        s["effort"] = b.effort if b.effort in ("low", "medium", "high") else ""
    store.save(s)
    return s


@app.delete("/api/sessions/{sid}")
def del_session(sid: str):
    if sid in _busy:
        raise HTTPException(409, "session is running; stop it first")
    store.delete(sid)
    return {"ok": True}


@app.post("/api/sessions/{sid}/stop")
def stop(sid: str):
    if sid in _stops:
        _stops[sid].set()
    return {"ok": True}


@app.post("/api/sessions/{sid}/compact")
def compact(sid: str):
    if sid in _busy:
        raise HTTPException(409, "session is running; try again when it finishes")
    s = _get(sid)
    return {"compacted": agent.compact(s)}


@app.post("/api/sessions/{sid}/queue")
def queue_message(sid: str, b: Chat):
    """Add a message while the agent is working; it is delivered at the agent's next step."""
    if sid not in _busy:
        raise HTTPException(409, "session is not running; send it as a normal message")
    if not b.text.strip():
        raise HTTPException(400, "empty message")
    _inboxes[sid].put(b.text)
    return {"ok": True}


@app.post("/api/sessions/{sid}/chat")
def chat(sid: str, b: Chat):
    s = _get(sid)
    if sid in _busy:
        raise HTTPException(409, "session is already running")
    if b.model and config.get_model(b.model):
        s["model"] = b.model
    if b.effort is not None:
        s["effort"] = b.effort if b.effort in ("low", "medium", "high") else ""
    if b.edit_index is not None:
        i, msgs = b.edit_index, s["messages"]
        if not (0 <= i < len(msgs)) or not agent._is_user_text(msgs[i]):
            raise HTTPException(400, "that message can't be edited")
        s["messages"] = msgs[:i]
        s["prompt_tokens"] = 0
        s.pop("compact_base", None)
        store.save(s)
    _live[sid] = s
    q: queue.Queue = queue.Queue()
    ev = _stops[sid] = threading.Event()
    inbox = _inboxes[sid] = queue.Queue()
    _busy.add(sid)

    from . import discover

    def worker():
        try:
            _, hidden = discover.usable_models()
            m = s.get("model", "")
            base = m if m in config.MODELS else m.split("/")[0]
            if base in hidden:  # e.g. the session was on 'claude' but only an OpenAI key is set: switch, don't fail
                alt = discover.fallback_model()
                if not alt:
                    raise errors.ConfigError("No model is available on this machine (" + "; ".join(f"{k}: {v}" for k, v in hidden.items())
                                             + "). Add an API key (OPENAI_API_KEY, GEMINI_API_KEY or ANTHROPIC_API_KEY) to your .env and restart.")
                q.put({"type": "notice", "model": alt,
                       "text": f"'{m}' isn't available here ({hidden[base]}); using '{alt}' instead."})
                s["model"] = alt
                store.save(s)
            agent.run_turn(s, b.text, q.put, ev, inbox)
        except Exception as e:
            import re as _re
            spec = config.get_model(s.get("model", ""))
            if spec and _re.search(r"HTTP (401|403|404)", str(e)):  # the provider said no: stop offering this model now
                discover.record(spec, "denied")
            msg, ref = errors.report_ref(e, f"chat session={sid} model={s.get('model')}")
            q.put({"type": "error", "error": msg, "ref": ref})
        finally:
            _busy.discard(sid)
            _live.pop(sid, None)
            left = agent._drain(inbox)  # queued but never delivered (e.g. stopped): UI sends them next
            if left:
                q.put({"type": "leftover", "texts": left})
            q.put({"type": "done"})

    threading.Thread(target=worker, daemon=True).start()

    def stream():
        while True:
            e = q.get()
            yield f"data: {json.dumps(e)}\n\n"
            if e["type"] == "done":
                return

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def serve(port: int = 8765):
    import threading, webbrowser, uvicorn, signal, sys
    store.ensure_default()
    url = f"http://127.0.0.1:{port}"
    print(f"UI: {url}  (Ctrl+C to quit)\nData folder: {config.HOME}\nLog (errors also print here): {config.HOME / 'harness.log'}   view: {url}/api/log")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    def _shutdown(sig, frame):
        print("\nStopping...")
        sys.exit(0)
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")

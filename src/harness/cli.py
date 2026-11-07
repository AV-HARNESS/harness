"""Command line: login | chat (default) | serve | projects."""
import argparse
try:
    import readline  # noqa: F401  (line editing + history; not available on Windows)
except ImportError:
    pass
import threading
from pathlib import Path

from . import agent, auth, config, store

DIM, CYAN, YEL, RED, RST = "\033[2m", "\033[36m", "\033[33m", "\033[31m", "\033[0m"


def _emit(e: dict) -> None:
    t = e["type"]
    if t == "text":
        print(e["text"])
    elif t == "tool_call":
        arg = ", ".join(f"{k}={str(v)[:80]!r}" for k, v in e["args"].items())
        print(f"{CYAN}> {e['name']}({arg}){RST}")
    elif t == "tool_result":
        out = e["output"]
        print(f"{DIM}{out[:600]}{'...' if len(out) > 600 else ''}{RST}")
    elif t == "compacted":
        print(f"{YEL}[context compacted]{RST}")


HELP = """projects:  /projects   /project <id>   /project new <name> [root]   /root <path>
blocks:    /blocks   /block <name>   (the agent also edits blocks itself)
sessions:  /new [title]   /sessions   /resume <id>   /title <text>
other:     /model flash|pro   /compact   /tokens   /quit"""


def _find_project(key: str) -> dict | None:
    key = key.strip().lower()
    for p in store.list_projects():
        if key in (p["id"], p["name"].lower()):
            return p
    return None


def _latest_or_new(project: dict) -> dict:
    ss = store.list_all(project["id"])
    return store.load(ss[0]["id"]) if ss else store.create(project["id"])


def project_for_cwd() -> dict:
    """The project whose root contains the current folder (deepest match), or a new one for this folder."""
    cwd = Path.cwd().resolve()
    best = None
    for p in store.list_projects():
        r = Path(p["root"]).resolve()
        if (cwd == r or r in cwd.parents) and (best is None or len(str(r)) > len(str(best[0]))):
            best = (r, p)
    if best:
        return best[1]
    return store.create_project(cwd.name or "project", str(cwd))


def chat(resume: str | None = None, project: str | None = None) -> None:
    store.ensure_default()
    if resume:
        s = store.load(resume)
        proj = store.get_project(s["project"])
    else:
        proj = (_find_project(project) if project else None) or project_for_cwd()  # default: the folder you are in
        s = _latest_or_new(proj)

    def banner():
        print(f"{DIM}data {config.HOME}{RST}")
        print(f"{DIM}project {proj['name']} ({proj['root']}) - session {s['id']} '{s['title']}' - {s['model']} - /help{RST}")

    banner()
    while True:
        try:
            line = input(f"\n[{proj['id']}:{s['model']}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line.startswith("/"):
            cmd, _, arg = line[1:].partition(" ")
            arg = arg.strip()
            try:
                if cmd in ("quit", "exit", "q"):
                    return
                elif cmd == "help":
                    print(HELP)
                elif cmd == "projects":
                    for p in store.list_projects():
                        print(f"{'*' if p['id'] == proj['id'] else ' '} {p['id']:20} {p['root']}")
                elif cmd == "project":
                    if arg.startswith("new "):
                        name, _, root = arg[4:].strip().partition(" ")
                        proj = store.create_project(name, root.strip() or None)
                        s = store.create(proj["id"])
                    else:
                        p = _find_project(arg)
                        if not p:
                            print(f"{RED}no such project{RST}")
                            continue
                        proj, s = p, _latest_or_new(p)
                    banner()
                elif cmd == "root":
                    proj = store.update_project(proj["id"], root=arg)
                    print(f"root -> {proj['root']}")
                elif cmd == "blocks":
                    for b in store.list_blocks(proj["id"]) or []:
                        print(f"{b['name']} ({b['size']} chars)")
                elif cmd == "block":
                    print(store.read_block(proj["id"], arg))
                elif cmd == "new":
                    s = store.create(proj["id"], arg or "New session", s["model"])
                    print(f"new session {s['id']}")
                elif cmd == "sessions":
                    for x in store.list_all(proj["id"])[:20]:
                        print(f"{x['id']}  {x['model']:5} {x['title']}")
                elif cmd == "resume":
                    s = store.load(arg)
                    proj = store.get_project(s["project"])
                    banner()
                elif cmd == "title":
                    s["title"] = arg
                    store.save(s)
                elif cmd == "model":
                    if arg in config.MODELS:
                        s["model"] = arg
                        store.save(s)
                    else:
                        print("use: /model <alias>  (" + ", ".join(config.MODELS) + ")")
                elif cmd == "compact":
                    print("compacted" if agent.compact(s, _emit) else "nothing to compact")
                elif cmd == "tokens":
                    print(f"{s.get('prompt_tokens', 0)} prompt tokens, {len(s['messages'])} messages")
                else:
                    print(HELP)
            except (KeyError, ValueError) as e:
                print(f"{RED}error: {e}{RST}")
            continue
        stop = threading.Event()
        try:
            agent.run_turn(s, line, _emit, stop)
        except KeyboardInterrupt:
            stop.set()
            print(f"{YEL}[interrupted]{RST}")
        except Exception as e:
            print(f"{RED}error: {e}{RST}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="harness")
    sub = ap.add_subparsers(dest="cmd")
    lg = sub.add_parser("login", help="one-time OAuth login")
    lg.add_argument("--port", type=int, default=0, help="fixed local port (needed if your OAuth client is a Web client with registered redirect URIs)")
    vf = sub.add_parser("verify", help="live check of auth, each model, and tool calling")
    vf.add_argument("aliases", nargs="*", help="model aliases (default: all configured)")
    c = sub.add_parser("chat", help="terminal chat (default)")
    c.add_argument("--resume", help="session id")
    c.add_argument("--project", help="project id or name")
    sv = sub.add_parser("serve", help="local web UI")
    sv.add_argument("--port", type=int, default=8765)
    sub.add_parser("projects", help="list projects")
    rd = sub.add_parser("redact", help="preview what would be masked in some text (reads stdin if none given)")
    rd.add_argument("text", nargs="*")
    a = ap.parse_args()
    if a.cmd == "login":
        auth.login(a.port)
    elif a.cmd == "verify":
        from .verify import main as verify_main
        raise SystemExit(verify_main(a.aliases))
    elif a.cmd == "serve":
        from .server import serve
        serve(a.port)
    elif a.cmd == "redact":
        import sys
        from collections import Counter

        from . import redact
        counts: Counter = Counter()
        print(redact.mask(" ".join(a.text) if a.text else sys.stdin.read(), counts))
        redact.report(counts)
    elif a.cmd == "projects":
        store.ensure_default()
        for p in store.list_projects():
            print(f"{p['id']:20} {p['root']}")
    else:
        chat(getattr(a, "resume", None), getattr(a, "project", None))

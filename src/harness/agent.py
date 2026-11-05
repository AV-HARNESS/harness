"""The agent loop: send history -> run requested tools -> feed results back -> repeat."""
import threading
from typing import Callable

from . import config, llm, store, tools

BASE_SYSTEM = (
    "You are a capable autonomous assistant with shell, file and web tools. "
    "Act directly: use tools to get things done, check your results, "
    "and keep going until the task is finished. Be concise in your final answer.\n\n"
    "Project: {name}\nProject root (working directory; relative paths resolve here): {root}\n"
    "Context blocks are persistent notes for this project and are shown below. When goals, decisions, "
    "architecture or open tasks change, update them with write_block so the next session starts informed."
)
MAX_BLOCK, MAX_BLOCKS_TOTAL = 8000, 40000


def build_system(project: dict, s: dict) -> str:
    system = BASE_SYSTEM.format(name=project["name"], root=project["root"])
    used = 0
    for b in store.list_blocks(project["id"]):
        text = store.read_block(project["id"], b["name"])[:MAX_BLOCK]
        if used + len(text) > MAX_BLOCKS_TOTAL:
            system += f"\n\n### {b['name']}\n(omitted: context budget; use read_block)"
            continue
        used += len(text)
        system += f"\n\n### block: {b['name']}\n{text}"
    if s.get("instructions"):
        system += "\n\n### session instructions\n" + s["instructions"]
    return system

Emit = Callable[[dict], None]


def _is_user_text(m: dict) -> bool:
    return m["role"] == "user" and any("text" in p for p in m["parts"])


def compact(s: dict, emit: Emit = lambda e: None) -> bool:
    """Summarize old turns, keep the last KEEP_TURNS full turns verbatim.

    This mirrors how Claude.ai handles long contexts: recent turns stay intact
    (so the model has full tool call detail for ongoing work) while older history
    is compressed into a dense summary to free up the context window.
    """
    user_idxs = [i for i, m in enumerate(s["messages"]) if _is_user_text(m)]
    if len(user_idxs) <= config.KEEP_TURNS:
        return False  # not enough history to compact
    cut = user_idxs[-config.KEEP_TURNS]  # keep last KEEP_TURNS user turns verbatim
    old, keep = s["messages"][:cut], s["messages"][cut:]
    ask = {"role": "user", "parts": [{"text": (
        "Summarize the conversation above so work can continue seamlessly. Include: "
        "goals, decisions made, key facts discovered, file paths touched, commands run, "
        "open tasks. Be dense and specific — this replaces the full history.")}]}
    content, _, _ = llm.generate(s["model"], old + [ask])
    summary = "".join(p.get("text", "") for p in content["parts"] if not p.get("thought"))
    s["messages"] = [
        {"role": "user", "parts": [{"text": f"[Summary of earlier conversation]\n{summary}"}]},
        {"role": "model", "parts": [{"text": "Understood. Continuing from that summary."}]},
    ] + keep
    s["prompt_tokens"] = 0
    store.save(s)
    emit({"type": "compacted", "summary": summary})
    return True


def run_turn(s: dict, text: str, emit: Emit, stop: threading.Event | None = None) -> None:
    project = store.get_project(s["project"])
    tools.PROJECT.set(project)
    s["messages"].append({"role": "user", "parts": [{"text": text}]})
    if s["title"] == "New session":
        s["title"] = text.strip().splitlines()[0][:50] or "New session"
    store.save(s)

    for _ in range(config.MAX_STEPS):
        if stop and stop.is_set():
            emit({"type": "text", "text": "(stopped)"})
            break
        if s.get("prompt_tokens", 0) > config.COMPACT_AT:
            compact(s, emit)
        system = build_system(project, s)  # rebuilt each step so block edits apply immediately
        content, usage, _ = llm.generate(s["model"], s["messages"], system, tools.DECLARATIONS)
        s["messages"].append(content)
        s["prompt_tokens"] = usage.get("promptTokenCount", 0)
        calls = []
        for p in content["parts"]:
            if p.get("thought"):
                continue
            if p.get("text"):
                emit({"type": "text", "text": p["text"]})
            if "functionCall" in p:
                calls.append(p["functionCall"])
        emit({"type": "usage", "prompt_tokens": s["prompt_tokens"]})
        if not calls:
            break
        results = []
        for c in calls:
            emit({"type": "tool_call", "name": c["name"], "args": c.get("args", {})})
            out = tools.run(c["name"], c.get("args", {}))
            emit({"type": "tool_result", "name": c["name"], "output": out})
            # Store a history-trimmed version to prevent old tool outputs bloating context.
            # Full output was already emitted to the UI above.
            stored = out if len(out) <= config.HISTORY_TOOL_TRIM else out[:config.HISTORY_TOOL_TRIM] + f"\n... [trimmed in history; {len(out)} chars total]"
            results.append({"functionResponse": {"name": c["name"], "response": {"output": stored}}})
        s["messages"].append({"role": "user", "parts": results})
        store.save(s)
    else:
        emit({"type": "text", "text": f"(stopped after {config.MAX_STEPS} steps; say 'continue' to go on)"})
    store.save(s)

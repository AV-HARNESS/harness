"""The agent loop: send history -> run requested tools -> feed results back -> repeat."""
import re
import threading
from pathlib import Path
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
GUIDE_FILES = ("AGENTS.md", "CLAUDE.md")
GUIDE_HOWTO = (
    "Keep it short and factual: what the project is, how to run/test/build it, folder layout, "
    "conventions, gotchas and decisions. Never put secrets, tokens or personal data in it.")


def _guide_section(root: str) -> str:
    """The project's own AGENTS.md / CLAUDE.md if it has one; otherwise ask the agent to write AGENTS.md."""
    for name in GUIDE_FILES:
        f = Path(root) / name
        if f.is_file():
            text = f.read_text(encoding="utf-8", errors="replace")[:MAX_BLOCK]
            return (f"\n\n### project file: {name} (in the project root)\n{text}\n\n"
                    f"When you learn something durable about this project that is missing or wrong in {name}, "
                    f"update it with edit_file. {GUIDE_HOWTO}")
    return ("\n\n### project guide\nThis project has no AGENTS.md or CLAUDE.md. After you have explored enough to "
            "know how the project works, create AGENTS.md in the project root with write_file, and keep it "
            f"updated as you learn more. {GUIDE_HOWTO}")


SELECT_ABOVE, SELECT_BUDGET = 6000, 16000  # below SELECT_ABOVE chars of blocks, load them all; above, pick per prompt
_WORD = re.compile(r"[a-z0-9_]{3,}")
_STOP = frozenset("the and for that this with from have what when where which how why can you your are was were not but "
                  "into about please just then them they there their will would should could also all any some more "
                  "make use using get let yes okay sure continue proceed".split())


def _words(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP]


def _recent_user_text(s: dict, n: int = 2) -> str:
    """The last couple of things the user actually typed (not tool results)."""
    out = []
    for m in reversed(s.get("messages", [])):
        if _is_user_text(m):
            out.append(" ".join(p["text"] for p in m["parts"] if "text" in p))
            if len(out) == n:
                break
    return " ".join(reversed(out))


def select_blocks(blocks: list[dict], texts: dict[str, str], s: dict) -> list[str] | None:
    """Names of the blocks worth loading for the user's current request, best first; None = load everything.

    Small block sets load whole. Otherwise blocks are ranked by keyword overlap with the recent prompt (block name
    counts triple, rarer words count more). A prompt with no usable words ("yes, do that") keeps the selection from
    the previous prompt, so follow-ups don't lose context.
    """
    if sum(len(x) for x in texts.values()) <= SELECT_ABOVE:
        return None
    q = _words(_recent_user_text(s))
    if not q:
        keep = [n for n in s.get("block_sel", []) if n in texts]
        return keep or None
    docs = {n: _words(n.replace("-", " ").replace("_", " ")) * 3 + _words(x) for n, x in texts.items()}
    df = {w: sum(1 for d in docs.values() if w in d) for w in set(q)}
    scored = []
    for n, d in docs.items():
        score = sum((1 + d.count(w) ** 0.5) / (1 + df[w]) for w in set(q) if w in d)
        if n.lower() in " ".join(q) or n.lower().replace("-", " ") in _recent_user_text(s).lower():
            score += 5  # the prompt names the block
        if score > 0:
            scored.append((score, n))
    chosen = [n for _, n in sorted(scored, reverse=True)]
    s["block_sel"] = chosen
    return chosen


def build_system(project: dict, s: dict) -> str:
    system = BASE_SYSTEM.format(name=project["name"], root=project["root"])
    system += _guide_section(project["root"])
    blocks = store.list_blocks(project["id"])
    texts = {b["name"]: store.read_block(project["id"], b["name"])[:MAX_BLOCK] for b in blocks}
    chosen = select_blocks(blocks, texts, s)
    order = list(texts) if chosen is None else chosen
    budget = MAX_BLOCKS_TOTAL if chosen is None else min(MAX_BLOCKS_TOTAL, SELECT_BUDGET)
    used = 0
    for name in order:
        text = texts[name]
        if used + len(text) > budget:
            system += f"\n\n### {name}\n(omitted: context budget; use read_block)"
            continue
        used += len(text)
        system += f"\n\n### block: {name}\n{text}"
    skipped = [n for n in texts if n not in order]
    if skipped:
        system += ("\n\n### other blocks (not loaded for this request; use read_block if relevant)\n"
                   + "\n".join(f"- {n}: {(texts[n].strip().splitlines() or [''])[0][:90]}" for n in skipped))
    if s.get("instructions"):
        system += "\n\n### session instructions\n" + s["instructions"]
    return system

def auto_title(s: dict) -> str | None:
    """A short descriptive title (3-6 words) for the session from its first exchange, or None."""
    first = next((" ".join(p["text"] for p in m["parts"] if "text" in p) for m in s["messages"] if _is_user_text(m)), "")
    answer = next((" ".join(p["text"] for p in m["parts"] if p.get("text") and not p.get("thought"))
                   for m in s["messages"] if m["role"] == "model"), "")
    if not first.strip() or not answer.strip():
        return None
    ask = ("Write a 3-6 word title for this conversation, like a document title (Title Case, no quotes, no trailing "
           "punctuation). Reply with the title only.\n\nUser: " + first[:600] + "\n\nAssistant: " + answer[:600])
    content, _, _ = llm.generate(s["model"], [{"role": "user", "parts": [{"text": ask}]}],
                                 "You write short, specific conversation titles.", None, effort="low")
    title = next((p["text"] for p in content["parts"] if p.get("text") and not p.get("thought")), "").strip()
    title = title.splitlines()[0].strip().strip("\"'`*#. ")[:60] if title else ""
    return title or None


Emit = Callable[[dict], None]


def _is_user_text(m: dict) -> bool:
    return m["role"] == "user" and any("text" in p for p in m["parts"])


KEEP_STEPS = 10  # messages kept verbatim when one long turn is itself the bulk of the context


def _chars(msgs: list) -> int:
    import json
    return len(json.dumps(msgs, ensure_ascii=False))


def _step_cut(messages: list) -> int | None:
    """Index of a model message to start the verbatim tail at (so a tool call is never separated from its
    result), leaving about KEEP_STEPS messages. None if there is nothing earlier to summarize."""
    i = max(len(messages) - KEEP_STEPS, 1)
    while i < len(messages) and messages[i]["role"] != "model":
        i += 1
    return i if 1 <= i < len(messages) - 1 else None


def compact(s: dict, emit: Emit = lambda e: None) -> bool:
    """Summarize old history so work can continue, keeping recent turns verbatim.

    Normally old turns are summarized and the last KEEP_TURNS user turns stay intact. When that would not
    shrink anything (one long turn of many tool steps is most of the context), cut inside the turn instead and
    keep only the last few steps, so compaction actually frees space.
    """
    msgs = s["messages"]
    user_idxs = [i for i, m in enumerate(msgs) if _is_user_text(m)]
    cut, ack = None, True
    if len(user_idxs) > config.KEEP_TURNS:
        cut = user_idxs[-config.KEEP_TURNS]
    if cut is None or _chars(msgs[cut:]) / 4 > config.COMPACT_AT / 2:
        step = _step_cut(msgs)
        if step is not None and (cut is None or step > cut):
            cut, ack = step, False
    if cut is None:
        return False  # not enough history to compact
    old, keep = msgs[:cut], msgs[cut:]
    ask = {"role": "user", "parts": [{"text": (
        "Summarize the conversation above so work can continue seamlessly. Include: "
        "goals, decisions made, key facts discovered, file paths touched, commands run, "
        "open tasks. Be dense and specific - this replaces the full history.")}]}
    content, _, _ = llm.generate(s["model"], old + [ask])
    summary = "".join(p.get("text", "") for p in content["parts"] if not p.get("thought"))
    head = [{"role": "user", "parts": [{"text": f"[Summary of earlier conversation]\n{summary}"}]}]
    if ack:
        head.append({"role": "model", "parts": [{"text": "Understood. Continuing from that summary."}]})
    s["messages"] = head + keep
    s["prompt_tokens"] = 0
    s["just_compacted"] = True  # the next usage report tells us how big the context is after compaction
    store.save(s)
    emit({"type": "compacted", "summary": summary})
    return True


def _drain(inbox) -> list[str]:
    """Messages the user queued while the agent was working."""
    out = []
    while inbox is not None:
        try:
            m = inbox.get_nowait()
            if str(m).strip():  # blank messages (e.g. an empty voice transcript) must never reach the model
                out.append(m)
        except Exception:
            break
    return out


def run_turn(s: dict, text: str, emit: Emit, stop: threading.Event | None = None, inbox=None) -> None:
    project = store.get_project(s["project"])
    tools.PROJECT.set(project)
    if not str(text).strip():
        return  # an empty user turn is dropped by Gemini and leaves the request ending with a model turn
    s["messages"].append({"role": "user", "parts": [{"text": text}]})
    if s["title"] == "New session":
        s["title"] = text.strip().splitlines()[0][:50] or "New session"
        s["title_auto"] = True  # a stand-in until auto_title() writes a proper one
    store.save(s)

    for _ in range(config.MAX_STEPS):
        if stop and stop.is_set():
            emit({"type": "text", "text": "(stopped)"})
            break
        pt = s.get("prompt_tokens", 0)
        # Not again until the context has grown ~25% past its size right after the last compaction; otherwise a
        # context that stays big (large files in recent steps) would be re-summarized on every single step.
        if pt > config.COMPACT_AT and pt > s.get("compact_base", 0) * 1.25:
            compact(s, emit)
        system = build_system(project, s)  # rebuilt each step so block edits apply immediately
        if s["messages"][-1]["role"] != "user":  # Gemini rejects requests ending with a model turn
            s["messages"].append({"role": "user", "parts": [{"text": "Continue."}]})
        content, usage, _ = llm.generate(s["model"], s["messages"], system, tools.DECLARATIONS, effort=s.get("effort", ""))
        s["messages"].append(content)
        s["prompt_tokens"] = usage.get("promptTokenCount", 0)
        if s.pop("just_compacted", False):
            s["compact_base"] = s["prompt_tokens"]
        calls = []
        for p in content["parts"]:
            if p.get("thought"):
                continue
            if p.get("text"):
                emit({"type": "text", "text": p["text"]})
            if "functionCall" in p:
                calls.append(p["functionCall"])
        emit({"type": "usage", "prompt_tokens": s["prompt_tokens"], "secs": usage.get("secs")})
        if not calls:
            queued = _drain(inbox)  # user typed more while this answer was being produced: keep going
            if not queued or (stop and stop.is_set()):
                if queued:
                    for t in queued:
                        inbox.put(t)  # stopped: hand them back to the caller
                break
            for t in queued:
                emit({"type": "user", "text": t})
            s["messages"].append({"role": "user", "parts": [{"text": t} for t in queued]})
            store.save(s)
            continue
        results = []
        for c in calls:
            emit({"type": "tool_call", "name": c["name"], "args": c.get("args", {})})
            out = tools.run(c["name"], c.get("args", {}))
            emit({"type": "tool_result", "name": c["name"], "output": out})
            # Store a history-trimmed version to prevent old tool outputs bloating context.
            # Full output was already emitted to the UI above.
            stored = out if len(out) <= config.HISTORY_TOOL_TRIM else out[:config.HISTORY_TOOL_TRIM] + f"\n... [trimmed in history; {len(out)} chars total]"
            results.append({"functionResponse": {"name": c["name"], "response": {"output": stored}}})
        queued = [] if (stop and stop.is_set()) else _drain(inbox)  # delivered with the tool results
        for t in queued:
            emit({"type": "user", "text": t})
        if queued:
            # Gemini endpoints can reject a turn mixing functionResponse and text parts (HTTP 400), so the queued
            # message rides inside the last tool result instead.
            note = "\n\n[The user sent this while you were working - read it and adapt]:\n" + "\n".join(queued)
            results[-1]["functionResponse"]["response"]["output"] += note
        s["messages"].append({"role": "user", "parts": results})
        store.save(s)
    else:
        emit({"type": "text", "text": f"(stopped after {config.MAX_STEPS} steps; say 'continue' to go on)"})
    store.save(s)

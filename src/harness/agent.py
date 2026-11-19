"""The agent loop: send history -> run requested tools -> feed results back -> repeat."""
import re
import threading
from pathlib import Path
from typing import Callable

from . import config, llm, repomap, store, tools
from .providers import STOP, CallStopped, CallTimeout

BASE_SYSTEM = (
    "You are a capable autonomous assistant with shell, file and web tools. "
    "Act directly: use tools to get things done, check your results, "
    "and keep going until the task is finished. Be concise in your final answer.\n"
    "Work in as few steps as possible: every step costs seconds. When several tool calls are independent "
    "(reading files, searching, listing, fetching), issue them together in ONE step instead of one per step; "
    "prefer read_files over many read_file calls and search over opening files to look for text. "
    "For broad exploration (finding where something lives, surveying many files) use delegate: a sub-agent reads "
    "it all in its own context and returns just the answer. Read a whole file in ONE read_file call (it returns up "
    "to 2000 lines) instead of many small slices (a repo map of the project is included below),  and open every file you will need in the same step. For a file "
    "longer than that, call outline first to find the right line range. Whatever you read stays in your context "
    "while you work, so do not read the same thing again. Do not slice files with bash python -c, type or findstr.\n"
    "Output length is the slowest part: generating text takes seconds per few hundred tokens. Change existing files "
    "with edit_file (only the changed lines; several changes to one file go in ONE call as edits=[...]), not by rewriting them with write_file; keep prose and plans short; "
    "do not repeat file contents or tool output back in your replies.\n\n"
    "Project: {name}\nProject root (working directory; relative paths resolve here): {root}\n"
    "Context blocks are persistent notes for this project and are shown below. When goals, decisions, "
    "architecture or open tasks change, update them with write_block so the next session starts informed."
)
BLOCK_WRITERS = frozenset({"write_block", "delete_block"})
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
    if config.REPO_MAP:
        try:
            rm = repomap.build(project["root"], config.REPO_MAP_CHARS)
        except Exception:  # a map is a convenience: never let it break a turn
            rm = ""
        if rm:
            system += ("\n\n### repo map (auto-generated; use it to go straight to the right file, then outline/read_file)\n" + rm)
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


SUB_SYSTEM = ("You are a research sub-agent working for another agent. Investigate the task with your read-only tools "
              "(issue independent calls together in one step), then reply with a dense, factual answer: findings, file "
              "paths and line numbers, and anything uncertain. Your reply is all the caller sees; do not paste large "
              "file contents.\nProject root (working directory): {root}")
SUB_MAX_STEPS, SUB_MAX_REPLY = 15, 6000


def run_subagent(task: str, model: str) -> str:
    """A fresh, read-only agent loop. Returns only its final answer, so the caller's history stays small."""
    project = tools.PROJECT.get()
    allowed = tools.READONLY - {"delegate"}  # no recursion, no writes
    decls = [{"functionDeclarations": [d for d in tools.DECLARATIONS[0]["functionDeclarations"] if d["name"] in allowed]}]
    msgs = [{"role": "user", "parts": [{"text": task}]}]
    system = SUB_SYSTEM.format(root=project["root"] if project else ".")
    for _ in range(SUB_MAX_STEPS):
        content, _, _ = llm.generate(model, msgs, system, decls)
        msgs.append(content)
        calls = [p["functionCall"] for p in content["parts"] if "functionCall" in p]
        if not calls:
            ans = "".join(p.get("text", "") for p in content["parts"] if not p.get("thought")).strip()
            return ans[:SUB_MAX_REPLY] or "(sub-agent returned nothing)"
        outs = _run_parallel([c for c in calls if c["name"] in allowed], lambda e: None) if len(calls) > 1 else None
        res = []
        for k, c in enumerate(calls):
            if c["name"] not in allowed:
                out = f"ERROR: {c['name']} is not available to a sub-agent"
            elif outs is not None:
                out = outs[[x for x in calls if x["name"] in allowed].index(c)]
            else:
                out = tools.run(c["name"], c.get("args", {}))
            out = out[:config.HISTORY_TOOL_TRIM]
            res.append({"functionResponse": {"name": c["name"], "response": {"output": out}}})
        msgs.append({"role": "user", "parts": res})
    return "(sub-agent ran out of steps) " + "".join(p.get("text", "") for p in msgs[-2]["parts"] if p.get("text"))[:SUB_MAX_REPLY]


WRITERS = frozenset({"edit_file", "write_file", "bash"})  # anything that can change a file


def _norm_read(c: dict) -> tuple:
    a = c.get("args") or {}
    return (str(a.get("path", "")), int(a.get("offset") or 1), int(a.get("limit") or 2000))


def dup_read(messages: list, call: dict) -> str | None:
    """If this exact read_file was already answered earlier in the conversation, the answer is still in full above,
    and nothing that can change files ran since, return a short note instead of reading the file again."""
    if call.get("name") != "read_file":
        return None
    want = _norm_read(call)
    for i in range(len(messages) - 1, 0, -1):
        m = messages[i]
        if m["role"] != "model":
            continue
        calls = [p["functionCall"] for p in m["parts"] if "functionCall" in p]
        for k in reversed(range(len(calls))):
            c = calls[k]
            if c["name"] in WRITERS:
                return None
            if c["name"] == "read_file" and _norm_read(c) == want and i + 1 < len(messages):
                resp = [p for p in messages[i + 1]["parts"] if p.get("functionResponse")]
                out = resp[k]["functionResponse"]["response"].get("output") if k < len(resp) else None
                if isinstance(out, str) and not out.startswith(("[old result", "ERROR")) and "[trimmed in history" not in out:
                    return "(unchanged: you already read this exact range earlier in this conversation and it is still above. Use that.)"
                return None
    return None


def _run_parallel(calls: list[dict], emit, skip: dict | None = None) -> list[str]:
    """Several read-only calls in one step run at the same time; results keep the model's order."""
    import contextvars
    from concurrent.futures import ThreadPoolExecutor
    skip = skip or {}
    for c in calls:
        emit({"type": "tool_call", "name": c["name"], "args": c.get("args", {})})
    with ThreadPoolExecutor(min(len(calls), 6)) as ex:
        futs = [None if k in skip else ex.submit(contextvars.copy_context().run, tools.run, c["name"], c.get("args", {}))
                for k, c in enumerate(calls)]
        outs = [skip[k] if f is None else f.result() for k, f in enumerate(futs)]
    for c, out in zip(calls, outs):
        emit({"type": "tool_result", "name": c["name"], "output": out})
    return outs


Emit = Callable[[dict], None]


def _is_user_text(m: dict) -> bool:
    return m["role"] == "user" and any("text" in p for p in m["parts"])


KEEP_STEPS = 10  # messages kept verbatim when one long turn is itself the bulk of the context


def _chars(msgs: list) -> int:
    import json
    return len(json.dumps(msgs, ensure_ascii=False))


def stub_in_batches(s: dict) -> int:
    """Run stub_old_results only once STUB_BATCH more messages have aged out. Rewriting a message changes the
    prompt from that point on and so invalidates the provider's prompt cache; doing it every step would defeat it."""
    after = config.STUB_AFTER
    boundary = len(s["messages"]) - after
    done = s.get("stub_upto", 1)
    if after <= 0 or boundary - done < config.STUB_BATCH:
        return 0
    s["stub_upto"] = boundary
    return stub_old_results(s["messages"], after, start=done)


STUB_MIN = 300  # results shorter than this are not worth stubbing


def stub_old_results(messages: list, after: int | None = None, start: int = 1) -> int:
    """Shrink tool results older than the last `after` messages to a one-line stub (in place).

    Every result stays in the history and is re-sent on each later call, so a long task drags its early file
    reads along forever. The stub keeps the tool name, its arguments and the first line, so the model still
    knows what was done and can re-run the tool if it needs the full text. Returns how many were shrunk.
    """
    after = config.STUB_AFTER if after is None else after
    if after <= 0:
        return 0
    boundary = max(len(messages) - after, 0)
    # Results inside the age window always stay. Older ones also stay while the newest results together fit the
    # size budget, so a code-reading task does not forget (and re-read) files it opened a few steps ago.
    budget = config.STUB_KEEP * 4  # tokens -> characters
    spent = sum(len(p["functionResponse"]["response"].get("output") or "") for m in messages[boundary:]
                for p in m["parts"] if isinstance(p.get("functionResponse", {}).get("response", {}).get("output"), str))
    n = 0
    for i in range(boundary - 1, 0, -1):  # newest first, so the budget is spent on the most recent results
        m = messages[i]
        if m["role"] != "user" or messages[i - 1]["role"] != "model":
            continue
        calls = [p["functionCall"] for p in messages[i - 1]["parts"] if "functionCall" in p]
        resp = [p for p in m["parts"] if p.get("functionResponse")]
        for k in reversed(range(len(resp))):
            fr = resp[k]["functionResponse"]
            call = calls[k] if k < len(calls) else {}
            out = fr["response"].get("output")
            if not isinstance(out, str) or len(out) < STUB_MIN or out.startswith("[old result"):
                continue
            if spent + len(out) <= budget:
                spent += len(out)
                continue
            args = ", ".join(f"{a}={str(v)[:40]!r}" for a, v in (call.get("args") or {}).items())[:160]
            first = out.strip().splitlines()[0][:100] if out.strip() else ""
            fr["response"]["output"] = (f"[old result dropped, {len(out)} chars: {fr.get('name', '')}({args}); "
                                        f"starts: {first!r}. Re-run the tool if you need it again.]")
            n += 1
    return n


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
    s["stub_upto"] = 1
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
    tok = STOP.set(stop)  # Stop also cuts a model call that is in flight
    try:
        _run_turn(s, text, emit, stop, inbox)
    finally:
        STOP.reset(tok)


def _run_turn(s: dict, text: str, emit: Emit, stop: threading.Event | None = None, inbox=None) -> None:
    project = store.get_project(s["project"])
    tools.PROJECT.set(project)
    tools.MODEL.set(s["model"])
    if not str(text).strip():
        return  # an empty user turn is dropped by Gemini and leaves the request ending with a model turn
    s["messages"].append({"role": "user", "parts": [{"text": text}]})
    if s["title"] == "New session":
        s["title"] = store.title_case(text.strip().splitlines()[0][:50]) or "New session"
        s["title_auto"] = True  # a stand-in until auto_title() writes a proper one
    store.save(s)

    # The system prompt is built once per user turn and kept byte-identical across steps, so the provider can reuse
    # its cache of the unchanged prefix (system + tools + earlier history). It is rebuilt when a block changes.
    system, explore = None, 0
    for _ in range(config.MAX_STEPS):
        if stop and stop.is_set():
            emit({"type": "text", "text": "(stopped)"})
            break
        pt = s.get("prompt_tokens", 0)
        # Not again until the context has grown ~25% past its size right after the last compaction; otherwise a
        # context that stays big (large files in recent steps) would be re-summarized on every single step.
        if pt > config.COMPACT_AT and pt > s.get("compact_base", 0) * 1.25:
            compact(s, emit)
        stub_in_batches(s)
        if system is None:
            system = build_system(project, s)
        if s["messages"][-1]["role"] != "user":  # Gemini rejects requests ending with a model turn
            s["messages"].append({"role": "user", "parts": [{"text": "Continue."}]})
        content, tries = None, 0
        while content is None:
            try:
                content, usage, _ = llm.generate(s["model"], s["messages"], system, tools.DECLARATIONS, effort=s.get("effort", ""),
                                                 on_text=lambda d: emit({"type": "text_delta", "text": d}))
            except CallStopped:
                break
            except CallTimeout as e:  # a hung call is cut and tried again, then the run stops with a clear message
                tries += 1
                if tries > config.CALL_RETRIES:
                    raise
                emit({"type": "notice", "text": f"The model call was cut after {e.secs}s. Retrying ({tries}/{config.CALL_RETRIES})."})
        if content is None:
            emit({"type": "text", "text": "(stopped)"})
            break
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
        prior = s["messages"][:-1]  # history before this step's own calls
        dups = {k: d for k, c in enumerate(calls) if (d := dup_read(prior, c))}
        outs = _run_parallel(calls, emit, dups) if len(calls) > 1 and all(c["name"] in tools.READONLY for c in calls) else None
        for k, c in enumerate(calls):
            if outs is None:
                emit({"type": "tool_call", "name": c["name"], "args": c.get("args", {})})
                out = dups.get(k) or tools.run(c["name"], c.get("args", {}))
                emit({"type": "tool_result", "name": c["name"], "output": out})
            else:
                out = outs[k]
            # Store a history-trimmed version to prevent old tool outputs bloating context.
            # Full output was already emitted to the UI above.
            stored = out if len(out) <= config.HISTORY_TOOL_TRIM else out[:config.HISTORY_TOOL_TRIM] + f"\n... [trimmed in history; {len(out)} chars total]"
            results.append({"functionResponse": {"name": c["name"], "response": {"output": stored}}})
            if c["name"] in BLOCK_WRITERS:
                system = None  # block edits apply on the next step
        explore = explore + 1 if all(c["name"] in tools.READONLY and c["name"] != "delegate" for c in calls) else 0
        if config.EXPLORE_LIMIT and explore and explore % config.EXPLORE_LIMIT == 0:
            results[-1]["functionResponse"]["response"]["output"] += (
                f"\n\n[harness note: {explore} read-only steps in a row. Stop exploring: answer from what you have, "
                "or hand the rest of the digging to delegate.]")
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

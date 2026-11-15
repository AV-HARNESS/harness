"""R-LOOP: the agent loop."""
import threading

import pytest

from harness import agent, config, store, tools
from conftest import covers
from helpers import call, gem, text


def run(session, msg, fake, *responses, stop=None):
    fake.push(*[(200, r) for r in responses])
    events = []
    agent.run_turn(session, msg, events.append, stop)
    return events


@covers("R-LOOP-01", "R-LOOP-08")
def test_tool_loop_runs_tools_and_finishes(session, fake, project):
    ev = run(session, "make a file", fake,
             gem(call("write_file", path="a.txt", content="hi")),
             gem(text("done")))
    assert (project["root"] and open(f"{project['root']}/a.txt").read()) == "hi"
    assert [e["type"] for e in ev if e["type"] != "usage"] == ["tool_call", "tool_result", "text"]
    assert len(fake.requests) == 2
    # second request carries the tool result back to the model
    last = fake.requests[1]["body"]["contents"][-1]
    assert "functionResponse" in last["parts"][0]
    saved = store.load(session["id"])
    assert saved["title"] == "make a file" and saved["messages"][0]["parts"][0]["text"] == "make a file"


@covers("R-LOOP-01")
def test_parallel_tool_calls_all_run(session, fake, project):
    run(session, "go", fake, gem(call("write_file", path="1.txt", content="1"), call("write_file", path="2.txt", content="2")),
        gem(text("ok")))
    assert open(f"{project['root']}/1.txt").read() == "1" and open(f"{project['root']}/2.txt").read() == "2"
    resp = fake.requests[1]["body"]["contents"][-1]["parts"]
    assert len(resp) == 2


@covers("R-LOOP-02")
def test_max_steps(session, fake, monkeypatch):
    monkeypatch.setattr(config, "MAX_STEPS", 3)
    ev = run(session, "loop", fake, *[gem(call("list_dir"))] * 5)
    assert len(fake.requests) == 3
    assert "stopped after 3 steps" in ev[-1]["text"]


@covers("R-LOOP-03")
def test_stop_before_next_model_call(session, fake):
    stop = threading.Event()
    stop.set()
    ev = run(session, "hi", fake, gem(text("never")), stop=stop)
    assert fake.requests == [] and ev[-1]["text"] == "(stopped)"


@covers("R-LOOP-04")
def test_history_trims_tool_output_but_ui_gets_full(session, fake, project, monkeypatch):
    monkeypatch.setattr(config, "HISTORY_TOOL_TRIM", 50)
    open(f"{project['root']}/big.txt", "w").write("x" * 500)
    ev = run(session, "read", fake, gem(call("read_file", path="big.txt")), gem(text("ok")))
    full = next(e for e in ev if e["type"] == "tool_result")["output"]
    stored = session["messages"][2]["parts"][0]["functionResponse"]["response"]["output"]
    assert len(full) > 500 and "trimmed in history" in stored and len(stored) < 200


@covers("R-LOOP-05")
def test_system_prompt_has_project_blocks_and_instructions_and_updates_each_step(session, fake, project):
    store.write_block(project["id"], "goals", "SHIP-IT")
    session["instructions"] = "be terse"
    run(session, "go", fake, gem(call("write_block", name="goals", content="NEW-GOAL")), gem(text("ok")))
    s1 = fake.requests[0]["body"]["systemInstruction"]["parts"][0]["text"]
    s2 = fake.requests[1]["body"]["systemInstruction"]["parts"][0]["text"]
    assert project["name"] in s1 and project["root"] in s1 and "SHIP-IT" in s1 and "be terse" in s1
    assert "NEW-GOAL" in s2 and "SHIP-IT" not in s2  # block edit applied on the very next step


@covers("R-LOOP-05")
def test_block_budget_caps(project, session):
    store.write_block(project["id"], "huge", "a" * 20000)
    sysmsg = agent.build_system(project, session)
    assert sysmsg.count("a") <= agent.MAX_BLOCK + 100
    for i in range(8):
        store.write_block(project["id"], f"b{i}", "z" * 7000)
    assert "omitted: context budget" in agent.build_system(project, session)


@covers("R-LOOP-06")
def test_thoughts_hidden_but_kept_verbatim(session, fake):
    sig = {"text": "secret reasoning", "thought": True, "thoughtSignature": "SIG123"}
    ev = run(session, "hi", fake, gem(sig, text("answer")))
    assert [e["text"] for e in ev if e["type"] == "text"] == ["answer"]
    assert session["messages"][1]["parts"][0] == sig


@covers("R-LOOP-07")
def test_compaction_replaces_old_turns_keeps_recent(session, fake, monkeypatch):
    monkeypatch.setattr(config, "KEEP_TURNS", 2)
    for i in range(5):
        session["messages"] += [{"role": "user", "parts": [{"text": f"q{i}"}]}, {"role": "model", "parts": [{"text": f"a{i}"}]}]
    fake.push((200, gem(text("SUMMARY-TEXT"))))
    events = []
    assert agent.compact(session, events.append)
    msgs = session["messages"]
    assert "SUMMARY-TEXT" in msgs[0]["parts"][0]["text"]
    kept = [m["parts"][0]["text"] for m in msgs[2:] if m["role"] == "user"]
    assert kept == ["q3", "q4"] and events[0]["type"] == "compacted"


@covers("R-LOOP-07")
def test_compaction_skipped_when_short_and_triggered_by_tokens(session, fake, monkeypatch):
    assert not agent.compact(session)
    monkeypatch.setattr(config, "KEEP_TURNS", 1)
    monkeypatch.setattr(config, "COMPACT_AT", 100)
    session["messages"] = [{"role": "user", "parts": [{"text": "old"}]}, {"role": "model", "parts": [{"text": "x"}]}]
    session["prompt_tokens"] = 5000
    run(session, "new", fake, gem(text("SUM")), gem(text("final")))
    assert "Summary of earlier conversation" in str(session["messages"][0])


@covers("R-LOOP-09")
def test_error_keeps_user_message(session, fake):
    fake.push((400, "bad request"))
    with pytest.raises(RuntimeError, match="HTTP 400"):
        agent.run_turn(session, "important question", lambda e: None)
    assert store.load(session["id"])["messages"][0]["parts"][0]["text"] == "important question"


@covers("R-LOOP-01")
def test_unknown_tool_does_not_crash_loop(session, fake):
    ev = run(session, "go", fake, gem(call("nope")), gem(text("ok")))
    assert "unknown tool" in next(e for e in ev if e["type"] == "tool_result")["output"]


@covers("R-LOOP-02")
def test_default_step_cap_is_200():
    import os, subprocess, sys
    env = {k: v for k, v in os.environ.items() if k != "HARNESS_MAX_STEPS"}
    env["PYTHONPATH"] = str(__import__("pathlib").Path(__file__).parent.parent / "src")
    out = subprocess.run([sys.executable, "-c", "from harness import config; print(config.MAX_STEPS)"],
                         capture_output=True, text=True, env=env).stdout.strip()
    assert out == "200"


@covers("R-LOOP-10")
def test_queued_message_is_delivered_with_tool_results_and_agent_continues(session, fake, project):
    import queue
    inbox = queue.Queue()
    inbox.put("also add tests")
    fake.push((200, gem(call("list_dir"))), (200, gem(text("done"))))
    events = []
    agent.run_turn(session, "go", events.append, None, inbox)
    assert {"type": "user", "text": "also add tests"} in events
    delivered = session["messages"][2]["parts"]
    assert len(delivered) == 1 and "also add tests" in delivered[0]["functionResponse"]["response"]["output"]


@covers("R-LOOP-10")
def test_queued_message_after_final_answer_starts_another_step(session, fake):
    import queue
    inbox = queue.Queue()
    inbox.put("one more thing")
    fake.push((200, gem(text("first answer"))), (200, gem(text("second answer"))))
    events = []
    agent.run_turn(session, "go", events.append, None, inbox)
    assert len(fake.requests) == 2 and events[-2]["type"] == "user" or any(e.get("text") == "second answer" for e in events)
    assert session["messages"][2] == {"role": "user", "parts": [{"text": "one more thing"}]}


@covers("R-LOOP-10")
def test_stopped_run_hands_queued_messages_back(session, fake):
    import queue
    inbox = queue.Queue()
    inbox.put("later")
    stop = threading.Event()
    fake.push((200, gem(text("answer"))))
    stop.set()
    agent.run_turn(session, "go", lambda e: None, stop, inbox)
    assert agent._drain(inbox) == ["later"]
@covers("R-LOOP-11")
def test_agents_md_is_loaded_or_requested(session, project):
    from pathlib import Path
    root = Path(project["root"])
    assert "create AGENTS.md" in agent.build_system(project, session)
    (root / "CLAUDE.md").write_text("run: make test", encoding="utf-8")
    sys_prompt = agent.build_system(project, session)
    assert "run: make test" in sys_prompt and "create AGENTS.md" not in sys_prompt and "update it with edit_file" in sys_prompt
    (root / "AGENTS.md").write_text("use pytest -q", encoding="utf-8")
    sys_prompt = agent.build_system(project, session)
    assert "use pytest -q" in sys_prompt and "run: make test" not in sys_prompt


def _long_turn(session, steps=12, size=4000):
    session["messages"] = [{"role": "user", "parts": [{"text": "big job"}]}]
    for i in range(steps):
        session["messages"] += [
            {"role": "model", "parts": [{"functionCall": {"name": "list_dir", "args": {"path": "x" * size}}}]},
            {"role": "user", "parts": [{"functionResponse": {"name": "list_dir", "response": {"output": f"r{i}"}}}]}]


@covers("R-LOOP-12")
def test_compaction_cuts_inside_one_long_turn_without_splitting_calls(session, fake, monkeypatch):
    monkeypatch.setattr(config, "COMPACT_AT", 1000)
    _long_turn(session)
    fake.push((200, gem(text("SUM"))))
    assert agent.compact(session)
    msgs = session["messages"]
    assert "SUM" in msgs[0]["parts"][0]["text"] and msgs[1]["role"] == "model" and "functionCall" in msgs[1]["parts"][0]
    assert len(msgs) <= agent.KEEP_STEPS + 2
    assert [m["role"] for m in msgs] == ["user"] + ["model", "user"] * ((len(msgs) - 1) // 2)  # alternation intact


@covers("R-LOOP-12")
def test_no_compaction_on_every_step_when_context_stays_big(session, fake, monkeypatch):
    monkeypatch.setattr(config, "COMPACT_AT", 1000)
    summaries = []
    real = agent.compact
    monkeypatch.setattr(agent, "compact", lambda s, emit=lambda e: None: summaries.append(1) or real(s, emit))
    _long_turn(session, steps=2)
    session["prompt_tokens"] = 5000
    # every reply reports a still-huge prompt (5000), as when recent steps hold big files
    big = lambda parts: {**gem(*parts), "usageMetadata": {"promptTokenCount": 5000}}
    fake.push((200, gem(text("SUM"))))                       # the one compaction summary
    for _ in range(4):
        fake.push((200, big([call("list_dir")])))
    fake.push((200, big([text("done")])))
    agent.run_turn(session, "go", lambda e: None, None)
    assert len(summaries) == 1


@covers("R-LOOP-13")
def test_blank_messages_never_reach_the_model(session, fake):
    events = run(session, "   ", fake)
    assert fake.requests == [] and session["messages"] == [] and events == []
    import queue
    q = queue.Queue(); q.put(""); q.put("  "); q.put("real")
    assert agent._drain(q) == ["real"]


@covers("R-LOOP-13")
def test_request_never_ends_with_a_model_turn(session, fake):
    session["messages"] = [{"role": "user", "parts": [{"text": "a"}]}, {"role": "model", "parts": [{"text": "b"}]}]
    fake.push((200, gem(text("ok"))))
    s = session
    s["messages"].append({"role": "model", "parts": [{"text": "c"}]})
    # run_turn always adds the user text first; simulate the guard by corrupting after append via a stop-free call
    agent.run_turn(s, "hi", lambda e: None)
    assert fake.requests[0]["body"]["contents"][-1]["role"] == "user"


@covers("R-LOOP-14")
def test_blocks_are_picked_per_prompt_and_followups_keep_them(project, session):
    store.write_block(project["id"], "billing-api", "Stripe webhook handling and invoice retries. " + "x " * 2500)
    store.write_block(project["id"], "ui-style", "Teal theme, sidebar layout and fonts. " + "y " * 2500)
    store.write_block(project["id"], "deploy", "Release checklist and rollback steps. " + "z " * 2500)
    ask = lambda text: session["messages"].append({"role": "user", "parts": [{"text": text}]})
    ask("why do invoice retries fail on the stripe webhook?")
    sysmsg = agent.build_system(project, session)
    assert "### block: billing-api" in sysmsg and "### block: ui-style" not in sysmsg
    assert "other blocks (not loaded" in sysmsg and "- ui-style: Teal theme" in sysmsg
    ask("yes, do that")  # no usable words: previous selection stays
    assert "### block: billing-api" in agent.build_system(project, session)
    ask("update the sidebar layout")
    s2 = agent.build_system(project, session)
    assert "### block: ui-style" in s2 and "### block: billing-api" not in s2


@covers("R-LOOP-14")
def test_small_block_sets_load_whole(project, session):
    store.write_block(project["id"], "a", "alpha notes")
    store.write_block(project["id"], "b", "beta notes")
    session["messages"].append({"role": "user", "parts": [{"text": "something unrelated"}]})
    sysmsg = agent.build_system(project, session)
    assert "alpha notes" in sysmsg and "beta notes" in sysmsg and "not loaded" not in sysmsg


@covers("R-LOOP-15")
def test_auto_title_is_short_and_cleaned(session, fake):
    session["messages"] = [{"role": "user", "parts": [{"text": "can you review my invoice retry logic for stripe webhooks please"}]},
                           {"role": "model", "parts": [{"text": "Sure, here is what I found..."}]}]
    fake.push((200, gem(text('"Stripe Webhook Retry Review."\nextra'))))
    assert agent.auto_title(session) == "Stripe Webhook Retry Review"
    assert fake.requests[-1]["body"]["generationConfig"]  # asked at low effort
    session["messages"] = session["messages"][:1]
    assert agent.auto_title(session) is None  # no answer yet: nothing to title



@covers("R-LOOP-16")
def test_text_arrives_as_deltas_then_the_final_text_event(session, fake, monkeypatch):
    import json
    from harness import config
    monkeypatch.setattr(config, "STREAM", True)
    ev = [{"candidates": [{"content": {"parts": [{"text": "Hello "}]}}]},
          {"candidates": [{"content": {"parts": [{"text": "world"}]}}], "usageMetadata": {"promptTokenCount": 5}}]
    fake.push((200, "".join("data: " + json.dumps(e) + "\n\n" for e in ev), "text/event-stream"))
    events = []
    agent.run_turn(session, "hi", events.append)
    kinds = [(e["type"], e.get("text")) for e in events if e["type"] in ("text_delta", "text")]
    assert kinds == [("text_delta", "Hello "), ("text_delta", "world"), ("text", "Hello world")]
    assert session["messages"][-1]["parts"] == [{"text": "Hello world"}]

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
    assert "functionResponse" in delivered[0] and delivered[-1] == {"text": "also add tests"}


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

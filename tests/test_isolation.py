"""R-SRV-16: sessions never mix - not across sessions, projects or concurrent runs."""
import json
import re
import threading
import time
from time import sleep as real_sleep  # conftest turns time.sleep into a no-op; the fake model needs a real pause
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

from conftest import covers
from harness import config, server, store


class EchoModel:
    """OpenAI-compatible fake: asks to write <MARK>.txt (marker taken from the user's text), then answers 'done <MARK>'."""

    def __init__(self, delay=0.05):
        outer = self
        self.delay = delay
        self.gate = None

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                real_sleep(outer.delay)
                if outer.gate is not None:
                    outer.gate.wait(10)  # hold the reply until the test lets go (to inspect a run in progress)
                msgs = body["messages"]
                marks = re.findall(r"MARK-\w+", json.dumps([m for m in msgs if m["role"] != "system"]))
                mark = marks[-1] if marks else "NONE"
                seen = sorted(set(marks))
                if msgs[-1]["role"] == "tool":
                    msg = {"content": f"done {mark} saw={','.join(seen)}"}
                else:
                    msg = {"content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {
                        "name": "write_file", "arguments": json.dumps({"path": f"{mark}.txt", "content": mark})}}]}
                data = json.dumps({"choices": [{"message": msg}], "usage": {"prompt_tokens": 5}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def client():
    store.ensure_default()
    return TestClient(server.app)


@pytest.fixture
def echo(monkeypatch):
    m = EchoModel()
    monkeypatch.setitem(config.MODELS, "echo", {"provider": "openai", "model": "e", "base_url": m.url, "api_key_env": ""})
    monkeypatch.setattr(config, "DEFAULT_MODEL", "echo")
    yield m
    m.close()


def _chat(sid, text):
    with TestClient(server.app) as c:
        return c.post(f"/api/sessions/{sid}/chat", json={"text": text, "model": "echo"}).text


@covers("R-SRV-16")
def test_concurrent_sessions_in_different_projects_never_mix(client, echo, tmp_path):
    projects = []
    for n in range(3):
        root = tmp_path / f"root{n}"
        root.mkdir()
        p = store.create_project(f"proj{n}", str(root))
        projects.append((p, [store.create(p["id"], model="echo") for _ in range(2)]))
    jobs = [(s["id"], f"MARK-p{pi}s{si}") for pi, (p, ss) in enumerate(projects) for si, s in enumerate(ss)]
    with ThreadPoolExecutor(len(jobs)) as ex:
        bodies = list(ex.map(lambda j: _chat(*j), jobs))
    assert all("done" in b for b in bodies)
    for (sid, mark), body in zip(jobs, bodies):
        assert f"done {mark} saw={mark}" in body  # the model only ever saw its own session's text
        saved = json.dumps(store.load(sid)["messages"])
        others = {m for _, m in jobs if m != mark}
        assert mark in saved and not any(o in saved for o in others)
    for pi, (p, ss) in enumerate(projects):  # files landed only in their own project's folder
        files = sorted(f.name for f in __import__("pathlib").Path(p["root"]).iterdir())
        assert files == sorted(f"MARK-p{pi}s{si}.txt" for si in range(2))


@covers("R-SRV-16")
def test_running_session_is_visible_and_edits_are_not_lost(client, echo, session):
    echo.gate = threading.Event()
    sid = session["id"]
    t = threading.Thread(target=_chat, args=(sid, "MARK-live"))
    t.start()
    deadline = time.time() + 5
    while sid not in server._busy and time.time() < deadline:
        time.sleep(0.01)
    d = client.get(f"/api/sessions/{sid}").json()
    assert d["running"] is True and "MARK-live" in json.dumps(d["messages"])  # a refresh/second tab sees the live run
    assert client.patch(f"/api/sessions/{sid}", json={"title": "renamed while running"}).status_code == 200
    assert client.post(f"/api/sessions/{sid}/compact").status_code == 409
    assert client.delete(f"/api/sessions/{sid}").status_code == 409
    echo.gate.set()
    t.join()
    final = client.get(f"/api/sessions/{sid}").json()
    assert final["running"] is False and final["title"] == "renamed while running"
    assert store.load(sid)["title"] == "renamed while running"  # persisted, and the run's own saves did not undo it
    assert "MARK-live" in json.dumps(final["messages"])


@covers("R-SRV-16")
def test_same_project_two_sessions_have_separate_histories(client, echo, project):
    a, b = store.create(project["id"], model="echo"), store.create(project["id"], model="echo")
    with ThreadPoolExecutor(2) as ex:
        list(ex.map(lambda j: _chat(*j), [(a["id"], "MARK-aaa"), (b["id"], "MARK-bbb")]))
    ja, jb = json.dumps(store.load(a["id"])["messages"]), json.dumps(store.load(b["id"])["messages"])
    assert "MARK-aaa" in ja and "MARK-bbb" not in ja and "MARK-bbb" in jb and "MARK-aaa" not in jb

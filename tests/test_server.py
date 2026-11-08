"""R-SRV: web API."""
import json
import threading

import pytest
from fastapi.testclient import TestClient

from harness import server, store
from conftest import covers
from helpers import call, gem, text


@pytest.fixture
def client():
    store.ensure_default()
    return TestClient(server.app)


def sse(resp):
    return [json.loads(l[6:]) for l in resp.text.splitlines() if l.startswith("data: ")]


@covers("R-SRV-02")
def test_project_crud(client, tmp_path):
    r = client.post("/api/projects", json={"name": "Alpha", "root": str(tmp_path)})
    assert r.status_code == 200 and r.json()["id"] == "alpha"
    assert client.post("/api/projects", json={}).status_code == 400
    assert client.post("/api/projects", json={"name": "x", "root": "/definitely/not/here"}).status_code == 400
    assert client.patch("/api/projects/alpha", json={"name": "Beta"}).json()["name"] == "Beta"
    assert client.patch("/api/projects/alpha", json={"root": "/nope/nope"}).status_code == 400
    assert client.patch("/api/projects/ghost", json={"name": "x"}).status_code == 404
    assert "alpha" in [p["id"] for p in client.get("/api/projects").json()]
    assert client.delete("/api/projects/alpha").json() == {"ok": True}
    assert client.delete("/api/projects/alpha").status_code == 404


@covers("R-SRV-06")
def test_delete_project_keeps_root(client, tmp_path):
    (tmp_path / "keep.txt").write_text("x")
    client.post("/api/projects", json={"name": "Del", "root": str(tmp_path)})
    client.delete("/api/projects/del")
    assert (tmp_path / "keep.txt").read_text() == "x"


@covers("R-SRV-02")
def test_blocks_api(client, project):
    pid = project["id"]
    assert client.put(f"/api/projects/{pid}/blocks/notes", json={"content": "hello"}).json() == {"ok": True}
    assert client.get(f"/api/projects/{pid}/blocks").json() == [{"name": "notes", "size": 5, "content": "hello"}]
    assert client.put(f"/api/projects/{pid}/blocks/bad name", json={"content": "x"}).status_code == 400
    assert client.put("/api/projects/ghost/blocks/n", json={"content": "x"}).status_code == 404
    client.delete(f"/api/projects/{pid}/blocks/notes")
    assert client.get(f"/api/projects/{pid}/blocks").json() == []


@covers("R-SRV-02")
def test_session_api(client, project):
    pid = project["id"]
    s = client.post("/api/sessions", json={"project": pid, "title": "T", "model": "pro"}).json()
    assert s["title"] == "T" and s["model"] == "pro"
    assert client.get(f"/api/sessions?project={pid}").json()[0]["id"] == s["id"]
    p = client.patch(f"/api/sessions/{s['id']}", json={"title": "New", "model": "bogus", "instructions": "i"}).json()
    assert p["title"] == "New" and p["model"] == "pro" and p["instructions"] == "i"  # unknown model ignored
    assert client.get(f"/api/sessions/{s['id']}").json()["title"] == "New"
    assert client.get("/api/sessions/zzzzzz").status_code == 404
    assert client.get("/api/sessions?project=ghost").status_code == 404
    assert client.post("/api/sessions", json={"project": "ghost"}).status_code == 404
    assert client.post(f"/api/sessions/{s['id']}/compact").json() == {"compacted": False}
    assert client.post("/api/sessions/nope/stop").json() == {"ok": True}
    client.delete(f"/api/sessions/{s['id']}")
    assert client.get(f"/api/sessions/{s['id']}").status_code == 404


@covers("R-SRV-03")
def test_chat_streams_events(client, session, fake):
    fake.push((200, gem(call("bash", command="echo hi"))), (200, gem(text("all done"))))
    r = client.post(f"/api/sessions/{session['id']}/chat", json={"text": "run it"})
    assert r.headers["content-type"].startswith("text/event-stream")
    ev = sse(r)
    assert [e["type"] for e in ev if e["type"] != "usage"] == ["tool_call", "tool_result", "text", "done"]
    assert "hi" in ev[2]["output"] if ev[2]["type"] == "tool_result" else True
    assert store.load(session["id"])["title"] == "run it"


@covers("R-SRV-03")
def test_chat_model_switch_and_error_event(client, session, fake):
    fake.push((500, "kaput"), (500, "kaput"), (500, "kaput"), (500, "kaput"), (500, "kaput"))
    ev = sse(client.post(f"/api/sessions/{session['id']}/chat", json={"text": "hi", "model": "fake"}))
    assert ev[-2]["type"] == "error" and "HTTP 500" in ev[-2]["error"] and ev[-1]["type"] == "done"
    assert server._busy == set()  # released after failure


@covers("R-SRV-04")
def test_busy_session_gets_409(client, session):
    server._busy.add(session["id"])
    assert client.post(f"/api/sessions/{session['id']}/chat", json={"text": "x"}).status_code == 409
    assert client.post("/api/sessions/nope/chat", json={"text": "x"}).status_code == 404


@covers("R-SRV-04")
def test_stop_endpoint_sets_event(client, session):
    ev = server._stops[session["id"]] = threading.Event()
    client.post(f"/api/sessions/{session['id']}/stop")
    assert ev.is_set()


@covers("R-SRV-05")
def test_config_endpoint_has_no_secrets(client, monkeypatch):
    from harness import config
    monkeypatch.setattr(config, "ACCESS_TOKEN", "SUPER-SECRET-TOKEN")
    monkeypatch.setattr(config, "CLIENT_SECRET", "SUPER-SECRET-CLIENT")
    monkeypatch.setenv("GEMINI_API_KEY", "SUPER-SECRET-KEY")
    r = client.get("/api/config")
    d = r.json()
    assert d["logged_in"] is True and d["has_api_key"] is True and d["default_model"] == "flash"
    assert d["models"]["flash"].startswith("gemini:")
    assert "SUPER-SECRET" not in r.text


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "<html" in r.text.lower()


@covers("R-SRV-01")
def test_serve_binds_localhost_only(monkeypatch):
    import uvicorn
    seen = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(kw))
    monkeypatch.setattr("webbrowser.open", lambda u: None)
    server.serve(1234)
    assert seen["host"] == "127.0.0.1" and seen["port"] == 1234
    assert server.serve.__code__.co_names.count("0.0.0.0") == 0


@covers("R-SRV-07")
def test_fs_lists_folders_and_files(client, tmp_path):
    tmp_path = tmp_path / "browse"
    tmp_path.mkdir()
    (tmp_path / "b_dir").mkdir()
    (tmp_path / "A_dir").mkdir()
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "x.py").write_text("1")
    (tmp_path / ".env").write_text("SECRET=1")
    (tmp_path / "key.pem").write_text("k")
    d = client.get("/api/fs", params={"path": str(tmp_path)}).json()
    assert d["path"] == str(tmp_path.resolve()) and d["parent"] == str(tmp_path.resolve().parent)
    assert [e["name"] for e in d["entries"]] == ["A_dir", "b_dir"] and all(e["dir"] for e in d["entries"])
    d = client.get("/api/fs", params={"path": str(tmp_path), "files": "1"}).json()
    assert [e["name"] for e in d["entries"]] == ["A_dir", "b_dir", "x.py"]  # no hidden, no .env, no *.pem
    assert client.get("/api/fs", params={"path": str(tmp_path / "nope")}).status_code == 400
    assert client.get("/api/fs", params={"path": str(tmp_path / "x.py")}).status_code == 400
    assert client.get("/api/fs").json()["path"]  # defaults to the home folder


@covers("R-SRV-08")
def test_fs_project_scope_stays_inside_root(client, project):
    from pathlib import Path
    root = Path(project["root"])
    (root / "sub").mkdir()
    (root / "sub" / "f.txt").write_text("x")
    d = client.get("/api/fs", params={"project": project["id"], "files": "1"}).json()
    assert d["path"] == str(root.resolve()) and d["parent"] is None and d["entries"][0]["rel"] == "sub"
    d = client.get("/api/fs", params={"project": project["id"], "files": "1", "path": str(root / "sub")}).json()
    assert [{k: v for k, v in e.items() if k != "mtime"} for e in d["entries"]] == [{"name": "f.txt", "dir": False, "rel": "sub/f.txt"}]
    assert d["entries"][0]["mtime"] > 0 and d["parent"] == str(root.resolve())
    assert client.get("/api/fs", params={"project": project["id"], "path": str(root.parent)}).status_code == 400
    assert client.get("/api/fs", params={"project": project["id"], "path": str(root / "..")}).status_code == 400
    assert client.get("/api/fs", params={"project": "ghost"}).status_code == 404


@covers("R-SRV-02")
def test_linking_a_folder_changes_where_the_agent_works(client, session, fake, tmp_path):
    new_root = tmp_path / "newroot"
    new_root.mkdir()
    assert client.patch(f"/api/projects/{session['project']}", json={"root": str(new_root)}).status_code == 200
    fake.push((200, gem(call("write_file", path="made.txt", content="hi"))), (200, gem(text("done"))))
    client.post(f"/api/sessions/{session['id']}/chat", json={"text": "go"})
    assert (new_root / "made.txt").read_text() == "hi"


@covers("R-UI-01")
def test_ui_has_composer_controls(client):
    html = client.get("/").text
    for el in ('id="plus"', 'id="menu"', 'id="m-folder"', 'id="m-attach"', 'id="m-instr"', 'id="model"', 'id="mic"',
               'id="voice"', 'id="send"', 'id="fdlg"', 'id="froot"', 'id="places"', 'id="crumbs"', 'id="fback"', 'id="dx"', 'id="dok"', 'id="wave"', 'id="projbtn"', "className = 'steps live'", 'pend.shift()', 'id="projmenu"', 'id="i-mic"', 'id="i-wave"', "/api/fs", "SpeechRecognition", "speechSynthesis"):
        assert el in html, el
    assert html.index('id="plus"') < html.index('id="model"') < html.index('id="mic"') < html.index('id="send"')  # layout order


@covers("R-SRV-09")
def test_places_for_the_folder_picker(client, tmp_path, monkeypatch):
    from pathlib import Path
    home = tmp_path / "fakehome"
    for n in ("Desktop", "Documents"):
        (home / n).mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    (tmp_path / "work").mkdir()
    client.post("/api/projects", json={"name": "W", "root": str(tmp_path / "work")})
    d = client.get("/api/places").json()
    names = [f["name"] for f in d["favorites"]]
    assert names == ["Home", "Desktop", "Documents"]  # Downloads missing -> not listed
    assert d["favorites"][1]["path"] == str(home / "Desktop")
    assert {"name": "work", "path": str((tmp_path / "work").resolve()), "project": "W"} in d["projects"]
    assert d["sep"] and isinstance(d["drives"], list)


@covers("R-SRV-10")
def test_queue_endpoint(client, session):
    import queue
    sid = session["id"]
    assert client.post(f"/api/sessions/{sid}/queue", json={"text": "x"}).status_code == 409
    server._busy.add(sid)
    server._inboxes[sid] = queue.Queue()
    assert client.post(f"/api/sessions/{sid}/queue", json={"text": "next"}).status_code == 200
    assert server._inboxes[sid].get_nowait() == "next"
    server._busy.discard(sid)


@covers("R-SRV-11", "R-SRV-12")
def test_models_endpoint_lists_confirms_and_filters(client, fake, monkeypatch):
    from harness import config, discover
    monkeypatch.setattr(config, "MODELS", {
        "oa": {"provider": "openai", "model": "x", "base_url": fake.url, "api_key_env": ""},
        "bad": {"provider": "openai", "model": "y", "base_url": fake.url + "/other", "api_key_env": "NOPE_KEY_X"},
        "vtx": {"provider": "gemini", "model": "gemini-x"}})
    for k in ("NOPE_KEY_X", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(config, "PROJECT", "p")
    monkeypatch.setattr(config, "MODEL_ALLOW", ["gpt-*"])
    monkeypatch.setattr(discover, "_start_probes", lambda: None)  # probe synchronously below
    monkeypatch.setitem(discover._cache, "at", 0.0)
    status = {"gpt-a": "ok", "gpt-b": "denied", "gpt-c": "unknown", "gpt-d": "ok"}
    monkeypatch.setattr(discover, "probe", lambda spec, mid: status[mid])
    fake.push((200, {"data": [{"id": n} for n in ("gpt-d", "gpt-c", "gpt-b", "gpt-a", "text-embedding-3", "whisper-1", "other-model")]}))

    first = client.get("/api/models?refresh=true").json()
    assert first["pending"] == 4 and all(not g["models"] for g in first["groups"])  # nothing confirmed yet
    discover.probe_pending()
    d = client.get("/api/models").json()
    by = {g["source"]: g for g in d["groups"]}
    assert d["pending"] == 0
    assert [(m["alias"], m["status"]) for m in by["oa"]["models"]] == [("oa/gpt-a", "ok"), ("oa/gpt-c", "unverified"), ("oa/gpt-d", "ok")]
    assert "NOPE_KEY_X" in by["bad"]["error"] and by["bad"]["models"] == [] and "vtx" not in by
    saved = (config.HOME / "model_access.json").read_text()
    assert "gpt-a" in saved and "NOPE" not in saved  # cached on disk, key material never stored


@covers("R-SRV-12")
def test_probe_classifies_by_http_status(fake):
    from harness import discover
    spec = {"provider": "openai", "model": "m", "base_url": fake.url, "api_key_env": ""}
    ok = {"choices": [{"message": {"content": "ok"}}], "usage": {}}
    fake.push((200, ok), (403, "no"), (404, "no"), (401, "no"), (429, "slow"), (400, "bad param"))
    assert [discover.probe(spec, "m") for _ in range(6)] == ["ok", "denied", "denied", "denied", "unknown", "unknown"]

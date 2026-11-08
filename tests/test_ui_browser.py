"""R-UI-05, end to end in a real browser: tabs and session switching never mix chats.

Needs playwright + a Chrome/Chromium; skipped otherwise (so it does not run in CI).
"""
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import requests

from conftest import covers
from test_isolation import EchoModel, real_sleep

playwright = pytest.importorskip("playwright.sync_api")
CHROME = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
pytestmark = pytest.mark.skipif(not CHROME, reason="no Chrome/Chromium available")
ROOT = Path(__file__).parent.parent


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def app(tmp_path):
    model = EchoModel(delay=0.0)
    home = tmp_path / "hh"
    home.mkdir()
    (home / "models.json").write_text(json.dumps({"echo": {"provider": "openai", "model": "e", "base_url": model.url, "api_key_env": ""}}))
    port = _free_port()
    env = {**os.environ, "HARNESS_HOME": str(home), "HARNESS_MODELS_FILE": str(home / "models.json"), "HARNESS_DEFAULT_MODEL": "echo", "PYTHONPATH": str(ROOT / "src"),
           "GEMINI_API_KEY": "", "HARNESS_REDACT": "0"}
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "harness.server:app", "--port", str(port), "--log-level", "warning"],
                            env=env, cwd=tmp_path, stdout=subprocess.DEVNULL, stderr=open(tmp_path / "server.err", "w"))
    base = f"http://127.0.0.1:{port}"
    for _ in range(300):
        try:
            requests.get(base + "/api/config", timeout=1)
            break
        except Exception:
            if proc.poll() is not None:
                pytest.fail("server died: " + (tmp_path / "server.err").read_text()[-1500:])
            real_sleep(0.1)
    api = lambda m, path, **kw: requests.request(m, base + path, timeout=10, **kw).json()
    projects = []
    for n in range(2):
        root = tmp_path / f"root{n}"
        root.mkdir()
        p = api("POST", "/api/projects", json={"name": f"proj{n}", "root": str(root)})
        s = api("POST", "/api/sessions", json={"project": p["id"], "model": "echo"})
        projects.append((p, s))
    yield base, model, projects, api
    proc.terminate()
    model.close()


def _msgs(page):
    return page.inner_text("#msgs")


def _wait_text(page, text, timeout=8000):
    page.wait_for_function("t => document.querySelector('#msgs').innerText.includes(t)", arg=text, timeout=timeout)


@covers("R-UI-05")
def test_tabs_and_session_switching_never_mix_chats(app):
    base, model, projects, api = app
    (p0, s0), (p1, s1) = projects
    # give each session its own conversation through the API
    for s, mark in ((s0, "MARK-zero"), (s1, "MARK-one")):
        r = requests.post(f"{base}/api/sessions/{s['id']}/chat", json={"text": mark, "model": "echo"}, timeout=20)
        assert '"type": "error"' not in r.text, r.text
    with playwright.sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=CHROME, args=["--no-sandbox"])
        ctx = browser.new_context()  # one context = shared localStorage, like tabs in one window
        tab0, tab1 = ctx.new_page(), ctx.new_page()
        tab0.goto(f"{base}/#p={p0['id']}&s={s0['id']}")
        tab1.goto(f"{base}/#p={p1['id']}&s={s1['id']}")
        _wait_text(tab0, "MARK-zero")
        _wait_text(tab1, "MARK-one")
        assert "MARK-one" not in _msgs(tab0) and "MARK-zero" not in _msgs(tab1)
        tab0.reload()  # reload must keep THIS tab's session, not the one last opened in another tab
        _wait_text(tab0, "MARK-zero")
        assert "MARK-one" not in _msgs(tab0)

        # switching sessions while a reply is still streaming must not paint it into the other session
        model.gate = threading.Event()
        tab0.fill("#in", "MARK-slow")
        tab0.click("#send")
        tab0.wait_for_function("document.querySelector('#send').title === 'Stop'")
        tab0.evaluate("(id) => open(id)", s1["id"])
        _wait_text(tab0, "MARK-one")
        model.gate.set()
        real_sleep(1.5)  # let the first session's stream finish while the other is on screen
        text = _msgs(tab0)
        assert "MARK-one" in text and "MARK-slow" not in text and "MARK-zero" not in text
        assert tab0.inner_text("#send") == "" and tab0.get_attribute("#send", "title") == "Send"  # other session is idle
        tab0.evaluate("(id) => open(id)", s0["id"])  # and back: the finished work is there
        _wait_text(tab0, "done MARK-slow")
        assert "MARK-one" not in _msgs(tab0)

        # a session running in one tab is followed live in another tab
        model.gate = threading.Event()
        tab0.fill("#in", "MARK-shared")
        tab0.click("#send")
        tab0.wait_for_function("document.querySelector('#send').title === 'Stop'")
        tab2 = ctx.new_page()
        tab2.goto(f"{base}/#p={p0['id']}&s={s0['id']}")
        _wait_text(tab2, "MARK-shared")
        tab2.wait_for_function("document.querySelector('#send').title === 'Stop'")
        model.gate.set()
        _wait_text(tab2, "done MARK-shared", 12000)
        _wait_text(tab0, "done MARK-shared")
        browser.close()
    for s, other in ((s0, "MARK-one"), (s1, "MARK-zero")):
        assert other not in json.dumps(api("GET", f"/api/sessions/{s['id']}")["messages"])

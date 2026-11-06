"""Test isolation: no test may touch real credentials, the real ~/.harness, or the network.

Environment is cleared BEFORE harness is imported (config reads it at import time and also loads .env files).
"""
import json
import os
import tempfile
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="harness-tests-")
for _k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "GOOGLE_REFRESH_TOKEN",
           "GOOGLE_ACCESS_TOKEN", "VERTEX_PROJECT", "VERTEX_HOST", "OAUTH_TOKEN_URL", "OAUTH_SCOPE", "OAUTH_AUDIENCE",
           "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "HARNESS_REDACT_TERMS", "HARNESS_BLOCK_FILES", "REQUESTS_CA_BUNDLE"):
    os.environ[_k] = ""
os.environ.update(HARNESS_HOME=_TMP, HARNESS_MODELS_FILE=_TMP + "/none.json", VERTEX_LOCATION="global",
                  HARNESS_DEFAULT_MODEL="flash", HARNESS_REDACT="1", OAUTH_CLIENT_AUTH="body",
                  HARNESS_MAX_STEPS="40", HARNESS_COMPACT_AT="80000", HARNESS_KEEP_TURNS="6",
                  HARNESS_HISTORY_TOOL_TRIM="2000")

import pytest  # noqa: E402

from harness import auth, config, server, store, tools  # noqa: E402

DEFAULT_MODELS = {"flash": {"provider": "gemini", "model": "gemini-flash"},
                  "pro": {"provider": "gemini", "model": "gemini-pro"}}


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "projects").mkdir(parents=True)
    monkeypatch.setattr(config, "HOME", home)
    monkeypatch.setattr(config, "PROJECTS_DIR", home / "projects")
    monkeypatch.setattr(config, "LEGACY_SESSIONS_DIR", home / "sessions")
    monkeypatch.setattr(config, "TOKEN_FILE", home / "token.json")
    monkeypatch.setattr(config, "MODELS", {k: dict(v) for k, v in DEFAULT_MODELS.items()})
    for name, val in dict(REDACT=True, REDACT_TERMS=[], ACCESS_TOKEN="", REFRESH_TOKEN="", TOKEN_URL="",
                          CLIENT_ID="", CLIENT_SECRET="", OAUTH_SCOPE="", OAUTH_AUDIENCE="", OAUTH_CLIENT_AUTH="body",
                          PROJECT="", LOCATION="global", HOST="", MAX_STEPS=40, COMPACT_AT=80000, KEEP_TURNS=6,
                          HISTORY_TOOL_TRIM=2000).items():
        monkeypatch.setattr(config, name, val)
    monkeypatch.setitem(auth._cc, "token", "")
    monkeypatch.setitem(auth._cc, "exp", 0.0)
    monkeypatch.setattr(auth, "_creds", None)
    monkeypatch.setattr(tools, "PROJECT", __import__("contextvars").ContextVar("project", default=None))  # agent.run_turn sets it and never resets
    monkeypatch.setattr(server, "_busy", set())
    monkeypatch.setattr(server, "_stops", {})
    monkeypatch.setattr("time.sleep", lambda s: None)  # retry backoff must not slow the suite
    yield


class FakeServer:
    """A local HTTP server that replays scripted responses and records requests."""

    def __init__(self):
        self.responses: deque = deque()
        self.requests: list = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _handle(self):
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n) if n else b""
                outer.requests.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()},
                                       "body": json.loads(raw) if raw else None})
                item = outer.responses.popleft() if outer.responses else (
                    200, {"candidates": [{"content": {"parts": [{"text": "(default)"}]}}]})
                status, payload, ctype = (item + ("application/json",))[:3] if isinstance(item, tuple) else (
                    200, item, "application/json")
                data = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_GET = do_POST = _handle

        self.httpd = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def push(self, *responses):
        self.responses.extend(responses)

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def fake():
    s = FakeServer()
    yield s
    s.close()


@pytest.fixture
def model(fake, monkeypatch):
    """Registers model alias 'fake' (Gemini API-key mode) that talks to the FakeServer."""
    monkeypatch.setenv("FAKE_KEY", "fake-key-123456789")
    monkeypatch.setitem(config.MODELS, "fake", {"provider": "gemini", "model": "m", "api_key_env": "FAKE_KEY",
                                                "base_url": fake.url + "/v1beta"})
    return "fake"


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    return store.create_project("proj", str(root))


@pytest.fixture
def session(project, model):
    return store.create(project["id"], model="fake")


@pytest.fixture
def inproject(project):
    tok = tools.PROJECT.set(project)
    yield project
    tools.PROJECT.reset(tok)


def covers(*ids):
    return pytest.mark.covers(*ids)

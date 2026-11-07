"""R-AUTH: token sources."""
import json
import time

import pytest

from harness import auth, config
from conftest import covers


class Resp:
    def __init__(self, status=200, body=None):
        self.ok, self.status_code, self._b = status < 400, status, body or {}
        self.text = json.dumps(self._b)

    def json(self):
        return self._b


@pytest.fixture
def cc(monkeypatch):
    monkeypatch.setattr(config, "TOKEN_URL", "https://sso.test/token")
    monkeypatch.setattr(config, "CLIENT_ID", "cid")
    monkeypatch.setattr(config, "CLIENT_SECRET", "csecret")
    calls = []

    def post(url, data=None, timeout=None, auth=None, **kw):
        calls.append({"url": url, "data": data, "auth": auth})
        return Resp(200, {"access_token": f"tok{len(calls)}", "expires_in": 3600})
    monkeypatch.setattr(auth.requests, "post", post)
    return calls


@covers("R-AUTH-01")
def test_static_token_wins(monkeypatch, cc):
    monkeypatch.setattr(config, "ACCESS_TOKEN", "static")
    assert auth.get_access_token() == "static" and cc == []


@covers("R-AUTH-02")
def test_client_credentials_body(cc, monkeypatch):
    monkeypatch.setattr(config, "OAUTH_SCOPE", "scope1")
    monkeypatch.setattr(config, "OAUTH_AUDIENCE", "aud1")
    assert auth.get_access_token() == "tok1"
    c = cc[0]
    assert c["url"] == "https://sso.test/token" and c["auth"] is None
    assert c["data"] == {"grant_type": "client_credentials", "scope": "scope1", "audience": "aud1",
                         "client_id": "cid", "client_secret": "csecret"}


@covers("R-AUTH-02")
def test_client_credentials_basic(cc, monkeypatch):
    monkeypatch.setattr(config, "OAUTH_CLIENT_AUTH", "basic")
    auth.get_access_token()
    assert cc[0]["auth"] == ("cid", "csecret") and "client_secret" not in cc[0]["data"]
    assert "scope" not in cc[0]["data"]


@covers("R-AUTH-03")
def test_token_cached_then_refreshed_near_expiry(cc):
    assert auth.get_access_token() == auth.get_access_token() == "tok1" and len(cc) == 1
    auth._cc["exp"] = time.time() + 30  # inside the 60s safety margin
    assert auth.get_access_token() == "tok2"


@covers("R-AUTH-04")
def test_errors_are_actionable(monkeypatch):
    monkeypatch.setattr(config, "TOKEN_URL", "https://sso.test/token")
    with pytest.raises(RuntimeError, match="GOOGLE_CLIENT_ID"):
        auth.get_access_token()
    monkeypatch.setattr(config, "CLIENT_ID", "c")
    monkeypatch.setattr(config, "CLIENT_SECRET", "s")
    monkeypatch.setattr(auth.requests, "post", lambda *a, **k: Resp(401, {"error": "invalid_client"}))
    with pytest.raises(RuntimeError, match="Token endpoint HTTP 401.*invalid_client"):
        auth.get_access_token()
    monkeypatch.setattr(config, "TOKEN_URL", "")
    with pytest.raises(RuntimeError, match="python -m harness login"):
        auth.get_access_token()


@covers("R-AUTH-05")
def test_is_logged_in(monkeypatch):
    assert not auth.is_logged_in()
    monkeypatch.setattr(config, "ACCESS_TOKEN", "t")
    assert auth.is_logged_in()
    monkeypatch.setattr(config, "ACCESS_TOKEN", "")
    monkeypatch.setattr(config, "TOKEN_URL", "u")
    assert not auth.is_logged_in()  # url alone is not enough
    monkeypatch.setattr(config, "CLIENT_ID", "a")
    monkeypatch.setattr(config, "CLIENT_SECRET", "b")
    assert auth.is_logged_in()
    monkeypatch.setattr(config, "TOKEN_URL", "")
    assert not auth.is_logged_in()
    config.TOKEN_FILE.write_text(json.dumps({"refresh_token": "r"}))
    assert auth.is_logged_in()

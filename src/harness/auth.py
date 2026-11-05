"""OAuth: client id + secret -> one-time browser login -> refresh token -> short-lived access tokens."""
import json
import threading
import time

import requests

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

from . import config

TOKEN_URI = "https://oauth2.googleapis.com/token"
_lock = threading.Lock()
_creds: Credentials | None = None
_cc: dict = {"token": "", "exp": 0.0}


def _client_credentials_token() -> str:
    """Org gateway: exchange client id/secret for a bearer token at OAUTH_TOKEN_URL (cached until expiry)."""
    with _lock:
        if _cc["token"] and _cc["exp"] > time.time() + 60:
            return _cc["token"]
        if not (config.CLIENT_ID and config.CLIENT_SECRET):
            raise RuntimeError("GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET are not set.")
        data = {"grant_type": "client_credentials"}
        if config.OAUTH_SCOPE:
            data["scope"] = config.OAUTH_SCOPE
        if config.OAUTH_AUDIENCE:
            data["audience"] = config.OAUTH_AUDIENCE
        kw: dict = {}
        if config.OAUTH_CLIENT_AUTH == "basic":
            kw["auth"] = (config.CLIENT_ID, config.CLIENT_SECRET)
        else:
            data.update(client_id=config.CLIENT_ID, client_secret=config.CLIENT_SECRET)
        r = requests.post(config.TOKEN_URL, data=data, timeout=30, **kw)
        if not r.ok:
            raise RuntimeError(f"Token endpoint HTTP {r.status_code}: {r.text[:500]}")
        j = r.json()
        _cc["token"], _cc["exp"] = j["access_token"], time.time() + float(j.get("expires_in", 3600))
        return _cc["token"]


def login(port: int = 0) -> None:
    """Interactive one-time login. Opens a browser, stores the refresh token."""
    if config.TOKEN_URL:
        print("OAUTH_TOKEN_URL is set: client-credentials mode, no login needed. Run: python -m harness verify")
        return
    from google_auth_oauthlib.flow import InstalledAppFlow

    if not (config.CLIENT_ID and config.CLIENT_SECRET):
        raise SystemExit("Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET first.")
    flow = InstalledAppFlow.from_client_config(
        {"installed": {
            "client_id": config.CLIENT_ID,
            "client_secret": config.CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": TOKEN_URI,
            "redirect_uris": ["http://localhost"],
        }},
        scopes=config.SCOPES,
    )
    creds = flow.run_local_server(port=port, prompt="consent", access_type="offline")
    config.TOKEN_FILE.write_text(json.dumps({"refresh_token": creds.refresh_token}))
    config.TOKEN_FILE.chmod(0o600)
    print(f"Saved refresh token to {config.TOKEN_FILE}")


def _refresh_token() -> str:
    if config.REFRESH_TOKEN:
        return config.REFRESH_TOKEN
    if config.TOKEN_FILE.exists():
        return json.loads(config.TOKEN_FILE.read_text()).get("refresh_token", "")
    return ""


def get_access_token() -> str:
    global _creds
    if config.ACCESS_TOKEN:
        return config.ACCESS_TOKEN
    if config.TOKEN_URL:
        return _client_credentials_token()
    with _lock:
        if _creds is None:
            rt = _refresh_token()
            if not rt:
                raise RuntimeError("Not logged in. Run: python -m harness login")
            _creds = Credentials(
                None, refresh_token=rt, token_uri=TOKEN_URI,
                client_id=config.CLIENT_ID, client_secret=config.CLIENT_SECRET,
                scopes=config.SCOPES,
            )
        if not _creds.valid:
            _creds.refresh(Request())
        return _creds.token


def is_logged_in() -> bool:
    return bool(config.ACCESS_TOKEN or (config.TOKEN_URL and config.CLIENT_ID and config.CLIENT_SECRET) or _refresh_token())

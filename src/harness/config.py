"""All settings come from environment variables so no secrets live in code.

A .env file is loaded automatically (from the current working directory, then ~/.harness/.env)
so you don't need to export variables in your shell. Never commit .env files.
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    # harness/.env first (package-level), then ~/.harness/.env (user-level), then cwd/.env
    _root = Path(__file__).parent.parent.parent  # harness/src/harness -> harness/
    load_dotenv(_root / ".env", override=False)
    load_dotenv(_root / "src" / "harness" / ".env", override=False)
    load_dotenv(override=False)
except ImportError:
    pass  # python-dotenv not installed; env vars must be set in the shell

# Default: .harness/ inside the project root (two levels up from src/harness/)
_PROJECT_ROOT = Path(__file__).parent.parent.parent  # harness/src/harness -> harness/
HOME = Path(os.environ.get("HARNESS_HOME", _PROJECT_ROOT / ".harness"))
PROJECTS_DIR = HOME / "projects"
LEGACY_SESSIONS_DIR = HOME / "sessions"  # pre-project layout, migrated on first run
TOKEN_FILE = HOME / "token.json"
HOME.mkdir(parents=True, exist_ok=True)
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

# --- auth (OAuth client for an installed/desktop app) ---
CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
REFRESH_TOKEN = os.environ.get("GOOGLE_REFRESH_TOKEN", "")  # optional: skip browser login
ACCESS_TOKEN = os.environ.get("GOOGLE_ACCESS_TOKEN", "")    # optional: static bearer token
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
# Org gateway mode (client-credentials grant, no browser): set OAUTH_TOKEN_URL to your org's token endpoint.
TOKEN_URL = os.environ.get("OAUTH_TOKEN_URL", "")
OAUTH_SCOPE = os.environ.get("OAUTH_SCOPE", "")
OAUTH_AUDIENCE = os.environ.get("OAUTH_AUDIENCE", "")
OAUTH_CLIENT_AUTH = os.environ.get("OAUTH_CLIENT_AUTH", "body")  # "body" or "basic"

# --- Vertex ---
PROJECT = os.environ.get("VERTEX_PROJECT", "")
LOCATION = os.environ.get("VERTEX_LOCATION", "global")
# Override if your org fronts Vertex with a gateway, e.g. https://my-gateway.example.com
HOST = os.environ.get("VERTEX_HOST", "")

def _load_models() -> dict:
    """alias -> spec. Built-ins are the two Gemini models; add more in ~/.harness/models.json
    (see models.example.json). Entries whose model starts with REPLACE are ignored, so the
    placeholders in the example file stay inert until you edit them."""
    import json
    models = {
        "flash": {"provider": "gemini", "model": os.environ.get("GEMINI_FLASH_MODEL", "gemini-3.5-flash")},
        "pro": {"provider": "gemini", "model": os.environ.get("GEMINI_PRO_MODEL", "gemini-3.6-flash")},
    }
    f = Path(os.environ.get("HARNESS_MODELS_FILE", HOME / "models.json")).expanduser()
    if f.exists():
        for alias, spec in json.loads(f.read_text()).items():
            if alias.startswith("_") or str(spec.get("model", "")).startswith("REPLACE"):
                continue
            models[alias] = spec
    return models


MODELS = _load_models()
DEFAULT_MODEL = os.environ.get("HARNESS_DEFAULT_MODEL", "flash")

# --- agent ---
WORKDIR = Path(os.environ.get("HARNESS_WORKDIR", os.getcwd())).resolve()  # default root for new projects
MAX_STEPS = int(os.environ.get("HARNESS_MAX_STEPS", "40"))
COMPACT_AT = int(os.environ.get("HARNESS_COMPACT_AT", "80000"))        # compact when prompt exceeds this
KEEP_TURNS = int(os.environ.get("HARNESS_KEEP_TURNS", "6"))            # full turns to keep after compaction
HISTORY_TOOL_TRIM = int(os.environ.get("HARNESS_HISTORY_TOOL_TRIM", "2000"))  # max chars stored per tool result in history

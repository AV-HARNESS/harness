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
HOME = Path(os.environ.get("HARNESS_HOME") or _PROJECT_ROOT / ".harness").expanduser()
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
# Per-model location, e.g. MODEL_LOCATIONS="gemini-3.1-pro-preview=global,gemini-3.7-flash=us"
# Wins over VERTEX_LOCATION; a "location" set on a models.json alias wins over both.
# Only offer models matching these patterns (comma-separated, * allowed), e.g. HARNESS_MODEL_ALLOW=gpt-4.1*,gemini-3.*
MODEL_ALLOW = [x.strip() for x in os.environ.get("HARNESS_MODEL_ALLOW", "").split(",") if x.strip()]
MODEL_LOCATIONS = dict(
    (k.strip(), v.strip()) for k, _, v in (x.partition("=") for x in os.environ.get("MODEL_LOCATIONS", "").split(",")) if k.strip() and v.strip())

_ENV_NAME = __import__("re").compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_DEFAULT_KEY_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "gemini": ""}


def _load_models() -> dict:
    """alias -> spec. Keys live ONLY in .env (never in models.json): the built-in aliases pick up whichever of
    OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY (or a Vertex login) is set, and the model picker shows
    every provider you have a key for. models.json is optional and only adds or overrides aliases (base_url,
    project, location, headers...); an alias without api_key_env uses that provider's standard key variable.
    Entries whose model starts with REPLACE are ignored, so the placeholders in the example file stay inert."""
    import json
    import sys
    models = {
        "flash": {"provider": "gemini", "model": os.environ.get("GEMINI_FLASH_MODEL", "gemini-3.7-flash")},
        "pro": {"provider": "gemini", "model": os.environ.get("GEMINI_PRO_MODEL", "gemini-3.1-pro-preview")},
        "openai": {"provider": "openai", "model": os.environ.get("OPENAI_MODEL", "gpt-4.1"),
                   "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"), "api_key_env": "OPENAI_API_KEY"},
        "claude": {"provider": "anthropic", "model": os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
                   "api_key_env": "ANTHROPIC_API_KEY"},
    }
    f = Path(os.environ.get("HARNESS_MODELS_FILE", HOME / "models.json")).expanduser()
    if f.exists():
        for alias, spec in json.loads(f.read_text(encoding="utf-8")).items():
            if alias.startswith("_") or str(spec.get("model", "")).startswith("REPLACE"):
                continue
            env = spec.get("api_key_env")
            if env and not _ENV_NAME.match(env):  # a real key pasted into models.json: never use or print it
                print(f"[harness] models.json: '{alias}' has a key where an environment variable NAME belongs; "
                      f"ignoring that entry. Put the key in .env (e.g. OPENAI_API_KEY=...) and rotate the pasted one.",
                      file=sys.stderr)
                continue
            if not env:  # standard key variable for the provider, unless it points at a custom server
                provider = spec.get("provider", "gemini")
                custom = spec.get("base_url") and "api.openai.com" not in spec["base_url"] and "anthropic.com" not in spec["base_url"]
                if not custom and _DEFAULT_KEY_ENV.get(provider):
                    spec = {**spec, "api_key_env": _DEFAULT_KEY_ENV[provider]}
            models[alias] = spec
    return models


MODELS = _load_models()


def get_model(alias: str) -> dict | None:
    """Spec for an alias. 'source/model-id' (what the model picker lists) means: the configured alias
    'source' with a different model id, so any model an endpoint offers can be used without editing models.json."""
    spec = MODELS.get(alias)
    if spec is None and "/" in alias:
        src, _, mid = alias.partition("/")
        if src in MODELS and mid:
            spec = {**MODELS[src], "model": mid}
    return spec
DEFAULT_MODEL = os.environ.get("HARNESS_DEFAULT_MODEL", "flash")

# --- agent ---
WORKDIR = Path(os.environ.get("HARNESS_WORKDIR", os.getcwd())).resolve()  # default root for new projects
MAX_STEPS = int(os.environ.get("HARNESS_MAX_STEPS", "200"))
COMPACT_AT = int(os.environ.get("HARNESS_COMPACT_AT", "80000"))        # compact when prompt exceeds this
KEEP_TURNS = int(os.environ.get("HARNESS_KEEP_TURNS", "6"))            # full turns to keep after compaction
HISTORY_TOOL_TRIM = int(os.environ.get("HARNESS_HISTORY_TOOL_TRIM", "2000"))  # max chars stored per tool result in history

# --- privacy: mask secrets before anything is sent to a model (see redact.py) ---
REDACT = os.environ.get("HARNESS_REDACT", "1") != "0"  # on by default; HARNESS_REDACT=0 disables
REDACT_TERMS = [t.strip() for t in os.environ.get("HARNESS_REDACT_TERMS", "").split(",") if t.strip()]
BLOCK_FILES = [  # file names the model may not read, write or edit (fnmatch, case-insensitive)
    ".env", ".env.*", "*.pem", "*.key", "*.pfx", "*.p12", "*.keystore", "id_rsa*", "id_ed25519*",
    "token.json", "credentials*", "secrets.*",
] + [t.strip().lower() for t in os.environ.get("HARNESS_BLOCK_FILES", "").split(",") if t.strip()]

# strict mode also masks long numbers and user names in paths; extra regexes (separated by ||) are masked too
REDACT_STRICT = os.environ.get("HARNESS_REDACT_STRICT", "0") == "1"
REDACT_PATTERNS = [t.strip() for t in os.environ.get("HARNESS_REDACT_PATTERNS", "").split("||") if t.strip()]
# audit log: every request exactly as sent (after masking), one JSON line each. "1" = <HOME>/audit.jsonl, or a path.
_audit = os.environ.get("HARNESS_AUDIT_LOG", "")
AUDIT_LOG = (HOME / "audit.jsonl") if _audit == "1" else (Path(_audit).expanduser() if _audit else None)
# dry run: build and mask the request, log/print it, but do not call the model
DRY_RUN = os.environ.get("HARNESS_DRY_RUN", "0") == "1"

# file tools may only touch the linked project folder (symlinks and .. are resolved). Set 0 to allow anywhere.
CONFINE = os.environ.get("HARNESS_CONFINE", "1") != "0"

AUTO_TITLE = os.environ.get("HARNESS_AUTO_TITLE", "1") != "0"  # model-written session titles (one small extra call per session)
STREAM = os.environ.get("HARNESS_STREAM", "1") != "0"  # show replies as they are written (falls back automatically if the endpoint refuses)

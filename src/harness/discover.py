"""Find the models each configured endpoint actually offers, so the UI can list them automatically.

Sources are the endpoints behind your configured aliases (OpenAI-compatible /models, Gemini API key
/models, Anthropic /v1/models). Vertex has no per-user model list, so Vertex aliases are shown as configured.
Listing is not proof of access (providers list models your key cannot call), so each listed model is probed once with a
tiny request in the background and only confirmed ones are offered. Outcomes are cached on disk for a day, keyed by a hash
of the endpoint and key (never the key itself): ok -> offered; 401/403/404 -> hidden; anything else -> offered as
"unverified" and retried later. A failing endpoint is reported, never raised.
"""
import fnmatch
import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from . import config, errors
from .providers.gemini import gemini_mode

TTL = 600            # how long a model list is reused
ACCESS_TTL = 86400   # how long ok/denied probe results are trusted
RETRY_TTL = 600      # how long before an inconclusive probe is retried
MAX_PROBES = 80      # per endpoint, so a huge catalogue cannot cause a flood of calls
_cache: dict = {"at": 0.0, "groups": []}
_lock = threading.Lock()
_probing = False
_SKIP = ("embed", "whisper", "tts", "dall-e", "moderation", "davinci", "babbage", "image", "audio", "realtime",
         "transcribe", "search", "aqa", "imagen", "veo", "learnlm")


def _get(url: str, headers: dict, params: dict | None = None) -> dict:
    r = requests.get(url, headers=headers, params=params, timeout=10)
    if not r.ok:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    return r.json()


def _chat_capable(mid: str) -> bool:
    return not any(w in mid.lower() for w in _SKIP)


def _list(spec: dict) -> list[str]:
    provider = spec.get("provider", "gemini")
    env = errors.env_name(spec["api_key_env"]) if spec.get("api_key_env") else ""
    key = os.environ.get(env, "") if env else ""
    if provider == "openai":
        if env and not key:
            raise errors.ConfigError(f"{env} is not set")
        base = spec.get("base_url", "https://api.openai.com/v1").rstrip("/")
        h = {"Authorization": f"Bearer {key}"} if key else {}
        data = _get(f"{base}/models", {**h, **spec.get("headers", {})})
        return [m["id"] for m in data.get("data", [])]
    if provider == "anthropic":
        if not key:
            raise RuntimeError(f"{env or 'api_key_env'} is not set")
        base = spec.get("base_url", "https://api.anthropic.com").rstrip("/")
        data = _get(f"{base}/v1/models", {"x-api-key": key, "anthropic-version": "2023-06-01", **spec.get("headers", {})},
                    {"limit": 100})
        return [m["id"] for m in data.get("data", [])]
    key = key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
    base = spec.get("base_url", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
    data = _get(f"{base}/models", {"x-goog-api-key": key, **spec.get("headers", {})}, {"pageSize": 200})
    return [m["name"].split("/", 1)[-1] for m in data.get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])]


def _fingerprint(spec: dict) -> str:
    env = spec.get("api_key_env", "")
    key = (os.environ.get(env, "") if env else "") or os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
    raw = f"{spec.get('provider', 'gemini')}|{spec.get('base_url', '')}|{key}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _access_file():
    return config.HOME / "model_access.json"


def _load_access() -> dict:
    try:
        return json.loads(_access_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_access(data: dict) -> None:
    tmp = _access_file().with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, _access_file())


def probe(spec: dict, model_id: str) -> str:
    """'ok' | 'denied' | 'unknown' - does this key really get an answer from this model?"""
    from .providers import gemini, openai, anthropic
    fn = {"gemini": gemini.generate, "openai": openai.generate, "anthropic": anthropic.generate}[spec.get("provider", "gemini")]
    s = {**spec, "model": model_id, "max_tokens": 8}
    try:
        fn(s, [{"role": "user", "parts": [{"text": "Reply with the single word: ok"}]}], "", None, 0)
        return "ok"
    except Exception as e:
        m = re.search(r"HTTP (\d+)", str(e))
        return "denied" if m and m.group(1) in ("401", "403", "404") else "unknown"


def _fresh(entry: dict | None) -> bool:
    if not entry:
        return False
    ttl = RETRY_TTL if entry["status"] == "unknown" else ACCESS_TTL
    return time.time() - entry["at"] < ttl


def probe_pending() -> None:
    """Probe every allowed, listed model that has no fresh result. Runs in a background thread."""
    global _probing
    try:
        todo = []
        access = _load_access()
        for g in _cache["groups"]:
            fp = _fingerprint(config.MODELS[g["source"]])
            for m in g["models"][:MAX_PROBES]:
                if not _fresh(access.get(fp, {}).get(m["id"])):
                    todo.append((g["source"], fp, m["id"]))
        if not todo:
            return
        with ThreadPoolExecutor(4) as ex:
            results = list(ex.map(lambda t: (t, probe(config.MODELS[t[0]], t[2])), todo))
        with _lock:
            access = _load_access()
            for (_, fp, mid), status in results:
                access.setdefault(fp, {})[mid] = {"status": status, "at": time.time()}
            _save_access(access)
    finally:
        _probing = False


def _start_probes() -> None:
    global _probing
    with _lock:
        if _probing:
            return
        _probing = True
    threading.Thread(target=probe_pending, daemon=True).start()


def _allowed(mid: str) -> bool:
    return not config.MODEL_ALLOW or any(fnmatch.fnmatch(mid, p) for p in config.MODEL_ALLOW)


def usable(spec: dict) -> tuple[bool, str]:
    """Is this alias configured well enough to even try? (credentials present; says what is missing if not)"""
    from . import auth
    provider = spec.get("provider", "gemini")
    env = spec.get("api_key_env", "")
    if env and not errors.ENV_NAME.match(env):  # a pasted key: never echo it
        return False, "api_key_env must be an environment variable NAME, not the key itself"
    if provider in ("openai", "anthropic"):
        if provider == "anthropic" and not env:
            return False, "api_key_env is not set in models.json"
        return (True, "") if not env or os.environ.get(env) else (False, f"{env} is not set")
    if gemini_mode(spec) == "api_key":
        ok = (os.environ.get(env) if env else "") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        return (True, "") if ok else (False, f"{env or 'GEMINI_API_KEY'} is not set")
    if not (spec.get("project") or config.PROJECT):
        return False, "VERTEX_PROJECT is not set"
    return (True, "") if auth.is_logged_in() else (False, "not logged in to Vertex")


def usable_models() -> tuple[dict, dict]:
    """(usable aliases -> spec, hidden alias -> reason) for what this machine's environment can actually run."""
    ok, hidden = {}, {}
    for alias, spec in config.MODELS.items():
        good, why = usable(spec)
        if good:
            ok[alias] = spec
        else:
            hidden[alias] = why
    return ok, hidden


def _sources() -> dict:
    """One entry per distinct endpoint: (provider, base_url, key env) -> first alias using it."""
    out: dict = {}
    for alias, spec in usable_models()[0].items():
        provider = spec.get("provider", "gemini")
        if provider == "gemini" and gemini_mode(spec) != "api_key":
            continue  # Vertex: cannot be listed
        out.setdefault((provider, spec.get("base_url", ""), spec.get("api_key_env", "")), alias)
    return out


def _lists(refresh: bool) -> list[dict]:
    if not refresh and time.time() - _cache["at"] < TTL and _cache["groups"]:
        return _cache["groups"]
    groups = []
    for (provider, _, _), alias in _sources().items():
        g = {"source": alias, "provider": provider, "models": [], "error": None}
        try:
            ids = sorted({m for m in _list(config.MODELS[alias]) if _chat_capable(m) and _allowed(m)})
            g["models"] = [{"alias": f"{alias}/{m}", "id": m} for m in ids]
        except Exception as e:  # endpoint down, bad key, no /models route...
            g["error"] = str(e)[:300]
        groups.append(g)
    _cache.update(at=time.time(), groups=groups)
    return groups


def discover(refresh: bool = False, recheck: bool = False, background: bool = True) -> dict:
    """{groups: [{source, provider, error, models: [{alias, id, status}]}], pending: n}
    status is 'ok' (confirmed) or 'unverified' (probe inconclusive); denied and not-yet-probed models are left out."""
    if recheck:
        with _lock:
            _save_access({})
    lists = _lists(refresh or recheck)
    if background:
        _start_probes()
    access = _load_access()
    groups, pending = [], 0
    for g in lists:
        fp = _fingerprint(config.MODELS[g["source"]])
        out = {**g, "models": []}
        for m in g["models"]:
            e = access.get(fp, {}).get(m["id"])
            if not e or (e["status"] == "unknown" and not _fresh(e)):
                pending += 1
            elif e["status"] == "ok":
                out["models"].append({**m, "status": "ok"})
            elif e["status"] == "unknown":
                out["models"].append({**m, "status": "unverified"})
        groups.append(out)
    return {"groups": groups, "pending": pending}

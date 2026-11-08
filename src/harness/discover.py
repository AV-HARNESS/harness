"""Find the models each configured endpoint actually offers, so the UI can list them automatically.

Sources are the endpoints behind your configured aliases (OpenAI-compatible /models, Gemini API key
/models, Anthropic /v1/models). Vertex has no per-user model list, so Vertex aliases are shown as configured.
Results are cached for a few minutes; a failing endpoint is reported, never raised.
"""
import os
import time

import requests

from . import config
from .providers.gemini import gemini_mode

TTL = 600
_cache: dict = {"at": 0.0, "groups": []}
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
    env = spec.get("api_key_env", "")
    key = os.environ.get(env, "") if env else ""
    if provider == "openai":
        if env and not key:
            raise RuntimeError(f"{env} is not set")
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


def _sources() -> dict:
    """One entry per distinct endpoint: (provider, base_url, key env) -> first alias using it."""
    out: dict = {}
    for alias, spec in config.MODELS.items():
        provider = spec.get("provider", "gemini")
        if provider == "gemini" and gemini_mode(spec) != "api_key":
            continue  # Vertex: cannot be listed
        out.setdefault((provider, spec.get("base_url", ""), spec.get("api_key_env", "")), alias)
    return out


def discover(refresh: bool = False) -> list[dict]:
    """[{source, provider, models: [{alias, id}], error}] - alias is what you pass as the model."""
    if not refresh and time.time() - _cache["at"] < TTL and _cache["groups"]:
        return _cache["groups"]
    groups = []
    for (provider, _, _), alias in _sources().items():
        g = {"source": alias, "provider": provider, "models": [], "error": None}
        try:
            ids = sorted({m for m in _list(config.MODELS[alias]) if _chat_capable(m)})
            g["models"] = [{"alias": f"{alias}/{m}", "id": m} for m in ids]
        except Exception as e:  # endpoint down, bad key, no /models route...
            g["error"] = str(e)[:300]
        groups.append(g)
    _cache.update(at=time.time(), groups=groups)
    return groups

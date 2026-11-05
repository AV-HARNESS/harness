"""Gemini provider: Vertex AI or Gemini API key."""
import os

from .. import auth, config
from . import _post


def gemini_mode(spec: dict) -> str:
    """'api_key' if the spec names a key env var (or GEMINI_API_KEY/GOOGLE_API_KEY is set and no Vertex
    project is configured); otherwise 'vertex' (Google OAuth login / access token)."""
    if spec.get("api_key_env"):
        return "api_key"
    project = spec.get("project") or config.PROJECT
    if not project and (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        return "api_key"
    return "vertex"


def _key(spec: dict) -> str:
    env = spec.get("api_key_env", "")
    k = os.environ.get(env, "") if env else ""
    return k or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")


def generate(spec: dict, contents: list, system: str, tools, retries: int) -> tuple[dict, dict, dict]:
    body: dict = {"contents": contents}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    if tools:
        body["tools"] = tools

    if gemini_mode(spec) == "api_key":
        key = _key(spec)
        if not key:
            raise RuntimeError(f"Gemini API key is not set (env {spec.get('api_key_env') or 'GEMINI_API_KEY'} is not set).")
        base = spec.get("base_url", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        url = f"{base}/models/{spec['model']}:generateContent"
        headers = {"x-goog-api-key": key, **spec.get("headers", {})}
    else:
        project = spec.get("project") or config.PROJECT
        if not project:
            raise RuntimeError("VERTEX_PROJECT is not set (or use a Gemini API key: set GEMINI_API_KEY).")
        loc = spec.get("location") or config.LOCATION
        host = spec.get("base_url") or config.HOST or (
            "https://aiplatform.googleapis.com" if loc == "global"
            else f"https://{loc}-aiplatform.googleapis.com")
        url = (f"{host.rstrip('/')}/v1/projects/{project}/locations/{loc}"
               f"/publishers/google/models/{spec['model']}:generateContent")
        headers = {"Authorization": f"Bearer {auth.get_access_token()}", **spec.get("headers", {})}

    data = _post(url, headers, body, retries)
    cands = data.get("candidates") or []
    if not cands:
        raise RuntimeError(f"No candidates returned: {str(data)[:800]}")
    cand = cands[0]
    content = cand.get("content") or {}
    if not content.get("parts"):
        content = {"parts": [{"text": f"(empty response, finishReason={cand.get('finishReason')})"}]}
    content["role"] = "model"  # parts kept verbatim: preserves thought signatures
    return content, data.get("usageMetadata", {}), cand


def gemini_alias() -> str | None:
    return next((a for a, s in config.MODELS.items() if s.get("provider", "gemini") == "gemini"), None)

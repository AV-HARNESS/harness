"""Gemini provider: Vertex AI or Gemini API key."""
import os

from .. import auth, config, errors
from . import _NO_STREAM, _post, _stream_events, _with_stream_fallback, gcache


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
    k = os.environ.get(errors.env_name(env), "") if env else ""
    return k or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")


def thinking_config(spec: dict) -> dict | None:
    """Map the session's effort (low/medium/high) to Gemini's thinking setting; None = model default."""
    effort, model = spec.get("effort"), spec.get("model", "")
    if effort not in ("low", "medium", "high"):
        return None
    if "gemini-3" in model:  # Gemini 3 takes a level; Pro has only low/high
        level = effort if "pro" not in model or effort == "low" else "high"
        return {"thinkingLevel": level}
    return {"thinkingBudget": {"low": 1024, "medium": 8192, "high": -1}[effort]}  # 2.5: token budget, -1 = dynamic


def generate(spec: dict, contents: list, system: str, tools, retries: int, on_text=None) -> tuple[dict, dict, dict]:
    body: dict = {"contents": contents}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    if tools:
        body["tools"] = tools
    think = thinking_config(spec)
    if think:
        body["generationConfig"] = {"thinkingConfig": think}

    if gemini_mode(spec) == "api_key":
        key = _key(spec)
        if not key:
            raise errors.ConfigError(f"{spec.get('api_key_env') and errors.env_name(spec['api_key_env']) or 'GEMINI_API_KEY'} is not set. Add it to your .env file and restart the harness.")
        base = spec.get("base_url", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        url = f"{base}/models/{spec['model']}:generateContent"
        headers = {"x-goog-api-key": key, **spec.get("headers", {})}
        cache_url, model_ref = f"{base}/cachedContents", f"models/{spec['model']}"
    else:
        project = spec.get("project") or config.PROJECT
        if not project:
            raise RuntimeError("VERTEX_PROJECT is not set (or use a Gemini API key: set GEMINI_API_KEY).")
        loc = spec.get("location") or config.MODEL_LOCATIONS.get(spec["model"]) or config.LOCATION
        host = spec.get("base_url") or config.HOST or (
            "https://aiplatform.googleapis.com" if loc == "global"
            else f"https://{loc}-aiplatform.googleapis.com")
        url = (f"{host.rstrip('/')}/v1/projects/{project}/locations/{loc}"
               f"/publishers/google/models/{spec['model']}:generateContent")
        headers = {"Authorization": f"Bearer {auth.get_access_token()}", **spec.get("headers", {})}
        cache_url = [f"{host.rstrip('/')}/{v}/projects/{project}/locations/{loc}/cachedContents" for v in ("v1beta1", "v1")]
        model_ref = f"projects/{project}/locations/{loc}/publishers/google/models/{spec['model']}"

    def fetch(b, url=url):
        def streamed(cb):
            parts, usage, last = [], {}, {}
            for ev in _stream_events(url.replace(":generateContent", ":streamGenerateContent") + "?alt=sse", headers, b, retries):
                usage = ev.get("usageMetadata") or usage
                last = (ev.get("candidates") or [last])[0]
                for p in (last.get("content") or {}).get("parts") or []:
                    if "text" in p and "functionCall" not in p:
                        thought = bool(p.get("thought"))
                        open_ = parts and "text" in parts[-1] and bool(parts[-1].get("thought")) == thought \
                            and "thoughtSignature" not in parts[-1]
                        if open_:  # merge chunks of one text part; a part that already carries its signature stays closed
                            parts[-1]["text"] += p["text"]
                            if p.get("thoughtSignature"):
                                parts[-1]["thoughtSignature"] = p["thoughtSignature"]
                        else:
                            parts.append(dict(p))
                        if p["text"] and not thought:
                            cb(p["text"])
                    else:
                        parts.append(p)
            return {"candidates": [{**last, "content": {"parts": parts}}], "usageMetadata": usage}

        return _with_stream_fallback(url, on_text, streamed, lambda: _post(url, headers, b, retries))

    plan = gcache.prepare(cache_url, model_ref, headers, system, tools, contents)
    if plan:
        no_stream_before = url in _NO_STREAM
        try:
            call_url = url.replace("/v1/", "/v1beta1/", 1) if "/v1beta1/" in plan[2] else url  # same API version as the cache
            data = fetch(gcache.request_body(body, contents, *plan), call_url)
        except RuntimeError as e:
            if not str(e).startswith("HTTP 4") or str(e).startswith("HTTP 429"):
                raise
            if not no_stream_before:
                _NO_STREAM.discard(url)  # the cache was at fault, not streaming
            gcache.disable(cache_url if isinstance(cache_url, str) else cache_url[0], f"use failed: {e}")  # e.g. cache expired or not allowed: send the full request
            data = fetch(body)
    else:
        data = fetch(body)
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

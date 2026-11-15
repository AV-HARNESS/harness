"""LLM providers: generate(alias, contents, system, tools) -> (content, usage, raw_candidate).

History is stored in Gemini 'contents' format; each provider translates to/from its own wire format,
so a session can be continued with a different model.

Shared utilities (_post, _lower_types, _text, _result_text, _declarations) live here and are imported
by the individual provider modules. The public entry point is generate().
"""
import time

import requests

from .. import config


# ---- shared utilities used by provider modules ----

def _post(url: str, headers: dict, body: dict, retries: int = 4) -> dict:
    delay = 2.0
    for attempt in range(retries + 1):
        r = requests.post(url, json=body, headers=headers, timeout=300)
        if r.status_code in (429, 500, 502, 503, 504) and attempt < retries:
            time.sleep(delay)
            delay *= 2
            continue
        if not r.ok:
            raise RuntimeError(f"HTTP {r.status_code} from {url.split('?')[0]}: {r.text[:1500]}")
        return r.json()


_NO_STREAM: set[str] = set()  # endpoints that refused streaming once; they use plain requests from then on


def _stream_events(url: str, headers: dict, body: dict, retries: int = 4):
    """POST and yield the JSON payload of each server-sent-event line as it arrives."""
    import json
    delay = 2.0
    for attempt in range(retries + 1):
        r = requests.post(url, json=body, headers=headers, timeout=300, stream=True)
        if r.status_code in (429, 500, 502, 503, 504) and attempt < retries:
            r.close()
            time.sleep(delay)
            delay *= 2
            continue
        if not r.ok:
            text = r.text[:1500]
            r.close()
            raise RuntimeError(f"HTTP {r.status_code} from {url.split('?')[0]}: {text}")
        break
    seen = False
    try:
        for raw in r.iter_lines(chunk_size=1):  # small chunks: deliver each event the moment it arrives
            line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload and payload != "[DONE]":
                seen = True
                yield json.loads(payload)
    finally:
        r.close()
    if not seen:
        raise RuntimeError("stream returned no events")


def _with_stream_fallback(key: str, on_text, streamed, plain):
    """Run streamed(on_text) when allowed; if the endpoint rejects streaming before any text arrived, remember
    that and fall back to the plain request so a gateway without streaming still works."""
    if not on_text or key in _NO_STREAM:
        return plain()
    got = []
    def cb(d):
        got.append(1)
        on_text(d)
    try:
        return streamed(cb)
    except Exception as e:
        if got or not str(e).startswith(("HTTP 4", "stream returned")):
            raise
        _NO_STREAM.add(key)
        return plain()


def _lower_types(schema):
    if isinstance(schema, dict):
        return {k: (v.lower() if k == "type" and isinstance(v, str) else _lower_types(v)) for k, v in schema.items()}
    if isinstance(schema, list):
        return [_lower_types(x) for x in schema]
    return schema


def _declarations(tools) -> list[dict]:
    """Extract plain function declarations from Gemini-format tool list; drops provider-specific tools."""
    out = []
    for t in tools or []:
        out += t.get("functionDeclarations", [])
    return out


def _text(m: dict) -> str:
    return "".join(p["text"] for p in m["parts"] if "text" in p and not p.get("thought"))


def _result_text(fr: dict) -> str:
    r = fr.get("response", {})
    return str(r.get("output", r))


# ---- audit / dry run ----

def _audit(alias, spec, contents, system, tools, counts) -> None:
    """Record the request exactly as it leaves this machine (already masked). Never records credentials."""
    import json
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "alias": alias, "provider": spec.get("provider", "gemini"),
           "model": spec.get("model"), "masked": dict(counts), "dry_run": config.DRY_RUN,
           "system": system, "contents": contents, "tools": [d["name"] for d in _declarations(tools)]}
    line = json.dumps(rec, ensure_ascii=False)
    if config.AUDIT_LOG:
        config.AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
        new = not config.AUDIT_LOG.exists()
        with open(config.AUDIT_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        if new:
            try:
                config.AUDIT_LOG.chmod(0o600)
            except OSError:
                pass
    else:
        import sys
        print("[dry-run] request that would be sent:\n" + json.dumps(rec, indent=2, ensure_ascii=False), file=sys.stderr)


# ---- public entry point ----

def generate(alias: str, contents: list, system: str = "", tools: list | None = None,
             retries: int = 4, effort: str = "", on_text=None) -> tuple[dict, dict, dict]:
    from .gemini import generate as _g
    from .openai import generate as _o
    from .anthropic import generate as _a

    spec = config.get_model(alias)
    if not spec:
        raise RuntimeError(f"Unknown model '{alias}'. Configured: {', '.join(config.MODELS)}")
    if config.REDACT:  # mask secrets/PII in the copy that leaves this machine (files and history stay intact)
        from .. import redact
        contents, system, counts = redact.mask_request(contents, system)
        redact.report(counts)
    else:
        counts = {}
    if config.AUDIT_LOG or config.DRY_RUN:
        _audit(alias, spec, contents, system, tools, counts)
    if config.DRY_RUN:
        return ({"role": "model", "parts": [{"text": "(dry run: request was masked and logged, not sent)"}]}, {}, {})
    if effort in ("low", "medium", "high"):
        spec = {**spec, "effort": effort}
    provider = spec.get("provider", "gemini")
    fn = {"gemini": _g, "openai": _o, "anthropic": _a}.get(provider)
    if not fn:
        raise RuntimeError(f"Unknown provider '{provider}'. Use one of: gemini, openai, anthropic")
    import time
    t0 = time.monotonic()
    content, usage, cand = fn(spec, contents, system, tools, retries, **({"on_text": on_text} if on_text and config.STREAM else {}))
    secs = round(time.monotonic() - t0, 1)
    usage = {**usage, "secs": secs}  # wall time of this model call, retries included
    from .. import errors
    errors.logger().info("model call %s: %.1fs prompt=%s thoughts=%s output=%s effort=%s",
                         alias, secs, usage.get("promptTokenCount", usage.get("prompt_tokens", "?")),
                         usage.get("thoughtsTokenCount", 0), usage.get("candidatesTokenCount", "?"), effort or "default")
    return content, usage, cand


# re-export for callers that import from providers directly
from .gemini import gemini_mode, gemini_alias  # noqa: E402, F401

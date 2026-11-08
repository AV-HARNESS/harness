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
             retries: int = 4) -> tuple[dict, dict, dict]:
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
    provider = spec.get("provider", "gemini")
    fn = {"gemini": _g, "openai": _o, "anthropic": _a}.get(provider)
    if not fn:
        raise RuntimeError(f"Unknown provider '{provider}'. Use one of: gemini, openai, anthropic")
    return fn(spec, contents, system, tools, retries)


# re-export for callers that import from providers directly
from .gemini import gemini_mode, gemini_alias  # noqa: E402, F401

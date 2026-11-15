"""Anthropic Messages API provider."""
import json
import os

from .. import errors

from . import _post, _stream_events, _with_stream_fallback, _declarations, _lower_types, _text, _result_text


def _key(spec: dict) -> str:
    env = spec.get("api_key_env", "")
    if not env:
        raise errors.ConfigError("api_key_env is required for Anthropic models in models.json (the NAME of the env var that holds the key).")
    k = os.environ.get(errors.env_name(env), "")
    if not k:
        raise errors.ConfigError(f"{env} is not set. Add it to your .env file and restart the harness.")
    return k


def _to_anthropic(contents: list) -> list:
    msgs = []
    for i, m in enumerate(contents):
        blocks = []
        if m["role"] == "user":
            for j, p in enumerate(x for x in m["parts"] if "functionResponse" in x):
                blocks.append({"type": "tool_result", "tool_use_id": f"call_{i - 1}_{j}",
                               "content": _result_text(p["functionResponse"])})
            blocks += [{"type": "text", "text": p["text"]} for p in m["parts"] if p.get("text")]
        else:
            if _text(m):
                blocks.append({"type": "text", "text": _text(m)})
            for j, p in enumerate(x for x in m["parts"] if "functionCall" in x):
                c = p["functionCall"]
                blocks.append({"type": "tool_use", "id": f"call_{i}_{j}",
                                "name": c["name"], "input": c.get("args", {})})
        if not blocks:
            continue
        role = "user" if m["role"] == "user" else "assistant"
        if msgs and msgs[-1]["role"] == role:  # the API requires alternating roles
            msgs[-1]["content"] += blocks
        else:
            msgs.append({"role": role, "content": blocks})
    return msgs


def generate(spec: dict, contents: list, system: str, tools, retries: int, on_text=None) -> tuple[dict, dict, dict]:
    base = spec.get("base_url", "https://api.anthropic.com").rstrip("/")
    headers = {"x-api-key": _key(spec), "anthropic-version": "2023-06-01", **spec.get("headers", {})}
    body = {"model": spec["model"], "max_tokens": spec.get("max_tokens", 8192),
            "messages": _to_anthropic(contents), **spec.get("extra", {})}
    if system:
        body["system"] = system
    decls = _declarations(tools)
    if decls:
        body["tools"] = [{"name": d["name"], "description": d["description"],
                          "input_schema": _lower_types(d["parameters"])} for d in decls]
    def streamed(cb):
        blocks, usage = {}, {}
        for ev in _stream_events(f"{base}/v1/messages", headers, {**body, "stream": True}, retries):
            kind = ev.get("type")
            if kind == "message_start":
                usage.update((ev.get("message") or {}).get("usage") or {})
            elif kind == "content_block_start":
                b = dict(ev.get("content_block") or {})
                if b.get("type") == "tool_use":
                    b["_json"] = ""
                blocks[ev["index"]] = b
            elif kind == "content_block_delta":
                d, b = ev.get("delta") or {}, blocks.setdefault(ev["index"], {"type": "text", "text": ""})
                if d.get("type") == "text_delta":
                    b["text"] = b.get("text", "") + d["text"]
                    cb(d["text"])
                elif d.get("type") == "input_json_delta":
                    b["_json"] = b.get("_json", "") + d.get("partial_json", "")
            elif kind == "message_delta":
                usage.update(ev.get("usage") or {})
        out = []
        for i in sorted(blocks):
            b = blocks[i]
            if b.get("type") == "tool_use":
                try:
                    b["input"] = json.loads(b.pop("_json", "") or "{}")
                except json.JSONDecodeError:
                    b["input"] = {}
            out.append(b)
        return {"content": out, "usage": usage}

    data = _with_stream_fallback(base, on_text, streamed, lambda: _post(f"{base}/v1/messages", headers, body, retries))
    parts = []
    for b in data.get("content", []):
        if b["type"] == "text" and b["text"]:
            parts.append({"text": b["text"]})
        elif b["type"] == "tool_use":
            parts.append({"functionCall": {"name": b["name"], "args": b.get("input", {})}})
    if not parts:
        parts = [{"text": "(empty response)"}]
    u = data.get("usage") or {}
    prompt = u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
    return {"role": "model", "parts": parts}, {"promptTokenCount": prompt}, {}

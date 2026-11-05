"""Anthropic Messages API provider."""
import os

from . import _post, _declarations, _lower_types, _text, _result_text


def _key(spec: dict) -> str:
    env = spec.get("api_key_env", "")
    if not env:
        raise RuntimeError(f"api_key_env is required for Anthropic models (model '{spec['model']}').")
    k = os.environ.get(env, "")
    if not k:
        raise RuntimeError(f"Environment variable {env} is not set (needed for model '{spec['model']}').")
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


def generate(spec: dict, contents: list, system: str, tools, retries: int) -> tuple[dict, dict, dict]:
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
    data = _post(f"{base}/v1/messages", headers, body, retries)
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

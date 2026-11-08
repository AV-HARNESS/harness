"""OpenAI-compatible provider (/chat/completions): OpenAI, OpenRouter, Ollama, vLLM, LiteLLM, gateways."""
import json
import os

from .. import errors

from . import _post, _declarations, _lower_types, _text, _result_text


def _key(spec: dict) -> str:
    env = spec.get("api_key_env", "")
    if not env:
        return ""
    k = os.environ.get(errors.env_name(env), "")
    if not k:
        raise errors.ConfigError(f"{env} is not set. Add it to your .env file and restart the harness.")
    return k


def _to_openai(system: str, contents: list) -> list:
    msgs = [{"role": "system", "content": system}] if system else []
    for i, m in enumerate(contents):
        if m["role"] == "user":
            for j, p in enumerate(x for x in m["parts"] if "functionResponse" in x):
                msgs.append({"role": "tool", "tool_call_id": f"call_{i - 1}_{j}",
                             "content": _result_text(p["functionResponse"])})
            t = "\n".join(p["text"] for p in m["parts"] if "text" in p)
            if t:
                msgs.append({"role": "user", "content": t})
        else:
            calls = [p["functionCall"] for p in m["parts"] if "functionCall" in p]
            msg = {"role": "assistant", "content": _text(m) or None}
            if calls:
                msg["tool_calls"] = [
                    {"id": f"call_{i}_{j}", "type": "function",
                     "function": {"name": c["name"], "arguments": json.dumps(c.get("args", {}))}}
                    for j, c in enumerate(calls)]
            msgs.append(msg)
    return msgs


def generate(spec: dict, contents: list, system: str, tools, retries: int) -> tuple[dict, dict, dict]:
    base = spec.get("base_url", "https://api.openai.com/v1").rstrip("/")
    headers = dict(spec.get("headers", {}))
    key = _key(spec)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    body = {"model": spec["model"], "messages": _to_openai(system, contents), **spec.get("extra", {})}
    decls = _declarations(tools)
    if decls:
        body["tools"] = []
        for d in decls:
            params = _lower_types(d["parameters"])
            if not params.get("required"):
                params.pop("required", None)
            body["tools"].append({"type": "function", "function": {
                "name": d["name"], "description": d["description"], "parameters": params}})
    data = _post(f"{base}/chat/completions", headers, body, retries)
    msg = (data.get("choices") or [{}])[0].get("message") or {}
    parts = []
    if msg.get("content"):
        parts.append({"text": msg["content"]})
    for tc in msg.get("tool_calls") or []:
        try:
            args = json.loads(tc["function"].get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        parts.append({"functionCall": {"name": tc["function"]["name"], "args": args}})
    if not parts:
        parts = [{"text": "(empty response)"}]
    return ({"role": "model", "parts": parts},
            {"promptTokenCount": (data.get("usage") or {}).get("prompt_tokens", 0)}, {})

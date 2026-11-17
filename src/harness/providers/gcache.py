"""Gemini explicit context cache.

A long session re-sends its whole prompt (system + tools + history) on every step. Gemini can keep that prefix
server-side: we create a cache of everything except the newest message, then each call sends only the new
messages plus a reference to the cache. That cuts latency and cost on big prompts.

Everything here is best-effort: if the gateway refuses to create or use a cache, caching is switched off for that
endpoint for a while and the call is made normally, so it can never break a chat.
"""
import hashlib
import json
import threading
import time

import requests

from .. import config, errors

_LOCK = threading.Lock()
_CACHES: dict = {}   # (endpoint, system+tools hash, first-message hash) -> {name, n, sig, exp, url}
_OFF: dict = {}      # endpoint -> time until which caching is switched off


def _h(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def est_tokens(obj) -> int:
    return len(json.dumps(obj, ensure_ascii=False)) // 4


def disable(endpoint: str, why: str) -> None:
    _OFF[endpoint] = time.time() + 600
    errors.logger().warning("gemini cache off for 10 min (%s): %s", endpoint.split("?")[0][-60:], why[:200])


def _delete(url: str, headers: dict) -> None:
    def go():
        try:
            requests.delete(url, headers=headers, timeout=15)
        except Exception:
            pass
    threading.Thread(target=go, daemon=True).start()


def prepare(create_url: str, model_ref: str, headers: dict, system: str, tools, contents: list):
    """(cache_name, n_cached_messages) to send the request with, or None to send it normally."""
    if not config.GEMINI_CACHE or time.time() < _OFF.get(create_url, 0) or len(contents) < 3:
        return None
    key = (create_url, _h([system, tools]), _h(contents[0]))
    now = time.time()
    with _LOCK:
        c = _CACHES.get(key)
    valid = bool(c and c["exp"] > now + 30 and c["n"] <= len(contents) - 1 and _h(contents[:c["n"]]) == c["sig"])
    if valid and est_tokens(contents[c["n"]:]) < config.GEMINI_CACHE_REFRESH:
        return c["name"], c["n"]  # the cache still covers nearly everything: reuse it
    cover = contents[:-1]  # everything but the newest message
    if est_tokens([system, tools, cover]) < config.GEMINI_CACHE_MIN:
        return (c["name"], c["n"]) if valid else None  # too small to be worth a cache
    body = {"model": model_ref, "contents": cover, "ttl": f"{config.GEMINI_CACHE_TTL}s"}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    if tools:
        body["tools"] = tools
    t0 = time.monotonic()
    try:
        r = requests.post(create_url, json=body, headers=headers, timeout=60)
        if not r.ok:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        name = r.json()["name"]
    except Exception as e:
        disable(create_url, f"create failed: {e}")
        return (c["name"], c["n"]) if valid else None
    errors.logger().info("gemini cache created: %d messages, ~%d tokens, %.1fs",
                         len(cover), est_tokens([system, tools, cover]), time.monotonic() - t0)
    with _LOCK:
        _CACHES[key] = {"name": name, "n": len(cover), "sig": _h(cover), "exp": now + config.GEMINI_CACHE_TTL}
    if c:
        _delete(create_url + "/" + c["name"].rsplit("/", 1)[-1], headers)
    return name, len(cover)


def request_body(body: dict, contents: list, name: str, n: int) -> dict:
    """The request that references the cache: system and tools live in the cache, so they are left out."""
    out = {k: v for k, v in body.items() if k not in ("systemInstruction", "tools", "contents")}
    out["contents"] = contents[n:]
    out["cachedContent"] = name
    return out

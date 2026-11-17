"""Gemini explicit context cache.

A long session re-sends its whole prompt (system + tools + history) on every step. Gemini can keep that prefix
server-side: we create a cache of everything except the newest message, then each call sends only the new
messages plus a reference to the cache. That cuts latency and cost on big prompts.

Everything here is best-effort: if the gateway refuses to create or use a cache, caching is switched off for that
endpoint for a while and the call is made normally, so it can never break a chat.
"""
import hashlib
import json
import re
import threading
import time

import requests

from .. import config, errors

_LOCK = threading.Lock()
_CACHES: dict = {}   # (endpoint, system+tools hash, first-message hash) -> {name, n, sig, exp, url}
_GOOD: dict = {}     # endpoint -> the create address that worked
_OFF: dict = {}      # endpoint -> time until which caching is switched off


def _h(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def est_tokens(obj) -> int:
    return len(json.dumps(obj, ensure_ascii=False)) // 4


def disable(endpoint: str, why: str) -> None:
    _OFF[endpoint] = time.time() + 600
    short = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", why)).strip()  # error pages are HTML: log just the words
    errors.logger().warning("gemini cache off for 10 min (%s): %s", endpoint.split("?")[0][-60:], short[:160])


def _delete(url: str, headers: dict) -> None:
    def go():
        try:
            requests.delete(url, headers=headers, timeout=15)
        except Exception:
            pass
    threading.Thread(target=go, daemon=True).start()


def prepare(create_urls, model_ref: str, headers: dict, system: str, tools, contents: list):
    """(cache_name, n_cached_messages, create_url_used) to send the request with, or None to send it normally.

    create_urls are alternative addresses of the same endpoint (Vertex gateways differ: /v1beta1 or /v1); the first
    that does not answer 404 is remembered and used from then on."""
    if isinstance(create_urls, str):
        create_urls = [create_urls]
    create_url = create_urls[0]  # identifies the endpoint
    if not config.GEMINI_CACHE or time.time() < _OFF.get(create_url, 0) or len(contents) < 3:
        return None
    key = (create_url, _h([system, tools]), _h(contents[0]))
    now = time.time()
    with _LOCK:
        c = _CACHES.get(key)
    valid = bool(c and c["exp"] > now + 30 and c["n"] <= len(contents) - 1 and _h(contents[:c["n"]]) == c["sig"])
    if valid and est_tokens(contents[c["n"]:]) < config.GEMINI_CACHE_REFRESH:
        return c["name"], c["n"], c["url"]  # the cache still covers nearly everything: reuse it
    cover = contents[:-1]  # everything but the newest message
    if est_tokens([system, tools, cover]) < config.GEMINI_CACHE_MIN:
        return (c["name"], c["n"], c["url"]) if valid else None  # too small to be worth a cache
    body = {"model": model_ref, "contents": cover, "ttl": f"{config.GEMINI_CACHE_TTL}s"}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    if tools:
        body["tools"] = tools
    t0 = time.monotonic()
    order = sorted(create_urls, key=lambda u: u != _GOOD.get(create_url))  # the address that worked before goes first
    name, used, err = None, None, None
    for u in order:
        try:
            r = requests.post(u, json=body, headers=headers, timeout=60)
            if r.status_code == 404 and u != order[-1]:
                err = "HTTP 404"
                continue  # try the other address
            if not r.ok:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            name, used = r.json()["name"], u
            break
        except Exception as e:
            err = str(e)
            break
    if not name:
        disable(create_url, f"create failed: {err}")
        return (c["name"], c["n"], c["url"]) if valid else None
    _GOOD[create_url] = used
    errors.logger().info("gemini cache created: %d messages, ~%d tokens, %.1fs",
                         len(cover), est_tokens([system, tools, cover]), time.monotonic() - t0)
    with _LOCK:
        _CACHES[key] = {"name": name, "n": len(cover), "sig": _h(cover), "exp": now + config.GEMINI_CACHE_TTL, "url": used}
    if c:
        _delete(c["url"] + "/" + c["name"].rsplit("/", 1)[-1], headers)
    return name, len(cover), used


def request_body(body: dict, contents: list, name: str, n: int, used: str = "") -> dict:
    """The request that references the cache: system and tools live in the cache, so they are left out."""
    out = {k: v for k, v in body.items() if k not in ("systemInstruction", "tools", "contents")}
    out["contents"] = contents[n:]
    out["cachedContent"] = name
    return out

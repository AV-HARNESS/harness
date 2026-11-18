"""Gemini explicit context cache.

A long session re-sends its whole prompt (system + tools + history) on every step. Gemini can keep that prefix
server-side: we create a cache of everything except the newest message, then each call sends only the new
messages plus a reference to the cache. That cuts latency and cost on big prompts.

Gateways differ, so a cache is tried in several "modes" (API version x location) and the first that works is
remembered. The cache is built in the background so a slow or failing attempt never delays a chat step. Everything is
best-effort: if nothing works, caching is switched off for a while and calls are made normally.
"""
import hashlib
import json
import re
import threading
import time

import requests

from .. import config, errors

_LOCK = threading.Lock()
_CACHES: dict = {}    # (endpoint, system+tools hash, first-message hash) -> {name, n, sig, exp, cand}
_GOOD: dict = {}      # endpoint -> the candidate (mode) that worked
_OFF: dict = {}       # endpoint -> time until which caching is switched off
_INFLIGHT: set = set()
_THREADS: list = []


def _h(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def est_tokens(obj) -> int:
    return len(json.dumps(obj, ensure_ascii=False)) // 4


def _short(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", str(text))).strip()  # error pages are HTML: keep just the words


def disable(endpoint: str, why: str, secs: int = 600) -> None:
    _OFF[endpoint] = time.time() + secs
    errors.logger().warning("gemini cache off for %d min (%s): %s", secs // 60, endpoint.split("?")[0][-60:], _short(why)[:200])


def wait() -> None:
    """Join background cache builds (used by tests)."""
    for t in list(_THREADS):
        t.join(10)


def _delete(url: str, headers: dict) -> None:
    def go():
        try:
            requests.delete(url, headers=headers, timeout=15)
        except Exception:
            pass
    threading.Thread(target=go, daemon=True).start()


def create_body(cand: dict, system: str, tools, cover: list) -> dict:
    body = {"model": cand["model_ref"], "contents": cover, "ttl": f"{config.GEMINI_CACHE_TTL}s"}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    if tools:
        body["tools"] = tools
    if config.GEMINI_CACHE_KMS_KEY:  # some orgs only allow resources encrypted with their own key (CMEK)
        body["encryptionSpec"] = {"kmsKeyName": config.GEMINI_CACHE_KMS_KEY}
    return body


def try_create(cands: list, endpoint: str, headers: dict, body_for) -> tuple:
    """Post the create request to each candidate mode until one works. -> (name, candidate, error text)."""
    order = sorted(cands, key=lambda c: c["create_url"] != (_GOOD.get(endpoint) or {}).get("create_url"))
    errs = []
    for c in order:
        try:
            r = requests.post(c["create_url"], json=body_for(c), headers=headers, timeout=60)
        except Exception as e:
            errs.append(f"{type(e).__name__}: {e}")
            break
        if r.ok:
            try:
                return r.json()["name"], c, ""
            except Exception as e:
                errs.append(f"bad reply: {e}")
                break
        msg = f"HTTP {r.status_code}: {_short(r.text)[:160]}"
        errs.append(msg)
        if r.status_code != 404:
            break  # the address exists but refuses (policy, permission, bad request): other modes will say the same
    return None, None, errs[-1] if errs else "no candidates"


def hint(err: str) -> str:
    if re.search(r"cmek|kms|restrictNonCmek", err, re.I) and not config.GEMINI_CACHE_KMS_KEY:
        return err + "  -> your org requires CMEK: set HARNESS_GEMINI_CACHE_KMS_KEY=projects/<p>/locations/<l>/keyRings/<r>/cryptoKeys/<k>"
    return err


def _build(cands, endpoint, headers, system, tools, cover, key, old, now) -> None:
    t0 = time.monotonic()
    try:
        name, cand, err = try_create(cands, endpoint, headers, lambda c: create_body(c, system, tools, cover))
        if not name:
            all404 = err.startswith("HTTP 404")
            disable(endpoint, "create failed: " + hint(err), 1800 if all404 else 600)
            return
        _GOOD[endpoint] = cand
        with _LOCK:
            _CACHES[key] = {"name": name, "n": len(cover), "sig": _h(cover), "exp": now + config.GEMINI_CACHE_TTL, "cand": cand}
        errors.logger().info("gemini cache created: %d messages, ~%d tokens, %.1fs (%s)", len(cover),
                             est_tokens([system, tools, cover]), time.monotonic() - t0, cand["create_url"].split("?")[0][-70:])
        if old:
            _delete(old["cand"]["create_url"] + "/" + old["name"].rsplit("/", 1)[-1], headers)
    finally:
        with _LOCK:
            _INFLIGHT.discard(key)


def prepare(cands, headers: dict, system: str, tools, contents: list):
    """(cache_name, n_cached_messages, candidate) to send the request with, or None to send it normally."""
    if not config.GEMINI_CACHE or len(contents) < 3:
        return None
    if cands[0]["label"] != "api-key" and not config.GEMINI_CACHE_VERTEX:
        return None  # Vertex: off unless HARNESS_GEMINI_CACHE=1 (implicit caching still applies)
    endpoint = cands[0]["create_url"]  # identifies the endpoint
    if time.time() < _OFF.get(endpoint, 0):
        return None
    key = (endpoint, _h([system, tools]), _h(contents[0]))
    now = time.time()
    with _LOCK:
        c = _CACHES.get(key)
    valid = bool(c and c["exp"] > now + 30 and c["n"] <= len(contents) - 1 and _h(contents[:c["n"]]) == c["sig"])
    reuse = (c["name"], c["n"], c["cand"]) if valid else None
    if valid and est_tokens(contents[c["n"]:]) < config.GEMINI_CACHE_REFRESH:
        return reuse  # the cache still covers nearly everything
    cover = json.loads(json.dumps(contents[:-1]))  # a snapshot: later history edits must not race the upload
    if est_tokens([system, tools, cover]) < config.GEMINI_CACHE_MIN:
        return reuse  # too small to be worth a cache
    with _LOCK:
        if key in _INFLIGHT:
            return reuse
        _INFLIGHT.add(key)
    args = (cands, endpoint, headers, system, tools, cover, key, c, now)
    if config.GEMINI_CACHE_ASYNC:  # build it in the background: this step goes out now, later steps use the cache
        t = threading.Thread(target=_build, args=args, daemon=True)
        _THREADS.append(t)
        t.start()
        return reuse
    _build(*args)
    with _LOCK:
        c = _CACHES.get(key)
    return (c["name"], c["n"], c["cand"]) if c else reuse


def request_body(body: dict, contents: list, name: str, n: int) -> dict:
    """The request that references the cache: system and tools live in the cache, so they are left out."""
    out = {k: v for k, v in body.items() if k not in ("systemInstruction", "tools", "contents")}
    out["contents"] = contents[n:]
    out["cachedContent"] = name
    return out


def probe(cands: list, headers: dict, say=print) -> bool:
    """Try every mode with a ~6k-token cache, use it once, delete it. Prints what each mode answered."""
    filler = [{"role": "user", "parts": [{"text": ("Anti-money laundering compliance involves monitoring transactions. " * 400) + " The secret key code is OMEGA-778899."}]}]
    working = None
    for c in cands:
        body = create_body(c, "", None, filler)
        try:
            r = requests.post(c["create_url"], json=body, headers=headers, timeout=60)
        except Exception as e:
            say(f"  {c['label']}: network error {e}")
            continue
        if not r.ok:
            say(f"  {c['label']}: HTTP {r.status_code} {hint(_short(r.text)[:200])}")
            continue
        name = r.json().get("name", "")
        say(f"  {c['label']}: created {name}")
        working = c
        try:
            q = {"cachedContent": name, "contents": [{"role": "user", "parts": [{"text": "What is the secret key code?"}]}]}
            u = requests.post(c["call_url"], json=q, headers=headers, timeout=60)
            md = (u.json().get("usageMetadata") or {}) if u.ok else {}
            say(f"    use: HTTP {u.status_code}, cached tokens {md.get('cachedContentTokenCount', 0)} of {md.get('promptTokenCount', '?')}"
                + ("" if u.ok else " " + _short(u.text)[:200]))
        finally:
            requests.delete(c["create_url"] + "/" + name.rsplit("/", 1)[-1], headers=headers, timeout=30)
        break
    return working is not None

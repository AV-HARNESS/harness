"""Mask secrets and other sensitive strings before a request leaves this machine.

Applied at one choke point (providers.generate) to the system prompt, user text, tool results and tool-call
arguments. Files on disk and local session history are NOT modified; only the copy sent to the model is.

This is a heuristic filter: it cuts accidental leakage of credentials and personal data. It does not hide
your source code (that is what the model is for) and cannot spot proprietary or client-specific content
unless you list it in HARNESS_REDACT_TERMS.
"""
import os
import re
import sys
from collections import Counter

from . import config

_SECRET_ENV_NAME = re.compile(r"(KEY|SECRET|TOKEN|PASSW|CREDENTIAL|CLIENT_ID|PRIVATE)", re.I)


def _luhn(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total, alt = total + d, not alt
    return total % 10 == 0


def _assign(m: re.Match) -> str:
    val = m.group(3)
    mixed = val.lower() != val and val.upper() != val
    if any(c.isdigit() for c in val) or (len(val) >= 20 and mixed and "." not in val):
        return f"{m.group(1)}{m.group(2)}[REDACTED:secret]"
    return m.group(0)  # looks like an identifier, not a secret: leave code alone


def _card(m: re.Match) -> str:
    digits = re.sub(r"\D", "", m.group(0))
    return "[REDACTED:card]" if 13 <= len(digits) <= 16 and _luhn(digits) else m.group(0)


# (label, regex, replacement string or function). Order matters: most specific first.
_PATTERNS: list[tuple[str, re.Pattern, object]] = [
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)"),
     "[REDACTED:private_key]"),
    ("api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}|\bsk-[A-Za-z0-9_\-]{20,}|\bgh[pousr]_[A-Za-z0-9]{30,}"
                           r"|\bgithub_pat_[A-Za-z0-9_]{30,}|\bAKIA[0-9A-Z]{16}\b|\bxox[baprs]-[A-Za-z0-9-]{10,}"),
     "[REDACTED:api_key]"),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"), "[REDACTED:jwt]"),
    ("bearer", re.compile(r"(?i)\b(Bearer\s+)[A-Za-z0-9._~+/=-]{20,}"), r"\1[REDACTED:token]"),
    ("url_credentials", re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^/\s:@]+:[^/\s@]+@"), r"\1[REDACTED:credentials]@"),
    ("secret", re.compile(r"(?i)(\b[\w.-]*(?:password|passwd|pwd|secret|token|api[_-]?key|apikey|access[_-]?key"
                          r"|private[_-]?key|credential)[\w.-]*)([\"']?\s*[:=]\s*[\"']?)([^\s\"',;()\[\]{}]{8,})"),
     _assign),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b"), "[REDACTED:email]"),
    ("internal_ip", re.compile(r"\b(?:10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d{1,3}\.\d{1,3}\b"),
     "[REDACTED:internal_ip]"),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED:ssn]"),
    ("card", re.compile(r"\b(?:\d[ -]?){13,16}\b"), _card),
]


_STRICT = [
    ("long_number", re.compile(r"\b\d{9,}\b"), "[REDACTED:number]"),
    ("user_path", re.compile(r"(?i)((?:[A-Z]:[\\/]+Users|/Users|/home)[\\/]+)(?!public\b|shared\b)[^\\/\s\"']+"),
     r"\1[REDACTED:user]"),
]


def _known_secrets() -> list[tuple[str, str]]:
    """Exact values of secret-looking environment variables (includes everything loaded from .env)."""
    found = {v: k for k, v in os.environ.items() if len(v) >= 8 and _SECRET_ENV_NAME.search(k)}
    return sorted(found.items(), key=lambda kv: -len(kv[0]))


def _sub(rx: re.Pattern, repl, text: str) -> tuple[str, int]:
    """regex substitute that counts only matches that were actually changed."""
    n = 0

    def f(m: re.Match) -> str:
        nonlocal n
        out = repl(m) if callable(repl) else m.expand(repl)
        n += out != m.group(0)
        return out

    return rx.sub(f, text), n


def mask(text: str, counts: Counter | None = None) -> str:
    """Return text with sensitive strings replaced by [REDACTED:<type>]."""
    counts = counts if counts is not None else Counter()
    for value, name in _known_secrets():
        n = text.count(value)
        if n:
            text = text.replace(value, f"[REDACTED:env:{name}]")
            counts["env_secret"] += n
    for label, rx, repl in _PATTERNS:
        text, n = _sub(rx, repl, text)
        if n:
            counts[label] += n
    extra = _STRICT if config.REDACT_STRICT else []
    for label, rx, repl in extra:
        text, n = _sub(rx, repl, text)
        if n:
            counts[label] += n
    for i, pat in enumerate(config.REDACT_PATTERNS):
        text, n = re.subn(pat, "[REDACTED:custom]", text)
        if n:
            counts["custom"] += n
    if config.REDACT_TERMS:
        rx = re.compile("|".join(re.escape(t) for t in config.REDACT_TERMS), re.I)
        text, n = rx.subn("[REDACTED:term]", text)
        if n:
            counts["term"] += n
    return text


def _walk(o, counts):
    if isinstance(o, str):
        return mask(o, counts)
    if isinstance(o, dict):
        return {k: _walk(v, counts) for k, v in o.items()}
    if isinstance(o, list):
        return [_walk(v, counts) for v in o]
    return o


def _part(p: dict, counts: Counter) -> dict:
    q = dict(p)  # other keys (thoughtSignature, inlineData, ...) are passed through untouched
    if "text" in q and not q.get("thought"):
        q["text"] = mask(q["text"], counts)
    if "functionResponse" in q:
        fr = q["functionResponse"]
        q["functionResponse"] = {**fr, "response": _walk(fr.get("response", {}), counts)}
    if "functionCall" in q:
        fc = q["functionCall"]
        q["functionCall"] = {**fc, "args": _walk(fc.get("args", {}), counts)}
    return q


def mask_request(contents: list, system: str) -> tuple[list, str, Counter]:
    """Masked copy of a model request. Inputs are not modified."""
    counts: Counter = Counter()
    out = [{**m, "parts": [_part(p, counts) for p in m["parts"]]} for m in contents]
    return out, mask(system or "", counts), counts


def report(counts: Counter) -> None:
    n = sum(counts.values())
    if n:
        detail = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
        print(f"[redact] masked {n} item(s) before sending ({detail})", file=sys.stderr)

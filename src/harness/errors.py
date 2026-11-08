"""User-facing errors: short and plain on screen, full detail (with a reference id) in the log file.

The log is HOME/harness.log. Everything written to it is passed through the redaction masks first.
"""
import logging
import re
import uuid
from logging.handlers import RotatingFileHandler

from . import config

ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_log: logging.Logger | None = None


_PLACEHOLDER = re.compile(r"replace|your[-_ ]|xxxx|changeme|<.*>|\.\.\.$", re.I)


def is_set(value: str | None) -> bool:
    """True for a real-looking value; empty or a copied placeholder (sk-REPLACE-ME, your-key-here) counts as not set."""
    return bool(value and value.strip() and not _PLACEHOLDER.search(value))


class ConfigError(RuntimeError):
    """A setup problem. The message is written for the user and is safe to show as is."""


def env_name(name: str) -> str:
    """Return an env var NAME, or raise. A real key pasted where a name belongs is never echoed back."""
    if not ENV_NAME.match(name or ""):
        raise ConfigError("api_key_env in models.json must be the NAME of an environment variable "
                          "(for example OPENAI_API_KEY), not the key itself. Move the key to your .env file, "
                          "and treat the key you pasted as exposed: rotate it.")
    return name


def logger() -> logging.Logger:
    global _log
    if _log is None:
        _log = logging.getLogger("harness")
        _log.handlers.clear()
        _log.setLevel(logging.INFO)
        try:
            h = RotatingFileHandler(config.HOME / "harness.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8")
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            _log.addHandler(h)
        except OSError:
            pass
        term = logging.StreamHandler()  # also print in the terminal where `serve` / the CLI runs
        term.setFormatter(logging.Formatter("[harness] %(levelname)s %(message)s"))
        _log.addHandler(term)
    return _log


def _safe(text: str) -> str:
    try:
        from . import redact
        return redact.mask(text)
    except Exception:
        return "(unprintable)"


def friendly(e: BaseException) -> str:
    if isinstance(e, ConfigError):
        return str(e)
    name, msg = type(e).__name__, str(e)
    m = re.search(r"HTTP (\d{3})", msg)
    if m:
        code = int(m.group(1))
        if code == 401:
            return "The provider rejected your API key or token (401). Check it in your .env, then restart the harness."
        if code == 403:
            return "Access denied (403): this key or token isn't allowed to use that model. Pick another model or ask for access."
        if code == 404:
            return "That model wasn't found at this endpoint (404). Pick another model, or check the model name and location."
        if code == 429:
            return "Rate limit or quota reached (429). Wait a bit and try again."
        if code == 400:
            return "The provider rejected the request (400)."
        if code >= 500:
            return f"The model provider is having trouble (HTTP {code}). Try again in a moment."
    if name in ("Timeout", "ReadTimeout", "ConnectTimeout") or "timed out" in msg.lower():
        return "The model took too long to answer. Try again."
    if name in ("ConnectionError", "ConnectTimeout", "SSLError", "ProxyError"):
        return "Can't reach the model endpoint. Check your network or VPN, and the endpoint address."
    if "VERTEX_PROJECT is not set" in msg or "not logged in" in msg.lower() or "Not logged in" in msg:
        return msg.split("\n")[0][:200]
    return "Something went wrong."


def report_ref(e: BaseException, context: str = "") -> tuple[str, str]:
    """Log everything; return (short on-screen message, reference id)."""
    ref = uuid.uuid4().hex[:6]
    logger().error("[%s] %s %s: %s", ref, context, type(e).__name__, _safe(str(e)))
    logger().debug("[%s] traceback", ref, exc_info=e)
    return f"{friendly(e)} (ref {ref})", ref


def report(e: BaseException, context: str = "") -> str:
    msg, _ = report_ref(e, context)
    return f"{msg[:-1]}, details in {config.HOME / 'harness.log'})"


def log_lines(ref: str = "", tail: int = 200) -> str:
    """Log text for the UI: lines for one reference id, or the last `tail` lines."""
    try:
        lines = (config.HOME / "harness.log").read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "(no log yet)"
    if ref:
        lines = [l for l in lines if f"[{ref}]" in l]
    return "\n".join(lines[-tail:]) or "(nothing logged for that reference)"

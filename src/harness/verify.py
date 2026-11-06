"""python -m harness verify [alias ...]
Live end-to-end check, run on YOUR machine: config -> auth -> a plain call per model -> a tool-call round trip."""
import time

from . import auth, config, llm

HINTS = [
    ("invalid_client", "Client ID/secret don't match, or the OAuth client type isn't 'Desktop app'."),
    ("invalid_grant", "Refresh token expired or revoked. Run: python -m harness login"),
    ("Token endpoint HTTP", "The org token endpoint rejected the client. Check OAUTH_TOKEN_URL, client id/secret, OAUTH_SCOPE/OAUTH_AUDIENCE (ask your platform team), or try OAUTH_CLIENT_AUTH=basic."),
    ("Not logged in", "Run: python -m harness login   (or set GOOGLE_ACCESS_TOKEN=$(gcloud auth print-access-token))"),
    ("SERVICE_DISABLED", "Enable the 'Vertex AI API' on the project (APIs & Services > Library)."),
    ("PERMISSION_DENIED", "Give your Google account 'Vertex AI User' (roles/aiplatform.user) on the project."),
    ("HTTP 403", "Permission problem: enable Vertex AI API and grant 'Vertex AI User' to your account."),
    ("HTTP 401", "Token rejected. Re-run login, or refresh GOOGLE_ACCESS_TOKEN (they last ~1 hour)."),
    ("HTTP 404", "Model ID or location is wrong. Check GEMINI_*_MODEL and VERTEX_LOCATION ('global' or e.g. 'us-central1')."),
    ("is not set", "Set that environment variable (see README)."),
]
ECHO = [{"functionDeclarations": [{
    "name": "echo", "description": "Echo text back.",
    "parameters": {"type": "OBJECT", "properties": {"text": {"type": "STRING"}}, "required": ["text"]}}]}]
ok_all = True


def check(label: str, fn) -> bool:
    global ok_all
    t = time.time()
    try:
        detail = fn() or ""
        print(f"  PASS  {label} ({time.time() - t:.1f}s) {detail}")
        return True
    except Exception as e:
        ok_all = False
        msg = str(e)
        print(f"  FAIL  {label}: {msg[:400]}")
        for key, hint in HINTS:
            if key in msg:
                print(f"        -> {hint}")
                break
        return False


def _ask(alias: str, text: str) -> str:
    c, _, _ = llm.generate(alias, [{"role": "user", "parts": [{"text": text}]}], retries=1)
    return llm._text(c).strip()


def _tool_round_trip(alias: str) -> str:
    hist = [{"role": "user", "parts": [{"text": "Call the echo tool with text 'hello-harness', then tell me what it returned."}]}]
    c1, _, _ = llm.generate(alias, hist, "Use tools when asked.", ECHO, retries=1)
    calls = [p["functionCall"] for p in c1["parts"] if "functionCall" in p]
    if not calls:
        raise RuntimeError(f"model did not call the tool; it said: {llm._text(c1)[:200]!r}")
    hist += [c1, {"role": "user", "parts": [{"functionResponse": {"name": calls[0]["name"], "response": {"output": "hello-harness"}}}]}]
    c2, _, _ = llm.generate(alias, hist, "Use tools when asked.", ECHO, retries=1)
    out = llm._text(c2)
    if not out:
        raise RuntimeError("no final answer after the tool result")
    return f"-> {out[:60]!r}"


def main(aliases: list[str]) -> int:
    global ok_all
    ok_all = True  # reset: the flag is module-level and must not leak between runs
    aliases = aliases or list(config.MODELS)
    def mode(a):
        sp = config.MODELS[a]
        return "gemini/" + llm.gemini_mode(sp) if sp.get("provider", "gemini") == "gemini" else sp["provider"]
    print(f"Vertex project: {config.PROJECT or '(not set)'}  location: {config.LOCATION}"
          f"{'  host: ' + config.HOST if config.HOST else ''}")
    print("Models: " + ", ".join(f"{a}={mode(a)}:{config.MODELS[a]['model']}" for a in aliases if a in config.MODELS))
    for a in aliases:
        if a not in config.MODELS:
            print(f"  FAIL  unknown alias '{a}'. Configured: {', '.join(config.MODELS)}")
            return 1
    if any(config.MODELS[a].get("provider", "gemini") == "gemini" and llm.gemini_mode(config.MODELS[a]) == "vertex"
           for a in aliases):
        print("\n[Google auth (Vertex)]")
        check("project set", lambda: config.PROJECT or (_ for _ in ()).throw(RuntimeError("VERTEX_PROJECT is not set")))
        check("access token", lambda: ("(static GOOGLE_ACCESS_TOKEN)" if config.ACCESS_TOKEN else "(client-credentials token)" if config.TOKEN_URL else "(refreshed from stored login)")
              if auth.get_access_token() else "")
    for a in aliases:
        print(f"\n[{a}]")
        if check("plain call", lambda: f"-> {_ask(a, 'Reply with exactly the word: pong')[:40]!r}"):
            check("tool-call round trip", lambda: _tool_round_trip(a))
    print("\nALL CHECKS PASSED" if ok_all else "\nSome checks failed; see hints above.")
    return 0 if ok_all else 1

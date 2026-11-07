# Gemini harness

Agent loop over Vertex AI Gemini (flash + pro) using your OAuth client ID/secret. Python CLI + local web UI.
Everything is organised into projects, stored locally under ~/.harness/projects/<id>/:
    project.json  name + root folder (the agent reads/edits files directly there) + default model
    blocks/*.md   persistent context notes, loaded into every chat in the project (edit by hand or let the agent write them)
    sessions/     full chat histories
Deleting a project removes blocks and sessions only, never the linked root folder.
No approval prompts: tools run immediately, so point root at a git repo to review/revert what the agent changes.

## Getting a Gemini (Vertex) token
Quickest test, no OAuth client needed (token lasts ~1 hour):
    gcloud auth login
    export GOOGLE_ACCESS_TOKEN=$(gcloud auth print-access-token)
    export VERTEX_PROJECT=your-project
Durable setup with your own OAuth client:
 1. Google Cloud console, your project: enable the "Vertex AI API".
 2. Your Google account needs the role "Vertex AI User" (roles/aiplatform.user) on the project.
 3. APIs & Services > OAuth consent screen: configure it and add yourself as a test user.
 4. Credentials > Create credentials > OAuth client ID > type "Desktop app". Copy the client ID and secret.
    (If your org already gave you an ID/secret, skip 3-4. A "Web" client needs `login --port <n>` with
    http://localhost:<n>/ registered as a redirect URI.)
 5. Set GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, VERTEX_PROJECT, then: python -m harness login
 6. Check everything: python -m harness verify

## Which login do I have? (client ID + secret)
- Client ID ends in .apps.googleusercontent.com -> a Google OAuth client: `python -m harness login` (browser, once).
- Your org gave you a token URL / gateway (Helix etc.) -> client-credentials, no browser. Set:
    OAUTH_TOKEN_URL=<org token endpoint>   GOOGLE_CLIENT_ID=...  GOOGLE_CLIENT_SECRET=...
    VERTEX_HOST=<gateway base URL>         VERTEX_PROJECT / VERTEX_LOCATION as the gateway expects
    optional: OAUTH_SCOPE, OAUTH_AUDIENCE, OAUTH_CLIENT_AUTH=basic
  The gateway must expose Vertex-style generateContent paths; if its URL layout differs, tell me and I'll adapt it.
- Just a Gemini API key (AI Studio) -> set GEMINI_API_KEY (and leave VERTEX_PROJECT unset). No login at all.
- Any single model can override the above in models.json: api_key_env, base_url, project, location, headers.

## Other models
Copy models.example.json to ~/.harness/models.json and edit. Each key is an alias shown in the UI/CLI.
    provider "gemini"     Vertex Gemini (uses the Google login above)
    provider "openai"     any OpenAI-compatible /chat/completions API: OpenAI, OpenRouter, Ollama, vLLM, LiteLLM, company gateways
    provider "anthropic"  Anthropic Messages API
Keys come from the env var named in api_key_env. Entries whose model starts with REPLACE are ignored.
History is stored in one neutral format, so you can switch models mid-session (safest between turns).
web_search needs a Gemini model configured; other tools work with every provider.

## Setup
    pip install -r harness/requirements.txt          # run from the folder containing harness/
    export GOOGLE_CLIENT_ID=...  GOOGLE_CLIENT_SECRET=...
    export VERTEX_PROJECT=your-project  VERTEX_LOCATION=global   # or us-central1 etc.
    python -m harness login        # one-time browser consent; stores refresh token
    python -m harness verify       # live test: auth, each model, tool calling

## Use
    python -m harness              # terminal chat   (/help for commands)
    python -m harness serve        # web UI at http://127.0.0.1:8765
    python -m harness chat --project <id> | --resume <session-id>
    python -m harness projects
    CLI: /projects /project <id> /project new <name> [root] /root <path> /blocks /block <name> /new /sessions /resume

## Optional env
    GEMINI_FLASH_MODEL / GEMINI_PRO_MODEL   model IDs (defaults gemini-2.5-flash / gemini-2.5-pro)
    VERTEX_HOST          gateway base URL if your org fronts Vertex
    GOOGLE_REFRESH_TOKEN skip browser login;  GOOGLE_ACCESS_TOKEN  static bearer token
    HARNESS_WORKDIR      default root for new projects (default: cwd)
    HARNESS_COMPACT_AT   auto-summarize when prompt tokens exceed this (default 600000)

## Layout
    auth.py OAuth | llm.py providers (gemini/openai/anthropic) | verify.py | tools.py bash/files/web | agent.py loop + compaction
    store.py projects/blocks/sessions | cli.py | server.py + static/index.html

## Privacy: what is masked before a model call
Everything sent to a model passes through `redact.py` first: the system prompt, your messages, tool results and
tool-call arguments. It masks API keys, tokens, JWTs, private keys, `password=`/`secret=` style values, URL
credentials, emails, internal IPs, SSNs and card numbers, the exact values of secret-looking environment variables
(including everything in your `.env`), and any words in `HARNESS_REDACT_TERMS`. Masked values appear to the model as
`[REDACTED:type]`; files on disk and your local history are not changed, and the model cannot write placeholders back.
Sensitive files (`.env`, `*.pem`, `*.key`, `token.json`, `.harness/` ...) are blocked from the file tools.
A one-line note on stderr says how many items were masked; it never prints the values.

Limits: your source code is still sent (that is the point of the tool), and proprietary or client-specific content is
only caught if you list it in `HARNESS_REDACT_TERMS`. Masking does not hide the calls themselves from your network or gateway.
Preview it: `python -m harness redact "some text"`. Tests: `PYTHONPATH=src python -m pytest tests`.

### See exactly what leaves your machine
- `HARNESS_DRY_RUN=1`: builds and masks the request, prints it to the terminal, sends nothing (no credentials needed).
- `HARNESS_AUDIT_LOG=1`: every request is also written, as sent, to `~/.harness/audit.jsonl` (or give a path). Open it to see what a gateway log would contain.
- `HARNESS_REDACT_STRICT=1`: also masks 9+ digit numbers and user names in paths. `HARNESS_REDACT_PATTERNS='ACME-\d{6}||cust_[a-z]+'` masks your own patterns.

## Spec and tests: how to trust this without reading code
- **`SPEC.md`** lists everything the harness promises, one ID per line (`R-LOOP-01`, `R-PRIV-06`, ...).
- **`tests/`** has at least one test per ID (`@covers("R-...")`). `tests/test_spec.py` fails if a spec line has no test, or a test points at an ID that isn't in the spec.
- Items marked `[manual]` need live credentials; run `python -m harness verify` for those.
- Tests use fake local servers only: no network, no real keys, no `~/.harness`.

```
pip install -e ".[dev]"
pytest                                   # all tests, ~20s
pytest -k R_PRIV -v                      # or: pytest tests/test_redact.py
git config core.hooksPath .githooks      # once per clone: runs pytest before every git push
```
GitHub Actions (`.github/workflows/tests.yml`) runs the same suite on every push and PR, on Python 3.11 and 3.12. A red X on a commit means a promise in SPEC.md broke.

Changing behaviour: edit `SPEC.md` and the test in the same commit.

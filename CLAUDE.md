# CLAUDE.md

Personal agent harness (loop + tools around an LLM). Python CLI + local web UI. Models: Gemini (Vertex or API key),
OpenAI-compatible, Anthropic. Keep it small and readable.

## Run (from repo root, which contains `harness/`)
```
python3 -m harness serve          # web UI at http://127.0.0.1:8765
python3 -m harness                # CLI chat
python3 -m harness verify         # live end-to-end check (auth + each model + tool call)
python3 -m harness login          # one-time Google OAuth login (Vertex only)
python3 -m harness projects       # list projects
```

## File layout

```
harness/
  config.py          env vars + model aliases (~/.harness/models.json)
  store.py           projects / blocks / sessions as plain files in ~/.harness/projects/<id>/
  auth.py            static token | client-credentials (OAUTH_TOKEN_URL) | Google browser login
  agent.py           agent loop (model → tools → results), compaction, system prompt
  tools.py           bash, read/write/edit_file, list_dir, web_fetch, web_search, block tools
  llm.py             thin shim — re-exports from providers/ for backwards compatibility
  providers/
    __init__.py      shared utilities (_post, _text, …) + generate() entry point
    gemini.py        Gemini provider (Vertex AI or API key)
    openai.py        OpenAI-compatible provider (/chat/completions)
    anthropic.py     Anthropic Messages API provider
  server.py          FastAPI + SSE API + static file serving
  cli.py             terminal REPL with /commands
  verify.py          live check: auth → plain call → tool-call round trip, per model
  static/
    index.html       single-page web UI
  __main__.py        entry point (python3 -m harness)
```

## Adding a provider
1. Add `harness/providers/<name>.py` with `generate(spec, contents, system, tools, retries)`.
2. Register it in `providers/__init__.py`'s `fn` dict.
3. Add `"provider": "<name>"` to a model entry in `~/.harness/models.json`.

## Key decisions
- History in Gemini `contents` format; every provider translates to/from it so sessions are portable.
- Project = root folder + context blocks (markdown notes) + sessions. Deleting a project never deletes the root.
- Local storage only (`~/.harness/`). Never commit secrets, tokens, `.env`, or `~/.harness`.
- No approval prompts or sandbox, by the owner's choice. Server binds to 127.0.0.1 only (no auth needed).
- Config via env vars only; never log secrets.
- Privacy: `redact.py` masks secrets/PII in every model request (hook is `providers.generate`); `tools._guard` blocks sensitive files. On by default; keep it that way. See `config.py` for the full list.
- Context compaction: when `prompt_tokens > HARNESS_COMPACT_AT`, the agent summarises history in-place.

## Spec and tests (source of truth)
- `SPEC.md` = what the harness promises (IDs `R-AREA-NN`). `tests/` verifies every ID via `@covers("R-...")`; `tests/test_spec.py` enforces the mapping.
- Behaviour change => update SPEC.md + tests in the same commit. Run `pytest` before pushing (CI runs it on every push; `.githooks/pre-push` too).
- Tests never touch network, real keys, or `~/.harness` (see `tests/conftest.py`).

## Status
Automated tests use fake local servers only. NOT yet run against real Vertex, OpenAI, Anthropic, or the org gateway.
First step: run `python3 -m harness verify` and fix what real APIs reveal (model IDs, URL layout, schema quirks).

## Backlog
2. Streaming responses
3. Load project `AGENTS.md`/`CLAUDE.md` into the system prompt
4. Task-list tool, sub-agents, MCP
5. Stop only works between steps
6. Browser-extension side panel

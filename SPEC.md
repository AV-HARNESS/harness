# Harness Spec

What the harness promises. Each requirement has an ID. Every ID is checked by at least one automated test
(`tests/`, tagged `@covers("R-...")`), and `tests/test_spec.py` fails if an ID has no test or a test points at an
ID that does not exist. If it is not here, it is not promised. `[manual]` = needs live credentials, cannot run in CI
(use `python -m harness verify`).

Change behaviour => change this file and the test in the same commit.

## 1. Agent loop (`agent.py`)
- **R-LOOP-01** A turn sends history to the model, runs every tool call it returns, feeds results back, and repeats until the model answers without tool calls.
- **R-LOOP-02** A turn stops after `HARNESS_MAX_STEPS` model calls (default 200) and tells the user to say "continue".
- **R-LOOP-03** A stop request ends the turn before the next model call (not mid-call).
- **R-LOOP-04** Tool output stored in history is trimmed to `HARNESS_HISTORY_TOOL_TRIM` chars; the UI event still gets the full output.
- **R-LOOP-05** The system prompt is rebuilt on every step with project name, root, all context blocks (per-block and total size caps) and session instructions, so block edits apply immediately.
- **R-LOOP-11** If the project root has `AGENTS.md` (or `CLAUDE.md`) it is added to the system prompt and the agent is told to keep it updated; if neither exists the agent is told to create `AGENTS.md` once it understands the project.
- **R-LOOP-12** Compaction never repeats on every step: after one, the next waits until the prompt has grown ~25% past its size right after it; and when one long turn is most of the context, it is cut inside the turn (keeping the last few steps, never splitting a tool call from its result).
- **R-LOOP-06** Model "thought" parts are never shown to the user, and are kept verbatim in history (thought signatures survive).
- **R-LOOP-07** When the prompt exceeds `HARNESS_COMPACT_AT`, older turns are replaced by a summary and the last `HARNESS_KEEP_TURNS` user turns are kept verbatim.
- **R-LOOP-08** A new session is titled from the first user message; the session is saved after every step.
- **R-LOOP-10** A message queued while the agent works is delivered at its next step (with the tool results, or as a new user message if the agent was about to finish), and the agent keeps going; if the run is stopped first, the queued messages are handed back to the caller.
- **R-LOOP-09** A turn that ends in an error leaves the user's message saved in the session.

## 2. Model providers (`providers/`)
- **R-PROV-01** Gemini with an API key: calls `<base>/models/<model>:generateContent` with `x-goog-api-key`.
- **R-PROV-02** Gemini on Vertex: calls `.../projects/<p>/locations/<l>/publishers/google/models/<m>:generateContent` with a Bearer token; `VERTEX_HOST` / per-model `base_url` point it at a gateway; `headers` are added.
- **R-PROV-03** Retryable statuses (429, 5xx) are retried with backoff; other errors raise with status and body.
- **R-PROV-04** OpenAI-compatible: translates history (tools, tool results) to `/chat/completions` and back; missing key env var gives a clear error.
- **R-PROV-05** Anthropic: translates to `/v1/messages` (tool_use / tool_result, alternating roles merged) and back; `api_key_env` is required.
- **R-PROV-06** Empty or missing model output yields a placeholder text, never a crash; no candidates raises.
- **R-PROV-07** Unknown model alias or provider gives an error listing valid choices.
- **R-PROV-08** History is provider-neutral (Gemini format): the same session can continue on a different provider.
- **R-PROV-10** Vertex location is chosen per model: alias `location`, else `MODEL_LOCATIONS` entry for that model, else `VERTEX_LOCATION`.
- **R-PROV-11** A model alias of the form `source/model-id` means the configured alias `source` with that model id, so any model an endpoint offers can be used without editing `models.json`.
- **R-PROV-12** Keys come only from `.env`: built-in `openai`/`claude`/`flash`/`pro` aliases use whichever provider keys are set (so every provider you have a key for appears); a `models.json` entry without `api_key_env` uses its provider's standard key variable, and an entry holding a pasted key is ignored with a warning.
- **R-PROV-09** `models.json` adds aliases; entries whose model starts with `REPLACE` or alias starts with `_` are ignored.

## 3. Auth (`auth.py`)
- **R-AUTH-01** `GOOGLE_ACCESS_TOKEN` is used as-is when set.
- **R-AUTH-02** With `OAUTH_TOKEN_URL`, a client-credentials token is fetched (secret in body, or Basic header when `OAUTH_CLIENT_AUTH=basic`; scope and audience sent when set).
- **R-AUTH-03** The client-credentials token is cached until shortly before expiry.
- **R-AUTH-04** Missing client id/secret, a rejected token request, or no login each give a clear, actionable error.
- **R-AUTH-05** `is_logged_in` reflects whether any usable credential source is configured.
- **R-AUTH-06** [manual] Browser login (`python -m harness login`) and real refresh-token exchange.

## 4. Storage (`store.py`)
- **R-STORE-01** Projects get unique slug ids; the root folder must exist.
- **R-STORE-02** Deleting a project deletes its blocks and sessions, never the root folder.
- **R-STORE-03** Context blocks: create, append, read, list, delete; names restricted to letters, digits, `-`, `_`.
- **R-STORE-04** Sessions persist full history and round-trip unchanged; listing is newest first.
- **R-STORE-05** Ids and names cannot escape the storage folder (path traversal rejected).
- **R-STORE-06** First run creates a `default` project and migrates legacy sessions into it.
- **R-STORE-09** All projects, blocks and sessions live in one central folder, `HARNESS_HOME` (default `<repo>/.harness`; `~` is expanded), never inside your project folders; the server and CLI print its path at start.
- **R-STORE-08** Files are written atomically, so a reader never sees a half-written session; an empty or partial file is retried briefly before it counts as an error.
- **R-STORE-07** Updating a project changes name, root (validated) and model (only known aliases).

## 5. Tools (`tools.py`)
- **R-TOOL-01** `bash` runs in the project root and returns exit code, stdout, stderr; honours timeout.
- **R-TOOL-02** Relative file paths resolve against the project root.
- **R-TOOL-03** `edit_file` replaces an exact string; errors if missing or ambiguous unless `replace_all`.
- **R-TOOL-04** Tool errors and unknown tools are returned to the model as text, never raised.
- **R-TOOL-05** Block tools read and write the active project's blocks.
- **R-TOOL-06** `web_fetch` returns readable text for HTML (scripts/styles dropped) and raw text otherwise.
- **R-TOOL-07** `web_search` needs a Gemini model and returns answer plus sources; says so otherwise.
- **R-TOOL-08** Very long output is clipped in the middle.

- **R-TOOL-09** File tools (read, write, edit, list) work only inside the linked project folder: `..`, absolute paths and symlinks that lead outside are refused (`HARNESS_CONFINE=0` turns this off). `bash` starts in the folder but is not sandboxed.
- **R-TOOL-10** All files are read and written as UTF-8 regardless of the OS default (emoji and non-English text work on Windows).

## 6. Privacy (`redact.py`, `tools.py`)
- **R-PRIV-01** Before any model call, API keys and tokens (Google, OpenAI/Anthropic style, GitHub, AWS, Slack, JWT, private keys, bearer headers) are replaced by `[REDACTED:kind]`.
- **R-PRIV-02** Password / secret / token style assignments are masked; ordinary code is left alone.
- **R-PRIV-03** URL credentials, emails, internal IPs, SSNs and Luhn-valid card numbers are masked.
- **R-PRIV-04** Values of secret environment variables, and terms in `HARNESS_REDACT_TERMS`, are masked wherever they appear.
- **R-PRIV-05** Masking applies to a copy: message structure, tool calls and thought signatures are preserved; local files and saved history keep the real text.
- **R-PRIV-06** The model cannot read, write or edit sensitive files (`.env`, keys, `token.json`, `credentials*`, anything in `.harness`, plus `HARNESS_BLOCK_FILES`); `.example/.sample/.template` files are allowed.
- **R-PRIV-07** The model cannot write `[REDACTED:` placeholders into files.
- **R-PRIV-08** `HARNESS_REDACT=0` turns masking off.
- **R-PRIV-09** Every provider path (Gemini, OpenAI, Anthropic, web_search, compaction, verify) goes through the same masking choke point.
- **R-PRIV-11** `HARNESS_REDACT_STRICT=1` also masks 9+ digit numbers and user names inside paths; default off.
- **R-PRIV-12** `HARNESS_REDACT_PATTERNS` (regexes separated by `||`) are masked as `[REDACTED:custom]`.
- **R-PRIV-13** `HARNESS_AUDIT_LOG` records every outgoing request as sent (masked system prompt, messages, tool names, mask counts), one JSON line each; it never contains the real values of masked secrets or any credential.
- **R-PRIV-14** `HARNESS_DRY_RUN=1` builds and masks the request, logs or prints it, and does not call the model or need credentials.
- **R-PRIV-10** [manual] Confirm in the org's LLM log viewer that no secret appears in a captured request.

## 7. Web server (`server.py`)
- **R-SRV-01** Binds to 127.0.0.1 only.
- **R-SRV-02** Project, block and session CRUD endpoints work; unknown ids return 404, bad input 400.
- **R-SRV-03** `POST /api/sessions/{id}/chat` streams server-sent events (text, tool_call, tool_result, usage, done); failures arrive as an `error` event then `done`.
- **R-SRV-04** A session already running rejects a second chat with 409.
- **R-SRV-05** `/api/config` reports models, default model and login state; never returns secrets.
- **R-SRV-06** Deleting a project via the API leaves its root folder intact.

- **R-SRV-07** `GET /api/fs` lists folders (and with `files=1` non-sensitive files) so the UI can link a folder; hidden entries are skipped.
- **R-SRV-08** With `project=<id>`, `/api/fs` stays inside that project's root (used for attaching files).
- **R-SRV-10** `POST /api/sessions/<id>/queue` adds a message to a running session (409 if it isn't running); unsent queued messages are returned in a `leftover` event so the UI can send them next.
- **R-SRV-11** `GET /api/models` lists, per configured endpoint (OpenAI-compatible, Gemini API key, Anthropic), the chat models it offers; a failing endpoint is reported as an error, not raised. The UI shows them in the model picker automatically.
- **R-SRV-14** Errors shown to the user are short and plain with a reference id; the full detail goes to `harness.log` in the data folder (masked). A key pasted where an env var NAME belongs is never echoed to the screen, the API or the log, and tells the user to rotate it.
- **R-SRV-16** Sessions never mix: concurrent runs in different projects or sessions keep separate histories and write files only into their own project folder; a running session is one shared live object, so refreshing, renaming or a second tab sees the true state and no edit is lost; compacting or deleting a running session is refused.
- **R-UI-05** Each browser tab keeps its own project and session (in the URL), run/queue state is per session, a response only paints into the session it belongs to, and a slow earlier click never overwrites a later one; a session running elsewhere is followed live.
- **R-SRV-15** If a chat's model is unavailable on this machine, the chat switches to the default (or first usable) model and says so; a 401/403/404 from the provider marks that model unusable immediately; with no usable model at all the error says which key to add.
- **R-SRV-13** `/api/config` lists only model aliases whose credentials exist in this environment (key env var set, or Vertex project + login); the others come back in `hidden_models` with the reason, and the default model falls back to the first usable alias.
- **R-SRV-12** Listed models are probed once with a tiny request in the background and only confirmed ones are offered: 401/403/404 hides a model, other failures show it as "unverified"; results are cached on disk (keyed by a hash of endpoint+key) and `HARNESS_MODEL_ALLOW` limits which model ids are considered at all.
- **R-SRV-09** `GET /api/places` returns the picker's sidebar: existing Home/Desktop/Documents/Downloads, the folders of your projects, and drives on Windows.
- **R-UI-01** The page has the Claude-style composer: a `+` menu (link folder, attach files, instructions, voice language), the model picker, a dictation mic, a voice-mode button and send; a Finder-style folder picker (places sidebar, breadcrumb, back/forward, search, list with dates); a dictation bar with live waveform and cancel (✕) / accept (✓) buttons; and the linked folder shown above the box.
- **R-UI-03** [manual] In a browser: consecutive tool steps collapse into one summary line ("Read 2 files, ran 1 command") that expands to the individual steps, each with its own output; a text reply ends the group.
- **R-UI-04** Dictation and voice mode work while the agent is running: what you say fills the box and sending it queues the message (R-LOOP-10); the mic is not disabled by a running turn.
- **R-UI-02** [manual] In a browser: dictation shows a live waveform with ✕ (discard) and ✓ (keep text), fills the box live and waits for you to send; voice mode sends when you stop talking and reads the reply aloud; the folder picker links a folder and the agent then works there; attached files appear as chips.

## 8. CLI and verify (`cli.py`, `verify.py`)
- **R-CLI-01** `python -m harness redact "<text>"` prints the masked text and a report.
- **R-CLI-03** `python -m harness` run inside a folder opens the project whose root contains that folder (deepest match), or creates one for it; `--project` / `--resume` override this.
- **R-CLI-02** `python -m harness projects` lists projects.
- **R-VER-01** `verify` exits 0 only when every check passes and prints an actionable hint for known failures.
- **R-VER-02** [manual] `verify` against real Vertex / gateway / OpenAI / Anthropic.

## 9. Repo hygiene
- **R-REPO-01** `.gitignore` excludes `.env`, `.harness/`, token files; no such file is tracked.
- **R-REPO-02** `.env.example` and `models.example.json` contain no real-looking secrets.
- **R-REPO-04** `scripts/check_secrets.py` (run by the pre-commit and pre-push hooks and in CI) fails on secret-looking text, on `.env`/`token.json`/`models.json` files, and on any term in your private `~/.harness-denylist`; test fixtures are exempt.
- **R-REPO-03** Every requirement ID above is covered by a test (enforced by `tests/test_spec.py`).

## Not promised (non-goals)
No tool approval prompts or sandbox (owner's choice). No auth on the web server (localhost only). Redaction is
pattern-based and best effort: it cannot catch a secret that looks like ordinary text. Streaming, sub-agents and
MCP are not implemented.

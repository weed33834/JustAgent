# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed

- **Dead `resources/database.py` gateway (644 lines).** Hand-rolled
  connection pool + five database backends (SQLite/MySQL/PostgreSQL/Redis/
  MongoDB) had zero runtime callers — the knowledge ETL uses stdlib
  `sqlite3` directly. Barrel exports removed from `justagent.resources`;
  re-introduce a DB layer via SQLAlchemy only when a real consumer appears.

### Changed

- **Lint debt cleared; CI now gates on ruff.** All ~80 ruff violations fixed
  across `src` and `tests`; new `lint` job runs `ruff check src tests`
  (blocking) plus a non-blocking `mypy` report (type coverage tracked
  separately, currently ~71 errors in 18 files).
- **Web console: evidence audit entry point.** The evidence panel gains an
  "证据链审计" button that calls `POST /api/judicial/evidence/audit` and
  reports verdict, per-category issues, and claim coverage in chat.

### Added

- **Legal vertical: full evidence-chain audit (M4).** New
  `EvidenceAuditor` performs four deterministic, LLM-free checks on top of
  the existing chain analysis:
  * **Custody integrity** — official collection methods (扣押/调取/提取/查封)
    require a custody trail (`Evidence.custody_chain`, new `CustodyEvent`
    model); chain dates must be well-formed, ordered, and not in the future.
  * **Timeline consistency** — malformed/future collection dates; evidence
    collected after the case filing date is flagged.
  * **Source independence** — support/corroborate relations whose ends cite
    the same `source_document_id` do not count as independent corroboration.
  * **Claim coverage** — every litigation claim is mapped to admissible
    evidence via discriminative-term overlap over proving objects (document-
    frequency filter removes generic words like 「合同」「被告」).
  Each audit ends in a three-tier verdict: 通过 / 有瑕疵 / 严重缺陷.
- **CLI**: `justagent judicial evidence audit CASE_ID [--format rich|json|markdown] [-o FILE]`.
- **Web**: `POST /api/judicial/evidence/audit` returns the full audit result.
- **Demo**: `examples/legal-sample-case/sample_case.py` — a realistic
  卷宗 fixture with four planted defects; runs without an LLM and prints the
  audit report.

### Changed

- `_tokenize` stop-word list extended with litigation-structure words
  （被告/原告/当事人/双方/事实/判令…）to prevent false claim-evidence matches.

### Changed

- **Scheduler: hand-rolled cron engine replaced with `croniter`.** The
  ~120-line custom field-matcher and minute-by-minute next-run scan are gone;
  cron semantics (including weekday-7 aliasing and impossible-date detection)
  now come from the battle-tested `croniter` library. `ScheduleParser`
  public API and persisted task shape are unchanged.
- **README repositioning (M3, en/zh/ja).** Lead narrative is now the engine —
  "an auditable multi-agent platform" (permission-checked actions, shadow-git
  checkpoints, audit logging) — with JustAgent Legal presented as a bundled
  vertical application. Package description updated to match.
- **Engine/vertical layering (M2).** The legal domain package moved from
  `justagent/judicial/` to `justagent/verticals/legal/`; its agent tool moved
  out of the engine built-ins (`agent/tools/builtin/judicial.py` →
  `verticals/legal/agent_tool.py`) and its CLI module out of
  `cli/commands/`. The engine now discovers vertical tools and CLI commands
  exclusively through `justagent.tools` / `justagent.cli` entry points — it
  holds zero direct imports or domain vocabulary for any vertical.
  `web/app.py` remains the single composition root that mounts vertical
  routes. Tests moved accordingly (`tests/judicial/` → `tests/legal/`).
- New CI job `layer-check` enforces the boundary: no vertical imports and no
  legal-domain vocabulary in engine modules (`scripts/layer-check.sh`).

## [3.1.0] - 2026-08-22

### Security

- **Web console: enforced authentication (default-deny).** All `/api/*` endpoints
  now require a credential — either the `JUSTAGENT_WEB_TOKEN` shared token or a
  session issued via `/api/auth/login`. Previously, an unset
  `JUSTAGENT_WEB_TOKEN` left every endpoint anonymous. An explicit escape hatch
  (`justagent web --no-auth`, or env `JUSTAGENT_WEB_NO_AUTH=1`) restores the old
  behavior for local development and prints a red warning.
- **Web console: PBKDF2 password hashing.** User passwords are now hashed with
  PBKDF2-HMAC-SHA256 at 600k iterations (was: a single HMAC-SHA256 pass).
  Legacy hashes verify transparently and are re-hashed on the next successful
  login — no password reset required.
- **Web console: persistent sessions.** Session tokens are stored in
  `~/.justagent/web_sessions.json` (atomic writes), so browser sessions survive
  process restarts. Write failures degrade to memory-only mode.

### Fixed

- **`doctor` crashed on import.** v3.0.2 renamed the cache class (`DiskCache`
  → `Cache`) but missed `cli/commands/doctor.py`; `justagent doctor` failed to
  load.
- **`core.cache.Cache` was uninstantiable with defaults.** The diskcache
  backend received an invalid `eviction_policy="LRU"` (KeyError); corrected to
  `least-recently-used`.
- **`core.cache.Cache.invalidate()` raised TypeError.** diskcache's `delete()`
  takes no `ignore_missing` argument; invalidating a missing key is now a true
  no-op as documented.
- **CI action bumps.** `actions/cache` 4→6 (#5), `astral-sh/setup-uv` 5→7 (#4).

### Added

- Initial public release.

### Changed

- **LLM gateway: LiteLLM → official OpenAI SDK.** All supported providers
  (OpenAI, Azure, OpenRouter, Ollama, vLLM, LM Studio, llama.cpp) expose
  OpenAI-compatible endpoints, so the heavy `litellm` dependency is replaced
  by the lighter official `openai` client with a per-provider `base_url`.
  Affects `unified_gateway.py`, `agent/runtime.py`, `knowledge/vector.py`;
  drops `litellm` from `pyproject.toml`, adds `openai`.
- **Fast-fail for key-required backends.** Remote providers (OpenAI,
  OpenRouter, Azure) without a configured API key now fail immediately instead
  of hanging on a doomed network call — e.g. `justagent fix` with no key exits
  cleanly instead of timing out.
- **Skip PyPI/wheel integration tests.** Distribution to PyPI is out of scope,
  so the sdist/wheel install and PyPI upload tests are skipped (they required
  network installs and publishing tooling).

### Removed

- **PyPI publishing tooling.** `twine` and `build` dropped from the `dev`
  extras — the project does not publish to PyPI. PyPI package/upload tests are
  skipped accordingly.

### Added

- **Legal research memo (`judicial research`).** Given a legal question, it
  retrieves relevant articles from the legal knowledge base and drafts a
  structured research memo (issue → legal basis → analysis → conclusion),
  using the LLM when a backend is configured (graceful fallback otherwise).
  This showcases the judicial positioning — automating legal research.
- **Web doc-type selector.** `/api/judicial/doc/types` lists all legal
  document templates; the web lets you pick a type and generate it for any
  case.

- **Runtime project/workspace switching.** The web can now switch between
  managed projects at runtime: a sidebar dropdown selects the active project,
  and all judicial endpoints resolve the state from that project
  (`X-JustAgent-Project` header → `<project>/.justagent/judicial_state.json`).
  `POST /api/projects` registers a project. Verified: two projects keep
  independent state.

- **Email notifications + reports + more charts + projects.**
  - Email notifications via stdlib `smtplib` (SMTP_SSL) with
    `JUSTAGENT_SMTP_*` env vars.
  - `/api/report` returns a printable HTML judicial summary (browser can
    print / save as PDF).
  - `/api/projects` lists managed projects + current project root.
  - System panel gained case-by-status pie, law-by-domain bar charts and a
    projects section.

- **Full frontend panel coverage of web endpoints.** Every backend capability
  now has a UI entry: audit log panel, legal knowledge search, evidence-chain
  analysis button, document generation per case, judicial state export
  (JSON), and session delete. (All 30+ `/api` endpoints are reachable from the
  browser.)

- **Multimodal image recognition.** `/api/vision` uses OpenAI multimodal
  content blocks to analyze images (needs a vision-capable model); the chat
  bar has an image button.

- **Multi-user accounts & roles.** `justagent web` now supports user accounts
  (stored in `data/users.json`, default `admin` account auto-created). Roles:
  `admin` / `editor` / `viewer`. Login at `/api/auth/login` issues a session
  token; `viewer` is read-only, `editor` can write, `admin` can manage users.
  `JUSTAGENT_WEB_TOKEN` shared-token auth still works as a fallback.

- **Web UX enhancements (browser-native + external libs).**
  - **Voice input/output**: mic (Web Speech API `SpeechRecognition`) and TTS
    speak (browser `speechSynthesis`) — no backend dependency.
  - **Metrics dashboard**: ECharts (CDN) bar chart of audit event counts.
  - **Schedule panel**: list & create scheduled tasks via `/api/schedule`.
  - **File attachment**: attach a file from the chat bar (content shown to the
    agent, which can read it via its tools).

- **Web SSE streaming + auth + deployment.**
  - `/api/chat/stream`: SSE event-streaming chat (assistant deltas, tool
    start/end, done); frontend uses it with client-side multi-turn history.
  - **Auth**: set `JUSTAGENT_WEB_TOKEN` to require `Authorization: Bearer
    <token>` on `/api/*` (the page itself stays open).
  - **Deployment**: `Dockerfile` builds the web service; run with
    `docker run -p 8000:8000 -e JUSTAGENT_WEB_TOKEN=... -e OPENAI_API_KEY=...
    justagent-web`. `.env.example` documents the web env vars.

- **Complete Web backend (1:1 with CLI capabilities).** `justagent web` now
  exposes a full browser console, not just chat + judicial:
  - chat (multi-turn, per-session memory via `continue_run`)
  - judicial (cases list/detail/summary, evidence list/analyze, laws list/add,
    document generation)
  - knowledge RAG search, config (redacted), models, metrics, audit, sessions,
    plugins, doctor diagnostics, system info.
  Endpoints: `/api/state`, `/api/system`, `/api/doctor`, `/api/config`,
  `/api/models`, `/api/metrics`, `/api/audit`, `/api/sessions`,
  `/api/plugins`, `/api/knowledge/search`, `/api/chat`, `/api/judicial/*`,
  `/api/health`. Web UI gained a "系统" panel for diagnostics/config/models/
  sessions/plugins.

- **Web interface (`justagent web`).** A browser chat interface (FastAPI +
  uvicorn, already in dependencies). Users can chat with the conversational
  agent (which has the `judicial` tool) and manage cases / evidence / legal
  knowledge / documents through the web — no CLI commands needed. Judicial
  features work without an LLM; chatting needs a configured model backend.
  Endpoints: `/api/chat`, `/api/state`, `/api/judicial/case|law|doc`, `/api/health`.

- **`judicial` agent tool.** The conversational agent can now manage the
  judicial subsystem directly from the chat interface (no CLI commands):
  list cases, summarize a case with timeline, list/analyze evidence, browse
  or search the legal knowledge base, and generate legal documents. The agent
  is wired to the same persisted judicial state as the CLI
  (`.justagent/judicial_state.json`).
- **`judicial case summary`** — concise case overview + chronological timeline.
- **`judicial evidence export`** — export evidence list + chain analysis to
  markdown or JSON (for archiving / sharing).
- **`judicial law export`** — export the legal knowledge base to markdown or
  JSON (backup / training / sharing).

- **`judicial law list` / `judicial law show`.** Legal professionals can now
  browse the legal knowledge base (list all articles, filter by
  domain/law/status, JSON output) and view a single article's full details by
  ID or citation — previously the law library only supported add and search.

- **`justagent info` command.** Prints a compact environment & project
  diagnostics summary: version, Python/platform, project root & config path,
  model backends (with key status), plugin count, and git state. Every section
  is defensively wrapped so the command never crashes on a missing optional
  component. Complements `justagent doctor`.

- **`justagent models` command.** Lists every configured model backend
  (provider, model, tier, endpoint, key status) and, with `--check`, runs a
  live connectivity health check. Complements `justagent doctor`.

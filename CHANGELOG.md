# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

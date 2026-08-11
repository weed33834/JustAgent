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

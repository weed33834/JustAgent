# AutoShip CLI

Terminal-native delivery pipeline — clean, verify, commit, and ship from the command line.

[English](README.md) · [中文](README.zh.md) · [日本語](README.ja.md)

## What it does

AutoShip runs a delivery pipeline in four stages:

1. **Clean** — format code, sort imports, strip dead code
2. **Verify** — run your test suite and linters
3. **Commit** — generate conventional commit messages backed by your choice of LLM
4. **Upload** — push artifacts to PyPI, Docker Hub, or custom registries

Every stage is optional and toggleable. Run them together or one at a time.

## Install

```bash
pip install autoship
```

AI-powered commits and security scanning require the optional extras:

```bash
pip install "autoship[ai,security]"
```

Python 3.11 or later. Git 2.30+ recommended.

## Quick start

```bash
cd your-project
autoship init
autoship ship
```

Or pick individual stages:

```bash
autoship clean
autoship verify
autoship commit
autoship upload
```

## Configuration

AutoShip reads `autoship.toml` from your project root:

```toml
[clean]
enabled = true
tools = ["ruff"]

[commit]
enabled = true
conventional_commits = true

[[model.backends]]
provider = "openai"
base_url = "https://api.openai.com/v1"
api_key = "${OPENAI_API_KEY}"
model = "gpt-4o-mini"

[security]
enabled = true
tools = ["semgrep"]
threshold = "medium"
```

Environment variables in config values (`${VAR}`) are expanded at runtime. See `autoship.toml.example` for the full set of options.

## AI backends

AutoShip routes LLM calls through [LiteLLM](https://github.com/BerriAI/litellm), which means any provider LiteLLM handles is supported out of the box — OpenAI, Anthropic, Ollama, OpenRouter, Azure, vLLM, LM Studio, llama.cpp, and 100+ others.

Configure one or more backends in `autoship.toml`:

```toml
[[model.backends]]
provider = "ollama"
base_url = "http://localhost:11434"
model = "llama3.2"

[[model.backends]]
provider = "openrouter"
api_key = "${OPENROUTER_API_KEY}"
model = "anthropic/claude-sonnet-4"
```

The gateway handles retry, rate limiting, and automatic failover across providers.

## Commands

| Command | Description |
|---------|-------------|
| `autoship init` | Initialize in the current directory |
| `autoship clean` | Format and lint source files |
| `autoship verify` | Run test suite and checks |
| `autoship commit` | Generate and create a commit |
| `autoship upload` | Build and publish artifacts |
| `autoship ship` | Run clean, verify, commit, upload in sequence |
| `autoship fix` | AI-powered code fix suggestions |
| `autoship lsp` | Start the language server |
| `autoship plugin` | Manage plugins |

## Development

```bash
git clone https://github.com/MS33834/autoship-cli.git
cd autoship-cli
pip install -e ".[dev]"
```

Test, lint, type-check:

```bash
pytest tests/ -v
ruff check src/ tests/
mypy src/
```

## Tech stack

| Layer | Component |
|-------|-----------|
| CLI framework | Typer |
| Plugin system | Pluggy |
| AI gateway | LiteLLM |
| Config / schema | Pydantic v2 + pydantic-settings |
| Terminal output | Rich |
| Logging | Structlog (Rich console + JSON file) |
| LSP server | pygls |
| Cache | diskcache |
| Metrics | prometheus-client |
| HTTP client | httpx |
| Lint / format | Ruff |
| Security scan | Semgrep |

## Mirrors

- [GitCode](https://gitcode.com/badhope/autoship-cli) — faster clone for users in mainland China

## License

MIT — see [LICENSE](LICENSE).

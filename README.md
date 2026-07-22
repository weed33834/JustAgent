# MyAgent

Personal dev assistant — an AI agent CLI that lives in your terminal and helps you ship code.

[English](README.md) · [中文](README.zh.md) · [日本語](README.ja.md)

## What it does

MyAgent is a local-first AI coding agent that grows with you. It runs an interactive agent loop, manages your project, and orchestrates a delivery pipeline — all from the command line.

Three modes, one tool:

1. **Agent mode** — chat with an AI agent that can read/write files, run commands, search the web, and call MCP tools. Plan first, act second, with checkpoint-based rollback so every action is reversible.
2. **Pipeline mode** — the classic `clean → verify → commit → upload` delivery flow, optional and toggleable.
3. **Project mode** — manage multiple local projects, sync configs, run cross-project operations.

## Install

```bash
pip install myagent
```

AI-powered commits, security scanning, and the agent mode require the optional extras:

```bash
pip install "myagent[ai,security]"
```

Python 3.11 or later. Git 2.30+ recommended.

## Quick start

### Agent mode (new)

```bash
cd your-project
myagent agent                    # interactive agent loop
myagent agent "refactor utils"   # one-shot task
myagent agent --plan "..."       # plan first, then act
myagent agent --yolo "..."       # auto-approve all tool calls
myagent agent --json "..."       # NDJSON event stream for CI
```

### Pipeline mode (classic)

```bash
cd your-project
myagent init
myagent ship                     # clean → verify → commit → upload
```

Or pick individual stages:

```bash
myagent clean
myagent verify
myagent commit
myagent upload
```

### Project mode (new)

```bash
myagent project list             # list managed projects
myagent project add ./my-app
myagent project run my-app ship  # run a command in a managed project
```

## Configuration

MyAgent reads `.myagent.toml` from your project root:

```toml
[clean]
enabled = true
tools = ["ruff"]

[commit]
enabled = true
conventional_commits = true

[agent]
enabled = true
plan_mode_default = false
max_iterations = 50
compaction_trigger_ratio = 0.9

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

Environment variables in config values (`${VAR}`) are expanded at runtime. See `.myagent.toml.example` for the full set of options.

## AI backends

MyAgent routes LLM calls through [LiteLLM](https://github.com/BerriAI/litellm), which means any provider LiteLLM handles is supported out of the box — OpenAI, Anthropic, Ollama, OpenRouter, Azure, vLLM, LM Studio, llama.cpp, and 100+ others.

Configure one or more backends in `.myagent.toml`:

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

## Agent capabilities

The agent mode is built around an iterative tool-calling loop with safety rails:

| Capability | Description |
|---|---|
| **Plan/Act modes** | Plan mode is read-only analysis; Act mode executes changes. Switch explicitly or auto-switch on approval. |
| **Tool calling** | Built-in tools: `read_file`, `write_file`, `edit_file`, `apply_patch`, `run_command`, `search_code`, `web_fetch`, `ask_question`. Plus MCP tools. |
| **Checkpoints** | Shadow git snapshots after every tool call. Restore files, conversation, or both. |
| **Compaction** | Auto-compact long conversations at 90% context budget. Basic (truncate) or agentic (LLM summary) modes. |
| **Loop detection** | Detect repeated tool calls (soft=3, hard=5) and break out of doom loops. |
| **Mistake tracker** | Count consecutive errors, stop or continue based on config. |
| **Permissions** | `allow` / `deny` / `ask` rules with `once` / `always` scope and wildcard patterns. |
| **Skills** | Load `SKILL.md` files from `.myagent/skills/` with progressive disclosure. |
| **Instructions** | Auto-discover `AGENTS.md` / `CLAUDE.md` / `CONTEXT.md` at multiple directory levels. |
| **Subagents** | Spawn read-only parallel research subagents with isolated context. |
| **MCP** | Connect Model Context Protocol servers (stdio / SSE / HTTP) with OAuth support. |

## Commands

| Command | Description |
|---------|-------------|
| `myagent agent` | Interactive AI agent (new) |
| `myagent project` | Manage multiple local projects (new) |
| `myagent init` | Initialize in the current directory |
| `myagent clean` | Format and lint source files |
| `myagent verify` | Run test suite and checks |
| `myagent commit` | Generate and create a commit |
| `myagent upload` | Build and publish artifacts |
| `myagent ship` | Run clean, verify, commit, upload in sequence |
| `myagent fix` | AI-powered code fix suggestions |
| `myagent config` | View and manage configuration |
| `myagent doctor` | Diagnose environment and dependencies |
| `myagent plugin` | Manage plugins |
| `myagent hooks` | Manage lifecycle hooks |
| `myagent lsp` | Language Server Protocol integration |
| `myagent artifacts` | Manage build artifacts |
| `myagent metrics` | Show usage and cost metrics |

## Development

```bash
git clone https://gitcode.com/badhope/myagent.git
cd myagent
uv sync --all-extras
```

Test, lint, type-check:

```bash
uv run pytest tests/ -v
uv run ruff check src/ tests/
uv run mypy src/
```

## Tech stack

| Layer | Component |
|-------|-----------|
| CLI framework | Typer |
| Plugin system | Pluggy |
| AI gateway | LiteLLM |
| Agent loop | Custom (tool-calling iteration with safety rails) |
| Config / schema | Pydantic v2 + pydantic-settings |
| Terminal output | Rich |
| Logging | Structlog (Rich console + JSON file) |
| LSP server | pygls |
| Cache | diskcache |
| Metrics | prometheus-client |
| HTTP client | httpx |
| Lint / format | Ruff |
| Security scan | Semgrep |
| MCP | Official `mcp` Python SDK |

## Architecture

```
src/myagent/
├── cli/                 # Typer commands
│   ├── commands/        # One module per command
│   └── main.py          # Entry point + global options
├── agent/               # Agent core (new)
│   ├── runtime.py       # Iterative tool-calling loop
│   ├── tools/           # Built-in tool definitions + executors
│   ├── modes.py         # Plan/Act mode switching
│   ├── safety.py        # Loop detection + mistake tracker
│   └── compaction.py    # Context compaction
├── checkpoint/          # Shadow git snapshots (new)
├── context/             # Context engineering (new)
│   ├── skill.py         # SKILL.md loader
│   ├── instruction.py   # AGENTS.md auto-discovery
│   └── repo_map.py      # tree-sitter repo map
├── permissions/         # Tool permission rules (new)
├── hooks/               # Lifecycle hooks (enhanced)
├── adapters/            # External integrations
│   ├── providers/       # LLM providers via unified gateway
│   ├── upload/          # PyPI / Docker / GitHub uploaders
│   ├── git_adapter.py
│   ├── model_gateway.py
│   └── tool_adapter.py
├── core/                # Core infrastructure
│   ├── config_center.py
│   ├── audit_logger.py
│   ├── hook_dispatcher.py
│   ├── plugin_registry.py
│   ├── sandbox.py
│   ├── cache.py
│   └── ...
├── plugins/             # Built-in plugins
├── models/              # Pydantic config schemas
├── utils/               # Shared utilities
└── registry/            # Plugin registry index
```

## Mirrors

- [GitCode](https://gitcode.com/badhope/myagent) — faster clone for users in mainland China

## License

MIT — see [LICENSE](LICENSE).

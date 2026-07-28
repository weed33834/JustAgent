# JustAgent

[English](README.md) | [中文](README.zh.md) | [日本語](README.ja.md)

A local-first AI coding agent that runs in the terminal. It reads, writes, and edits code, runs commands, and searches the web, with permission controls on every action and persistence for every conversation.

## What it does

JustAgent is a local-first AI coding agent packaged as a single CLI tool, comparable to Cline, Aider, OpenCode, and Continue.dev. It runs an iterative tool-calling loop: the LLM reads the codebase, proposes changes, executes them through tools, and verifies the results. Every destructive action passes through a permission engine, and every conversation can be saved and resumed.

Three modes are available:

1. **Agent mode** — interactive REPL or one-shot task. Plan first (read-only), then act (with permission prompts), or enable Yolo mode to auto-approve tool calls. Multi-turn conversations persist across sessions.
2. **Pipeline mode** — an optional `clean → verify → commit → upload` shortcut for release workflows.
3. **Project mode** — manage multiple local projects and run cross-project operations.

## Install

```bash
pip install justagent
```

AI-powered commits, security scanning, and agent mode require the optional extras:

```bash
pip install "justagent[ai,security]"
```

Python 3.11 or later. Git 2.30+ recommended.

## Quick start

### Agent mode

```bash
cd your-project
justagent agent                    # interactive REPL (multi-turn chat)
justagent agent "refactor utils"   # one-shot task
justagent agent --plan "..."       # plan first (read-only, no edits)
justagent agent --yolo "..."       # auto-approve all tool calls
justagent agent -i --resume <id>   # resume a saved session
justagent agent --json "..."       # NDJSON event stream for automation
```

### Pipeline mode (optional)

```bash
cd your-project
justagent init
justagent ship                     # clean → verify → commit → upload
```

Or run individual stages:

```bash
justagent clean
justagent verify
justagent commit
justagent upload
```

### Project mode

```bash
justagent project list             # list managed projects
justagent project add ./my-app
justagent project run my-app ship  # run a command in a managed project
```

## Configuration

JustAgent reads `.justagent.toml` from the project root:

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

Environment variables in config values (`${VAR}`) are expanded at runtime. See `.justagent.toml.example` for the full set of options.

## AI backends

JustAgent routes LLM calls through [LiteLLM](https://github.com/BerriAI/litellm), so any provider LiteLLM supports is available without additional configuration — OpenAI, Anthropic, Ollama, OpenRouter, Azure, vLLM, LM Studio, llama.cpp, and more than a hundred others.

Configure one or more backends in `.justagent.toml`:

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

Agent mode is built on an iterative tool-calling loop with safety controls:

| Capability | Description |
|---|---|
| **Plan/Act modes** | Plan mode is read-only analysis; Act mode executes changes with permission prompts. Switch with `--plan` / `--yolo` or `/mode` in the REPL. |
| **Tool calling** | Built-in tools: `read_file`, `write_to_file`, `replace_in_file`, `apply_patch`, `run_command`, `search`, `web_fetch`, `ask_question`. Plus MCP tools. |
| **Session persistence** | Conversations saved to `~/.justagent/sessions/`. Resume with `--resume <id>`. Slash commands: `/tokens`, `/history`, `/diff`. |
| **Permissions** | `allow` / `deny` / `ask` rules with `once` / `always` scope and wildcard patterns. Wired into `write_to_file`, `replace_in_file`, `apply_patch`, `run_command`. |
| **Change tracking** | Tracks every file created/modified/deleted during a run, with line-count deltas. Shows a summary table at the end. |
| **Checkpoints** | Shadow git snapshots after every tool call. Restore files, conversation, or both. |
| **Compaction** | Auto-compact long conversations at 90% context budget. Basic (truncate) or agentic (LLM summary) modes. |
| **Loop detection** | Detect repeated tool calls (soft=3, hard=5) and break out of repetitive loops. |
| **Mistake tracker** | Count consecutive errors, stop or continue based on config. |
| **Repo map** | Regex-based symbol extraction (Python/JS/TS/Rust/Go) formatted as a compact tree. |
| **Skills** | Load `SKILL.md` files from `.justagent/skills/` with progressive disclosure. |
| **Subagents** | Spawn read-only parallel research subagents with isolated context. |
| **MCP** | Connect Model Context Protocol servers (stdio / SSE / HTTP) with OAuth support. |

### Slash commands (in interactive REPL)

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/mode <act\|plan\|yolo>` | Switch agent mode |
| `/tokens` | Show token usage breakdown |
| `/history` | Show conversation history |
| `/diff` | Show git diff |
| `/lint` | Run linter |
| `/test` | Run tests |
| `/add <path>` | Add file to context |
| `/drop <path>` | Drop file from context |
| `/clear` | Clear conversation |
| `/exit` | Exit REPL |

## Commands

| Command | Description |
|---------|-------------|
| `justagent agent` | Interactive AI agent (REPL or one-shot) |
| `justagent session` | Manage saved sessions (list / show / resume / delete) |
| `justagent project` | Manage multiple local projects |
| `justagent init` | Initialize in the current directory |
| `justagent clean` | Format and lint source files |
| `justagent verify` | Run test suite and checks |
| `justagent commit` | Generate and create a commit |
| `justagent upload` | Build and publish artifacts |
| `justagent ship` | Run clean, verify, commit, upload in sequence |
| `justagent fix` | AI-powered code fix suggestions |
| `justagent config` | View and manage configuration |
| `justagent doctor` | Diagnose environment and dependencies |
| `justagent plugin` | Manage plugins |
| `justagent hooks` | Manage lifecycle hooks |
| `justagent lsp` | Language Server Protocol integration |
| `justagent artifacts` | Manage build artifacts |
| `justagent metrics` | Show usage and cost metrics |

## Development

```bash
git clone https://gitcode.com/badhope/justagent.git
cd justagent
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
| Agent loop | Custom (tool-calling iteration with safety controls) |
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
src/justagent/
├── cli/                 # Typer commands
│   ├── commands/        # One module per command (agent, session, project, ...)
│   ├── display.py       # Rich-based terminal output (spinners, panels, diff)
│   └── main.py          # Entry point + global options
├── agent/               # Agent core
│   ├── runtime.py       # Iterative tool-calling loop (run / continue_run / reset)
│   ├── tools/           # Built-in tool definitions + executors
│   │   ├── base.py      # Tool, ToolContext, ToolResult, request_permission()
│   │   ├── registry.py  # Tool registration
│   │   └── builtin/     # read_file, write_to_file, replace_in_file, apply_patch,
│   │                    # run_command, search, web_fetch, ask_question
│   ├── plan_act.py      # Plan/Act/Yolo mode system prompt + tool filtering
│   ├── session.py       # Session persistence (save / resume / serialize)
│   ├── slash_commands.py # /help /mode /tokens /history /diff /lint /test ...
│   ├── change_tracker.py # File change tracking with line-count deltas
│   ├── loop_detection.py # Repeated-tool-call loop detection
│   ├── mistake_tracker.py # Consecutive error tracking
│   ├── compaction.py    # Context compaction (basic + agentic)
│   ├── subagent.py      # Read-only parallel research subagents
│   └── mcp_client.py    # Model Context Protocol client
├── checkpoint/          # Shadow git snapshots
├── context/             # Context engineering
│   ├── skill.py         # SKILL.md loader
│   └── repo_map.py      # Regex-based repo symbol map
├── permissions/         # Tool permission engine (allow / deny / ask rules)
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
│   ├── batch_ops.py     # Cross-project batch operations
│   ├── sandbox.py
│   ├── cache.py
│   └── ...
├── plugins/             # Built-in plugins (security_scan, typecheck, docker_ship, ...)
├── models/              # Pydantic config schemas
├── utils/               # Shared utilities
└── registry/            # Plugin registry index
```

## Mirrors

- [GitCode](https://gitcode.com/badhope/justagent) — faster clone for users in mainland China

## License

MIT — see [LICENSE](LICENSE).

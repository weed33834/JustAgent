# MyAgent

A local-first AI coding agent that lives in your terminal. Chat with an AI that can read, write, and edit your code, run commands, and search the web — all with permission controls and session persistence.

[English](README.md) · [中文](README.zh.md) · [日本語](README.ja.md)

## What it does

MyAgent is a local-first AI coding agent — think Cline / Aider / OpenCode / Continue.dev, but as a single CLI tool. It runs an iterative tool-calling loop: the LLM reads your codebase, proposes changes, executes them via tools, and verifies the results. Every destructive action goes through a permission engine; every conversation can be saved and resumed.

Three modes, one tool:

1. **Agent mode** — interactive REPL or one-shot. Plan first (read-only), then act (with permission prompts). Or go full Yolo (auto-approve). Multi-turn conversations persist across sessions.
2. **Pipeline mode** — optional `clean → verify → commit → upload` shortcut for when you just want to ship a release.
3. **Project mode** — manage multiple local projects, run cross-project operations.

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

### Agent mode

```bash
cd your-project
myagent agent                    # interactive REPL (multi-turn chat)
myagent agent "refactor utils"   # one-shot task
myagent agent --plan "..."       # plan first (read-only, no edits)
myagent agent --yolo "..."       # auto-approve all tool calls
myagent agent -i --resume <id>   # resume a saved session
myagent agent --json "..."       # NDJSON event stream for automation
```

### Pipeline mode (optional)

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

### Project mode

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
| **Plan/Act modes** | Plan mode is read-only analysis; Act mode executes changes with permission prompts. Switch with `--plan` / `--yolo` or `/mode` in the REPL. |
| **Tool calling** | Built-in tools: `read_file`, `write_to_file`, `replace_in_file`, `apply_patch`, `run_command`, `search`, `web_fetch`, `ask_question`. Plus MCP tools. |
| **Session persistence** | Conversations saved to `~/.myagent/sessions/`. Resume with `--resume <id>`. Slash commands: `/tokens`, `/history`, `/diff`. |
| **Permissions** | `allow` / `deny` / `ask` rules with `once` / `always` scope and wildcard patterns. Wired into `write_to_file`, `replace_in_file`, `apply_patch`, `run_command`. |
| **Change tracking** | Tracks every file created/modified/deleted during a run, with line-count deltas. Shows a summary table at the end. |
| **Checkpoints** | Shadow git snapshots after every tool call. Restore files, conversation, or both. |
| **Compaction** | Auto-compact long conversations at 90% context budget. Basic (truncate) or agentic (LLM summary) modes. |
| **Loop detection** | Detect repeated tool calls (soft=3, hard=5) and break out of doom loops. |
| **Mistake tracker** | Count consecutive errors, stop or continue based on config. |
| **Repo map** | Regex-based symbol extraction (Python/JS/TS/Rust/Go) formatted as a compact tree. |
| **Skills** | Load `SKILL.md` files from `.myagent/skills/` with progressive disclosure. |
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
| `myagent agent` | Interactive AI agent (REPL or one-shot) |
| `myagent session` | Manage saved sessions (list / show / resume / delete) |
| `myagent project` | Manage multiple local projects |
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

- [GitCode](https://gitcode.com/badhope/myagent) — faster clone for users in mainland China

## License

MIT — see [LICENSE](LICENSE).

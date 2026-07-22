# MyAgent

本地优先的 AI 编码智能体，运行在终端中。与 AI 对话，让它读取、写入、编辑你的代码，运行命令，搜索网页——所有操作都有权限控制，所有对话都可保存和恢复。

[English](README.md) · [中文](README.zh.md) · [日本語](README.ja.md)

## 它做什么

MyAgent 是一个本地优先的 AI 编码智能体——可以理解为 Cline / Aider / OpenCode / Continue.dev，但封装成一个 CLI 工具。它运行一个迭代的工具调用循环：LLM 读取你的代码库，提出修改，通过工具执行，然后验证结果。每个破坏性操作都经过权限引擎；每个对话都可以保存和恢复。

三种模式，一个工具：

1. **Agent 模式** — 交互式 REPL 或一次性任务。先规划（只读），再执行（带权限提示）。或全 Yolo 模式（自动批准）。多轮对话跨会话持久化。
2. **Pipeline 模式** — 可选的 `clean → verify → commit → upload` 快捷方式，用于发布时。
3. **Project 模式** — 管理多个本地项目，运行跨项目操作。

## 安装

```bash
pip install myagent
```

AI 提交、安全扫描和 agent 模式需要可选依赖：

```bash
pip install "myagent[ai,security]"
```

Python 3.11 或更高版本。推荐 Git 2.30+。

## 快速开始

### Agent 模式

```bash
cd your-project
myagent agent                    # 交互式 REPL（多轮对话）
myagent agent "refactor utils"   # 一次性任务
myagent agent --plan "..."       # 先规划（只读，不修改文件）
myagent agent --yolo "..."       # 自动批准所有工具调用
myagent agent -i --resume <id>   # 恢复已保存的会话
myagent agent --json "..."       # NDJSON 事件流（自动化用）
```

### Pipeline 模式（可选）

```bash
cd your-project
myagent init
myagent ship                     # clean → verify → commit → upload
```

或单独执行各阶段：

```bash
myagent clean
myagent verify
myagent commit
myagent upload
```

### Project 模式

```bash
myagent project list             # 列出管理的项目
myagent project add ./my-app
myagent project run my-app ship  # 在项目中运行命令
```

## 配置

MyAgent 从项目根目录读取 `.myagent.toml`：

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

配置值中的环境变量（`${VAR}`）会在运行时展开。完整选项见 `.myagent.toml.example`。

## AI 后端

MyAgent 通过 [LiteLLM](https://github.com/BerriAI/litellm) 路由 LLM 调用，LiteLLM 支持的任何提供商都开箱即用——OpenAI、Anthropic、Ollama、OpenRouter、Azure、vLLM、LM Studio、llama.cpp 等 100+ 家。

在 `.myagent.toml` 中配置一个或多个后端：

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

网关处理重试、限流和跨提供商自动故障转移。

## Agent 能力

Agent 模式基于迭代的工具调用循环，带安全护栏：

| 能力 | 说明 |
|------|------|
| **Plan/Act 模式** | Plan 模式只读分析；Act 模式带权限提示执行修改。用 `--plan` / `--yolo` 或 REPL 中 `/mode` 切换。 |
| **工具调用** | 内置工具：`read_file`、`write_to_file`、`replace_in_file`、`apply_patch`、`run_command`、`search`、`web_fetch`、`ask_question`。外加 MCP 工具。 |
| **会话持久化** | 对话保存到 `~/.myagent/sessions/`。用 `--resume <id>` 恢复。Slash 命令：`/tokens`、`/history`、`/diff`。 |
| **权限引擎** | `allow` / `deny` / `ask` 规则，支持 `once` / `always` 作用域和通配符。已接入 `write_to_file`、`replace_in_file`、`apply_patch`、`run_command`。 |
| **变更追踪** | 追踪每次运行中创建/修改/删除的文件，计算行数差异，结束时显示汇总表。 |
| **检查点** | 每次工具调用后创建影子 git 快照。可恢复文件、对话或两者。 |
| **上下文压缩** | 90% 上下文预算时自动压缩长对话。基础（截断）或智能（LLM 摘要）模式。 |
| **循环检测** | 检测重复工具调用（软=3，硬=5），跳出死循环。 |
| **错误追踪** | 统计连续错误，根据配置停止或继续。 |
| **Repo Map** | 正则提取 Python/JS/TS/Rust/Go 符号，格式化为紧凑树。 |
| **Skills** | 从 `.myagent/skills/` 加载 `SKILL.md` 文件，渐进式展示。 |
| **子智能体** | 生成只读并行研究子智能体，隔离上下文。 |
| **MCP** | 连接 Model Context Protocol 服务器（stdio / SSE / HTTP），支持 OAuth。 |

### Slash 命令（交互式 REPL 中）

| 命令 | 说明 |
|------|------|
| `/help` | 显示可用命令 |
| `/mode <act\|plan\|yolo>` | 切换 agent 模式 |
| `/tokens` | 显示 token 用量明细 |
| `/history` | 显示对话历史 |
| `/diff` | 显示 git diff |
| `/lint` | 运行 linter |
| `/test` | 运行测试 |
| `/add <path>` | 添加文件到上下文 |
| `/drop <path>` | 从上下文移除文件 |
| `/clear` | 清空对话 |
| `/exit` | 退出 REPL |

## 命令

| 命令 | 说明 |
|------|------|
| `myagent agent` | 交互式 AI 智能体（REPL 或一次性） |
| `myagent session` | 管理已保存的会话（list / show / resume / delete） |
| `myagent project` | 管理多个本地项目 |
| `myagent init` | 在当前目录初始化 |
| `myagent clean` | 格式化和 lint 源文件 |
| `myagent verify` | 运行测试套件和检查 |
| `myagent commit` | 生成并创建提交 |
| `myagent upload` | 构建并发布产物 |
| `myagent ship` | 依次执行 clean、verify、commit、upload |
| `myagent fix` | AI 驱动的代码修复建议 |
| `myagent config` | 查看和管理配置 |
| `myagent doctor` | 诊断环境和依赖 |
| `myagent plugin` | 管理插件 |
| `myagent hooks` | 管理生命周期钩子 |
| `myagent lsp` | 语言服务器协议集成 |
| `myagent artifacts` | 管理构建产物 |
| `myagent metrics` | 显示用量和成本指标 |

## 开发

```bash
git clone https://gitcode.com/badhope/myagent.git
cd myagent
uv sync --all-extras
```

测试、lint、类型检查：

```bash
uv run pytest tests/ -v
uv run ruff check src/ tests/
uv run mypy src/
```

## 技术栈

| 层 | 组件 |
|----|------|
| CLI 框架 | Typer |
| 插件系统 | Pluggy |
| AI 网关 | LiteLLM |
| Agent 循环 | 自研（工具调用迭代 + 安全护栏） |
| 配置 / schema | Pydantic v2 + pydantic-settings |
| 终端输出 | Rich |
| 日志 | Structlog（Rich 控制台 + JSON 文件） |
| LSP 服务器 | pygls |
| 缓存 | diskcache |
| 指标 | prometheus-client |
| HTTP 客户端 | httpx |
| Lint / 格式化 | Ruff |
| 安全扫描 | Semgrep |
| MCP | 官方 `mcp` Python SDK |

## 架构

```
src/myagent/
├── cli/                 # Typer 命令
│   ├── commands/        # 每个命令一个模块（agent, session, project, ...）
│   ├── display.py       # Rich 终端输出（spinner、面板、diff）
│   └── main.py          # 入口 + 全局选项
├── agent/               # Agent 核心
│   ├── runtime.py       # 迭代工具调用循环（run / continue_run / reset）
│   ├── tools/           # 内置工具定义 + 执行器
│   │   ├── base.py      # Tool, ToolContext, ToolResult, request_permission()
│   │   ├── registry.py  # 工具注册
│   │   └── builtin/     # read_file, write_to_file, replace_in_file, apply_patch,
│   │                    # run_command, search, web_fetch, ask_question
│   ├── plan_act.py      # Plan/Act/Yolo 模式 system prompt + 工具过滤
│   ├── session.py       # 会话持久化（保存 / 恢复 / 序列化）
│   ├── slash_commands.py # /help /mode /tokens /history /diff /lint /test ...
│   ├── change_tracker.py # 文件变更追踪（行数差异）
│   ├── loop_detection.py # 重复工具调用循环检测
│   ├── mistake_tracker.py # 连续错误追踪
│   ├── compaction.py    # 上下文压缩（基础 + 智能）
│   ├── subagent.py      # 只读并行研究子智能体
│   └── mcp_client.py    # Model Context Protocol 客户端
├── checkpoint/          # 影子 git 快照
├── context/             # 上下文工程
│   ├── skill.py         # SKILL.md 加载器
│   └── repo_map.py      # 正则 repo 符号图
├── permissions/         # 工具权限引擎（allow / deny / ask 规则）
├── adapters/            # 外部集成
│   ├── providers/       # LLM 提供商（统一网关）
│   ├── upload/          # PyPI / Docker / GitHub 上传器
│   ├── git_adapter.py
│   ├── model_gateway.py
│   └── tool_adapter.py
├── core/                # 核心基础设施
│   ├── config_center.py
│   ├── audit_logger.py
│   ├── hook_dispatcher.py
│   ├── plugin_registry.py
│   ├── batch_ops.py     # 跨项目批量操作
│   ├── sandbox.py
│   ├── cache.py
│   └── ...
├── plugins/             # 内置插件（security_scan, typecheck, docker_ship, ...）
├── models/              # Pydantic 配置 schema
├── utils/               # 共享工具
└── registry/            # 插件注册表索引
```

## 镜像

- [GitCode](https://gitcode.com/badhope/myagent) — 中国大陆用户快速克隆

## 许可证

MIT — 见 [LICENSE](LICENSE)。

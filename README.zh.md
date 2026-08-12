# JustAgent - 司法智能体平台

[English](README.md) | [中文](README.zh.md) | [日本語](README.ja.md)

> **Intelligence for Justice —— 智能赋能司法**

JustAgent 是一个司法智能体平台，辅助甚至自动化完成司法机关的职责，智能化司法流程。它通过多智能体协作循环完成案件材料梳理、证据审查、法律文书生成——每个操作都带权限控制，每个对话都可全程留痕与恢复。

## 它做什么

JustAgent 是一个面向司法机关与政府部门的司法智能体平台，封装为单个 CLI 工具。它运行迭代的工具调用循环：LLM 读取案件材料，提出处理动作，通过工具执行，然后验证结果。每个破坏性操作都经过权限引擎，每个对话都可以保存与恢复——让司法过程可审计、可控制。

提供三种模式：

1. **Agent 模式** — 交互式 REPL 或一次性任务。先规划（只读分析案件卷宗），再执行（带权限提示），或启用 Yolo 模式自动批准工具调用。多轮对话跨会话持久化。
2. **Pipeline 模式** — 可选的 `clean → verify → commit → upload` 快捷方式，用于司法文书的发布流程。
3. **Project 模式** — 管理多个案件项目，运行跨项目操作。

## 三大核心功能

| 功能 | 说明 |
|------|------|
| **案件材料梳理** | 摄入、分类、汇总案件材料（卷宗、笔录、物证清单）。抽取实体、时间线与关系，形成结构化案情图谱，便于快速研判。 |
| **证据审查** | 交叉核验证据的一致性、完整性与可采性。标记矛盾、缺失与证据链问题，并回溯到原始材料出处。 |
| **法律文书生成** | 依据案情上下文起草起诉书、判决书、裁定书等法律文书。套用标准化模板，支持辖区感知的格式与条款库。 |
| **司法流程自动化** | 编排多步司法程序（立案 → 审查 → 庭审 → 裁判）。通过编排层协调专业化智能体，全程审计留痕。 |

## 架构设计（多智能体协作）

JustAgent 通过编排层协调专业化智能体，下层共享知识库与统一的权限引擎：

```
┌───────────────────────────────────────────────────────────┐
│                  编排器 / 协调器                          │
│        （流程编排 · 决策路由 · 智能体网格）                │
└───────────┬───────────────┬───────────────┬───────────────┘
            │               │               │
            ▼               ▼               ▼
   ┌────────────────┐ ┌──────────────┐ ┌────────────────┐
   │  案件材料梳理   │ │   证据审查    │ │   法律文书生成  │
   │    智能体       │ │    智能体     │ │    智能体       │
   └────────┬────────┘ └──────┬───────┘ └───────┬────────┘
            │                 │                 │
            ▼                 ▼                 ▼
   ┌──────────────────────────────────────────────────────┐
   │  知识层（RAG / 向量 / 知识图谱 / 文档处理）           │
   │  安全层（RBAC / SSO / 加密 / 审计）                  │
   │  权限引擎 · 检查点 · 审计日志                        │
   └──────────────────────────────────────────────────────┘
```

各专业化智能体共享同一知识层（法律语料库、案件库、证据库），并在统一的权限引擎下运行，全程审计留痕。每个智能体动作都建立检查点、可回溯，使司法过程透明、可辩护。

## 安装

```bash
pip install justagent
```

AI 文书生成、证据分析与 agent 模式需要可选依赖：

```bash
pip install "justagent[ai,security]"
```

Python 3.11 或更高版本。推荐 Git 2.30+。

## 快速开始

### Agent 模式

```bash
cd your-case-project
justagent agent                    # 交互式 REPL（多轮对话）
justagent agent "审查案件 A001 证据"   # 一次性任务
justagent agent --plan "..."       # 先规划（只读，不修改文件）
justagent agent --yolo "..."       # 自动批准所有工具调用
justagent agent -i --resume <id>   # 恢复已保存的会话
justagent agent --json "..."       # NDJSON 事件流（自动化用）
```

### Pipeline 模式（可选）

```bash
cd your-case-project
justagent init
justagent ship                     # clean → verify → commit → upload
```

或单独执行各阶段：

```bash
justagent clean
justagent verify
justagent commit
justagent upload
```

### Project 模式

```bash
justagent project list             # 列出管理的案件项目
justagent project add ./case-2026-001
justagent project run case-2026-001 ship  # 在案件中运行命令
```

## 配置

JustAgent 从项目根目录读取 `.justagent.toml`：

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

配置值中的环境变量（`${VAR}`）会在运行时展开。完整选项（含司法场景示例）见 `.justagent.toml.example`。

## AI 后端

JustAgent 通过官方 [OpenAI SDK](https://github.com/openai/openai-python) 路由 LLM 调用。所有受支持的提供商——OpenAI、Ollama、OpenRouter、Azure、vLLM、LM Studio、llama.cpp——都暴露 OpenAI 兼容端点，因此一个客户端配合按提供商设置的 base URL 即可全部覆盖，无需额外配置。对于敏感的司法数据，可使用自托管后端（Ollama、vLLM、llama.cpp）保证数据不出域。

在 `.justagent.toml` 中配置一个或多个后端：

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

## 模块说明

Agent 模式基于迭代的工具调用循环，带安全控制：

| 模块 | 说明 |
|------|------|
| **Plan/Act 模式** | Plan 模式只读分析；Act 模式带权限提示执行修改。用 `--plan` / `--yolo` 或 REPL 中 `/mode` 切换。 |
| **工具调用** | 内置工具：`read_file`、`write_to_file`、`replace_in_file`、`apply_patch`、`run_command`、`search`、`web_fetch`、`ask_question`。外加 MCP 工具。 |
| **会话持久化** | 对话保存到 `~/.justagent/sessions/`。用 `--resume <id>` 恢复。Slash 命令：`/tokens`、`/history`、`/diff`。 |
| **权限引擎** | `allow` / `deny` / `ask` 规则，支持 `once` / `always` 作用域和通配符。已接入 `write_to_file`、`replace_in_file`、`apply_patch`、`run_command`。 |
| **变更追踪** | 追踪每次运行中创建/修改/删除的文件，计算行数差异，结束时显示汇总表。 |
| **检查点** | 每次工具调用后创建影子 git 快照。可恢复文件、对话或两者。 |
| **上下文压缩** | 90% 上下文预算时自动压缩长对话。基础（截断）或智能（LLM 摘要）模式。 |
| **循环检测** | 检测重复工具调用（软=3，硬=5），跳出重复循环。 |
| **错误追踪** | 统计连续错误，根据配置停止或继续。 |
| **Repo Map** | 正则提取 Python/JS/TS/Rust/Go 符号，格式化为紧凑树。 |
| **Skills** | 从 `.justagent/skills/` 加载 `SKILL.md` 文件，渐进式展示。 |
| **子智能体** | 生成只读并行研究子智能体，隔离上下文。 |
| **MCP** | 连接 Model Context Protocol 服务器（stdio / SSE / HTTP），支持 OAuth。 |
| **知识层** | 文档处理、ETL、知识图谱、RAG 与向量存储，用于案件语料与法律参考。 |
| **编排层** | 协调器、决策路由、智能体网格与流程引擎，支撑多智能体协作。 |
| **安全与合规** | RBAC、SSO、加密、数据保护与合规检查，面向司法与政府场景。 |

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
| `justagent agent` | 交互式 AI 智能体（REPL 或一次性） |
| `justagent session` | 管理已保存的会话（list / show / resume / delete） |
| `justagent project` | 管理多个案件项目 |
| `justagent init` | 在当前目录初始化 |
| `justagent clean` | 格式化和 lint 源文件 |
| `justagent verify` | 运行测试套件和检查 |
| `justagent commit` | 生成并创建提交 |
| `justagent upload` | 构建并发布产物 |
| `justagent ship` | 依次执行 clean、verify、commit、upload |
| `justagent fix` | AI 驱动的修复建议 |
| `justagent config` | 查看和管理配置 |
| `justagent doctor` | 诊断环境和依赖 |
| `justagent plugin` | 管理插件 |
| `justagent hooks` | 管理生命周期钩子 |
| `justagent lsp` | 语言服务器协议集成 |
| `justagent artifacts` | 管理构建产物 |
| `justagent metrics` | 显示用量和成本指标 |

## 开发指南

```bash
git clone https://gitcode.com/badhope/justagent.git
cd justagent
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
| Agent 循环 | 自研（工具调用迭代 + 安全控制） |
| 多智能体编排 | Coordinator / mesh / workflow / decision |
| 知识层 | RAG · 向量 · 知识图谱 · 文档 ETL |
| 配置 / schema | Pydantic v2 + pydantic-settings |
| 终端输出 | Rich |
| 日志 | Structlog（Rich 控制台 + JSON 文件） |
| LSP 服务器 | pygls |
| 缓存 | diskcache |
| 指标 | prometheus-client |
| HTTP 客户端 | httpx |
| Lint / 格式化 | Ruff |
| 安全扫描 | Semgrep |
| 安全与合规 | RBAC · SSO · 加密 · 数据保护 |
| MCP | 官方 `mcp` Python SDK |

## 架构

```
src/justagent/
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
├── orchestration/       # 多智能体协作
│   ├── coordinator.py   # 智能体协调
│   ├── decision.py      # 决策路由
│   ├── mesh.py          # 智能体网格
│   └── workflow.py      # 司法流程编排
├── knowledge/           # 司法知识层
│   ├── rag.py           # 检索增强生成
│   ├── vector.py        # 向量存储
│   ├── graph.py         # 知识图谱
│   ├── document.py      # 文档处理
│   └── etl.py           # 抽取 / 转换 / 加载
├── communication/       # 智能体间与团队沟通
│   ├── audit.py         # 审计留痕
│   ├── broadcast.py     # 广播
│   ├── meeting.py       # 会议 / 庭审协调
│   ├── messaging.py     # 消息
│   └── notification.py  # 通知
├── security/            # 司法级安全
│   ├── rbac.py          # 基于角色的访问控制
│   ├── sso.py           # 单点登录
│   ├── encryption.py    # 加密
│   ├── data_protection.py # 数据保护
│   └── compliance.py    # 合规检查
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

- [GitCode](https://gitcode.com/badhope/justagent) — 中国大陆用户快速克隆

## 许可证

Apache-2.0 — 见 [LICENSE](LICENSE)。

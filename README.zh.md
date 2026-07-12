# AutoShip CLI

终端原生的交付流水线——格式化、校验、提交、发布，一条命令走完。

[English](README.md) · [中文](README.zh.md) · [日本語](README.ja.md)

## 这个工具做什么

AutoShip 把交付流程拆成四个可独立开关的阶段：

1. **Clean** — 代码格式化、导入排序、死代码清理
2. **Verify** — 跑测试套件和 lint 检查
3. **Commit** — 用你指定的 LLM 生成 conventional commit 消息
4. **Upload** — 制品推送到 PyPI、Docker Hub 或自定义仓库

四个阶段一起跑也行，单独跑也行。

## 安装

```bash
pip install autoship
```

需要 AI 提交消息和代码安全扫描的话，装上可选依赖：

```bash
pip install "autoship[ai,security]"
```

Python 3.11 以上，Git 2.30 以上推荐。

## 快速上手

```bash
cd 你的项目目录
autoship init
autoship ship
```

也可以单步执行：

```bash
autoship clean
autoship verify
autoship commit
autoship upload
```

## 配置

在项目根目录放一个 `autoship.toml`：

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

配置值中可以用 `${VAR}` 引用环境变量，运行时会自动展开。完整配置项参考 `autoship.toml.example`。

## AI 后端

AutoShip 通过 [LiteLLM](https://github.com/BerriAI/litellm) 统一调用大模型，LiteLLM 支持的供应商全部开箱可用——OpenAI、Anthropic、Ollama、OpenRouter、Azure、vLLM、LM Studio、llama.cpp 等 100 多家。

在 `autoship.toml` 里配置一个或多个后端：

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

网关内置重试、限流和自动故障转移。

## 命令列表

| 命令 | 说明 |
|------|------|
| `autoship init` | 在当前目录初始化 |
| `autoship clean` | 格式化并 lint 源码 |
| `autoship verify` | 执行测试套件 |
| `autoship commit` | 生成并创建提交 |
| `autoship upload` | 构建并发布制品 |
| `autoship ship` | 按序执行全流程 |
| `autoship fix` | AI 辅助的代码修复建议 |
| `autoship lsp` | 启动语言服务器 |
| `autoship plugin` | 插件管理 |

## 开发

```bash
git clone https://github.com/MS33834/autoship-cli.git
cd autoship-cli
pip install -e ".[dev]"
```

测试、lint、类型检查：

```bash
pytest tests/ -v
ruff check src/ tests/
mypy src/
```

## 技术栈

| 层次 | 组件 |
|------|------|
| CLI 框架 | Typer |
| 插件系统 | Pluggy |
| AI 网关 | LiteLLM |
| 配置 / Schema | Pydantic v2 + pydantic-settings |
| 终端输出 | Rich |
| 日志 | Structlog（Rich 控制台 + JSON 文件） |
| LSP 服务端 | pygls |
| 缓存 | diskcache |
| 指标 | prometheus-client |
| HTTP 客户端 | httpx |
| Lint / 格式化 | Ruff |
| 安全扫描 | Semgrep |

## 镜像仓库

- [GitCode](https://gitcode.com/badhope/autoship-cli) — 国内用户克隆更快

## 许可证

MIT — 详见 [LICENSE](LICENSE)。

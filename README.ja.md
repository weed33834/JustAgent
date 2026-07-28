# JustAgent

[English](README.md) | [中文](README.zh.md) | [日本語](README.ja.md)

ローカルファーストの AI コーディングエージェント。ターミナルで動作し、コードの読み取り・書き込み・編集、コマンド実行、ウェブ検索を行います。すべての操作に権限管理があり、すべての会話を保存・再開できます。

## 概要

JustAgent はローカルファーストの AI コーディングエージェントで、単一の CLI ツールとして提供され、Cline / Aider / OpenCode / Continue.dev と同等の位置づけにあります。反復的なツール呼び出しループを実行し、LLM がコードベースを読み、変更を提案し、ツールで実行し、結果を検証します。すべての破壊的操作は権限エンジンを経由し、すべての会話は保存・再開できます。

3つのモードを提供します：

1. **Agent モード** — インタラクティブ REPL またはワンショットタスク。まず計画（読み取り専用）、次に実行（権限プロンプト付き）、または Yolo モードでツール呼び出しを自動承認。複数ターンの会話はセッション間で永続化されます。
2. **Pipeline モード** — オプションの `clean → verify → commit → upload` ショートカット。リリースフロー用。
3. **Project モード** — 複数のローカルプロジェクトを管理、クロスプロジェクト操作を実行。

## インストール

```bash
pip install justagent
```

AI コミット、セキュリティスキャン、agent モードにはオプション依存関係が必要：

```bash
pip install "justagent[ai,security]"
```

Python 3.11 以降。Git 2.30+ 推奨。

## クイックスタート

### Agent モード

```bash
cd your-project
justagent agent                    # インタラクティブ REPL（複数ターン会話）
justagent agent "refactor utils"   # ワンショットタスク
justagent agent --plan "..."       # まず計画（読み取り専用、編集なし）
justagent agent --yolo "..."       # すべてのツール呼び出しを自動承認
justagent agent -i --resume <id>   # 保存されたセッションを再開
justagent agent --json "..."       # NDJSON イベントストリーム（自動化用）
```

### Pipeline モード（オプション）

```bash
cd your-project
justagent init
justagent ship                     # clean → verify → commit → upload
```

または各ステージを個別に実行：

```bash
justagent clean
justagent verify
justagent commit
justagent upload
```

### Project モード

```bash
justagent project list             # 管理プロジェクト一覧
justagent project add ./my-app
justagent project run my-app ship  # プロジェクト内でコマンド実行
```

## 設定

JustAgent はプロジェクトルートの `.justagent.toml` を読み取ります：

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

設定値内の環境変数（`${VAR}`）は実行時に展開されます。全オプションは `.justagent.toml.example` を参照。

## AI バックエンド

JustAgent は [LiteLLM](https://github.com/BerriAI/litellm) 経由で LLM 呼び出しをルーティングします。LiteLLM が対応するすべてのプロバイダーを追加設定なしで利用できます——OpenAI、Anthropic、Ollama、OpenRouter、Azure、vLLM、LM Studio、llama.cpp など 100 社以上。

`.justagent.toml` で1つ以上のバックエンドを設定：

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

ゲートウェイはリトライ、レート制限、プロバイダー間の自動フェイルオーバーを処理します。

## Agent 機能

Agent モードは安全制御付きの反復ツール呼び出しループに基づいています：

| 機能 | 説明 |
|------|------|
| **Plan/Act モード** | Plan モードは読み取り専用分析、Act モードは権限プロンプト付きで変更を実行。`--plan` / `--yolo` または REPL で `/mode` で切り替え。 |
| **ツール呼び出し** | 組み込みツール：`read_file`、`write_to_file`、`replace_in_file`、`apply_patch`、`run_command`、`search`、`web_fetch`、`ask_question`。MCP ツールも利用可能。 |
| **セッション永続化** | 会話は `~/.justagent/sessions/` に保存。`--resume <id>` で再開。Slash コマンド：`/tokens`、`/history`、`/diff`。 |
| **権限エンジン** | `allow` / `deny` / `ask` ルール、`once` / `always` スコープとワイルドカード対応。`write_to_file`、`replace_in_file`、`apply_patch`、`run_command` に組み込み済み。 |
| **変更追跡** | 実行中に作成/変更/削除されたファイルを追跡し、行数差分を計算。終了時にサマリテーブルを表示。 |
| **チェックポイント** | 各ツール呼び出し後にシャドー git スナップショットを作成。ファイル、会話、または両方を復元可能。 |
| **コンテキスト圧縮** | 90% コンテキスト予算で長い会話を自動圧縮。ベーシック（切り詰め）またはエージェント（LLM 要約）モード。 |
| **ループ検出** | 繰り返しツール呼び出しを検出（soft=3、hard=5）、反復ループから脱出。 |
| **ミステイクトラッカー** | 連続エラーをカウント、設定に基づき停止または継続。 |
| **Repo Map** | Python/JS/TS/Rust/Go のシンボルを正規表現で抽出、コンパクトなツリーとして整形。 |
| **Skills** | `.justagent/skills/` から `SKILL.md` をロード、段階的開示。 |
| **サブエージェント** | 読み取り専用の並列研究サブエージェントを生成、コンテキストを分離。 |
| **MCP** | Model Context Protocol サーバー（stdio / SSE / HTTP）に接続、OAuth サポート。 |

### Slash コマンド（インタラクティブ REPL 内）

| コマンド | 説明 |
|---------|------|
| `/help` | 利用可能なコマンドを表示 |
| `/mode <act\|plan\|yolo>` | agent モードを切り替え |
| `/tokens` | トークン使用量の内訳を表示 |
| `/history` | 会話履歴を表示 |
| `/diff` | git diff を表示 |
| `/lint` | リンターを実行 |
| `/test` | テストを実行 |
| `/add <path>` | ファイルをコンテキストに追加 |
| `/drop <path>` | ファイルをコンテキストから削除 |
| `/clear` | 会話をクリア |
| `/exit` | REPL を終了 |

## コマンド

| コマンド | 説明 |
|---------|------|
| `justagent agent` | インタラクティブ AI エージェント（REPL またはワンショット） |
| `justagent session` | 保存されたセッションを管理（list / show / resume / delete） |
| `justagent project` | 複数のローカルプロジェクトを管理 |
| `justagent init` | 現在のディレクトリで初期化 |
| `justagent clean` | ソースファイルをフォーマット・リント |
| `justagent verify` | テストスイートとチェックを実行 |
| `justagent commit` | コミットを生成・作成 |
| `justagent upload` | ビルドしてアーティファクトを公開 |
| `justagent ship` | clean、verify、commit、upload を順番に実行 |
| `justagent fix` | AI 駆動のコード修正提案 |
| `justagent config` | 設定の表示・管理 |
| `justagent doctor` | 環境と依存関係を診断 |
| `justagent plugin` | プラグインを管理 |
| `justagent hooks` | ライフサイクルフックを管理 |
| `justagent lsp` | Language Server Protocol 統合 |
| `justagent artifacts` | ビルドアーティファクトを管理 |
| `justagent metrics` | 使用量とコスト指標を表示 |

## 開発

```bash
git clone https://gitcode.com/badhope/justagent.git
cd justagent
uv sync --all-extras
```

テスト、リント、型チェック：

```bash
uv run pytest tests/ -v
uv run ruff check src/ tests/
uv run mypy src/
```

## 技術スタック

| レイヤー | コンポーネント |
|---------|---------------|
| CLI フレームワーク | Typer |
| プラグインシステム | Pluggy |
| AI ゲートウェイ | LiteLLM |
| Agent ループ | 自作（ツール呼び出し反復 + 安全制御） |
| 設定 / schema | Pydantic v2 + pydantic-settings |
| ターミナル出力 | Rich |
| ログ | Structlog（Rich コンソール + JSON ファイル） |
| LSP サーバー | pygls |
| キャッシュ | diskcache |
| メトリクス | prometheus-client |
| HTTP クライアント | httpx |
| リント / フォーマット | Ruff |
| セキュリティスキャン | Semgrep |
| MCP | 公式 `mcp` Python SDK |

## アーキテクチャ

```
src/justagent/
├── cli/                 # Typer コマンド
│   ├── commands/        # コマンドごとに1モジュール（agent, session, project, ...）
│   ├── display.py       # Rich ベースのターミナル出力（スピナー、パネル、diff）
│   └── main.py          # エントリポイント + グローバルオプション
├── agent/               # Agent コア
│   ├── runtime.py       # 反復ツール呼び出しループ（run / continue_run / reset）
│   ├── tools/           # 組み込みツール定義 + 実行器
│   │   ├── base.py      # Tool, ToolContext, ToolResult, request_permission()
│   │   ├── registry.py  # ツール登録
│   │   └── builtin/     # read_file, write_to_file, replace_in_file, apply_patch,
│   │                    # run_command, search, web_fetch, ask_question
│   ├── plan_act.py      # Plan/Act/Yolo モード system prompt + ツールフィルタリング
│   ├── session.py       # セッション永続化（保存 / 再開 / シリアライズ）
│   ├── slash_commands.py # /help /mode /tokens /history /diff /lint /test ...
│   ├── change_tracker.py # ファイル変更追跡（行数差分）
│   ├── loop_detection.py # 繰り返しツール呼び出しループ検出
│   ├── mistake_tracker.py # 連続エラー追跡
│   ├── compaction.py    # コンテキスト圧縮（ベーシック + エージェント）
│   ├── subagent.py      # 読み取り専用並列研究サブエージェント
│   └── mcp_client.py    # Model Context Protocol クライアント
├── checkpoint/          # シャドー git スナップショット
├── context/             # コンテキストエンジニアリング
│   ├── skill.py         # SKILL.md ローダー
│   └── repo_map.py      # 正規表現ベースの repo シンボルマップ
├── permissions/         # ツール権限エンジン（allow / deny / ask ルール）
├── adapters/            # 外部統合
│   ├── providers/       # LLM プロバイダー（統合ゲートウェイ）
│   ├── upload/          # PyPI / Docker / GitHub アップローダー
│   ├── git_adapter.py
│   ├── model_gateway.py
│   └── tool_adapter.py
├── core/                # コアインフラ
│   ├── config_center.py
│   ├── audit_logger.py
│   ├── hook_dispatcher.py
│   ├── plugin_registry.py
│   ├── batch_ops.py     # クロスプロジェクトバッチ操作
│   ├── sandbox.py
│   ├── cache.py
│   └── ...
├── plugins/             # 組み込みプラグイン（security_scan, typecheck, docker_ship, ...）
├── models/              # Pydantic 設定 schema
├── utils/               # 共有ユーティリティ
└── registry/            # プラグインレジストリインデックス
```

## ミラー

- [GitCode](https://gitcode.com/badhope/justagent) — 中国本土ユーザー向け高速クローン

## ライセンス

MIT — [LICENSE](LICENSE) を参照。

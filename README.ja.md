# JustAgent：司法インテリジェンスエージェントプラットフォーム

[English](README.md) | [中文](README.zh.md) | [日本語](README.ja.md)

> **Intelligence for Justice —— 司法を知的にエンパワーする**

JustAgent は司法機関および政府部門を支援する司法インテリジェンスエージェントプラットフォームです。司法職務を補助・自動化し、司法プロセスをインテリジェント化します。複数エージェントの連携ループにより、事件資料の整理、証拠審査、法的文書生成を実行します — すべての操作に権限管理があり、すべての会話を監査可能に保存・再開できます。

## 概要

JustAgent は司法機関・政府部門向けの司法インテリジェンスエージェントプラットフォームで、単一の CLI ツールとして提供されます。反復的なツール呼び出しループを実行し、LLM が事件資料を読み、アクションを提案し、ツールで実行し、結果を検証します。すべての破壊的操作は権限エンジンを経由し、すべての会話は保存・再開できます — 司法プロセスを監査可能かつ制御可能に保ちます。

3つのモードを提供します：

1. **Agent モード** — インタラクティブ REPL またはワンショットタスク。まず計画（事件記録の読み取り専用分析）、次に実行（権限プロンプト付き）、または Yolo モードでツール呼び出しを自動承認。複数ターンの会話はセッション間で永続化されます。
2. **Pipeline モード** — オプションの `clean → verify → commit → upload` ショートカット。司法文書の公開フロー用。
3. **Project モード** — 複数の事件プロジェクトを管理、クロスプロジェクト操作を実行。

## コア機能

| 機能 | 説明 |
|------|------|
| **事件資料の整理** | 事件ファイル（卷宗、調書、証拠リスト）を取り込み、分類・要約。エンティティ、時系列、関係を抽出し、構造化された事件地図として整理し迅速な検討を支援。 |
| **証拠審査** | 証拠の整合性、完全性、証拠能力を交差検証。矛盾、欠落、証拠連鎖の問題をフラグ付け、出典資料に追溯到。 |
| **法的文書生成** | 事件コンテキストから起訴状、判決書、決定書などの法定文書を起草。標準化テンプレートを適用し、管轄対応のフォーマットと条項ライブラリをサポート。 |
| **司法ワークフロー自動化** | 複数ステップの司法手続き（受付 → 審査 → 審理 → 裁判）を編成。オーケストレーション層で専門エージェントを連携させ、完全な監査ログを記録。 |

## アーキテクチャ設計（マルチエージェント連携）

JustAgent はオーケストレーション層で専門エージェントを連携させ、下層に共有ナレッジベースと統一権限エンジンを置きます：

```
┌───────────────────────────────────────────────────────────┐
│                  オーケストレータ / 調整器               │
│        （ワークフロー編成 · 意思決定ルーティング · メッシュ）│
└───────────┬───────────────┬───────────────┬───────────────┘
            │               │               │
            ▼               ▼               ▼
   ┌────────────────┐ ┌──────────────┐ ┌────────────────┐
   │  事件資料整理   │ │  証拠審査    │ │  法的文書生成   │
   │   エージェント  │ │   エージェント│ │   エージェント  │
   └────────┬────────┘ └──────┬───────┘ └───────┬────────┘
            │                 │                 │
            ▼                 ▼                 ▼
   ┌──────────────────────────────────────────────────────┐
   │  ナレッジ層（RAG / ベクトル / ナレッジグラフ / 文書）   │
   │  セキュリティ層（RBAC / SSO / 暗号化 / 監査）         │
   │  権限エンジン · チェックポイント · 監査ログ            │
   └──────────────────────────────────────────────────────┘
```

各専門エージェントは共通のナレッジ層（法令コーパス、事件リポジトリ、証拠ストア）を共有し、統一された権限エンジンの下で動作します。すべてのエージェントアクションは完全な監査ログとともに記録され、チェックポイント化・復元可能で、司法プロセスを透明かつ防御可能に保ちます。

## インストール

```bash
pip install justagent
```

AI 文書生成、証拠分析、agent モードにはオプション依存関係が必要：

```bash
pip install "justagent[ai,security]"
```

Python 3.11 以降。Git 2.30+ 推奨。

## クイックスタート

### Agent モード

```bash
cd your-case-project
justagent agent                    # インタラクティブ REPL（複数ターン会話）
justagent agent "事件 A001 の証拠を審査"   # ワンショットタスク
justagent agent --plan "..."       # まず計画（読み取り専用、編集なし）
justagent agent --yolo "..."       # すべてのツール呼び出しを自動承認
justagent agent -i --resume <id>   # 保存されたセッションを再開
justagent agent --json "..."       # NDJSON イベントストリーム（自動化用）
```

### Pipeline モード（オプション）

```bash
cd your-case-project
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
justagent project list             # 管理事件プロジェクト一覧
justagent project add ./case-2026-001
justagent project run case-2026-001 ship  # プロジェクト内でコマンド実行
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

設定値内の環境変数（`${VAR}`）は実行時に展開されます。全オプション（司法シナリオ例を含む）は `.justagent.toml.example` を参照。

## AI バックエンド

JustAgent は公式 [OpenAI SDK](https://github.com/openai/openai-python) 経由で LLM 呼び出しをルーティングします。サポートするすべてのプロバイダー（OpenAI、Ollama、OpenRouter、Azure、vLLM、LM Studio、llama.cpp）は OpenAI 互換エンドポイントを公開しており、プロバイダーごとの base URL を設定した単一クライアントで追加設定なしにすべてをカバーできます。機密性の高い司法データには、セルフホストバックエンド（Ollama、vLLM、llama.cpp）でデータを域内に留めることができます。

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

## 機能モジュール

Agent モードは安全制御付きの反復ツール呼び出しループに基づいています：

| モジュール | 説明 |
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
| **ナレッジ層** | 文書処理、ETL、ナレッジグラフ、RAG、ベクトルストレージ。事件コーパスと法令参照用。 |
| **オーケストレーション層** | 調整器、意思決定ルーティング、エージェントメッシュ、ワークフローエンジン。マルチエージェント連携を支える。 |
| **セキュリティ・コンプライアンス** | RBAC、SSO、暗号化、データ保護、コンプライアンスチェック。司法・政府ユースに最適化。 |

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
| `justagent project` | 複数の事件プロジェクトを管理 |
| `justagent init` | 現在のディレクトリで初期化 |
| `justagent clean` | ソースファイルをフォーマット・リント |
| `justagent verify` | テストスイートとチェックを実行 |
| `justagent commit` | コミットを生成・作成 |
| `justagent upload` | ビルドしてアーティファクトを公開 |
| `justagent ship` | clean、verify、commit、upload を順番に実行 |
| `justagent fix` | AI 駆動の修正提案 |
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
| マルチエージェントオーケストレーション | Coordinator / mesh / workflow / decision |
| ナレッジ層 | RAG · ベクトル · ナレッジグラフ · 文書 ETL |
| 設定 / schema | Pydantic v2 + pydantic-settings |
| ターミナル出力 | Rich |
| ログ | Structlog（Rich コンソール + JSON ファイル） |
| LSP サーバー | pygls |
| キャッシュ | diskcache |
| メトリクス | prometheus-client |
| HTTP クライアント | httpx |
| リント / フォーマット | Ruff |
| セキュリティスキャン | Semgrep |
| セキュリティ・コンプライアンス | RBAC · SSO · 暗号化 · データ保護 |
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
├── orchestration/       # マルチエージェント連携
│   ├── coordinator.py   # エージェント調整
│   ├── decision.py      # 意思決定ルーティング
│   ├── mesh.py          # エージェントメッシュ
│   └── workflow.py      # 司法ワークフロー編成
├── knowledge/           # 司法ナレッジ層
│   ├── rag.py           # 検索拡張生成
│   ├── vector.py        # ベクトルストレージ
│   ├── graph.py         # ナレッジグラフ
│   ├── document.py      # 文書処理
│   └── etl.py           # 抽出 / 変換 / ロード
├── communication/       # エージェント間・チーム通信
│   ├── audit.py         # 監査トレイル
│   ├── broadcast.py     # ブロードキャスト
│   ├── meeting.py       # 会議 / 審理調整
│   ├── messaging.py     # メッセージング
│   └── notification.py  # 通知
├── security/            # 司法グレードのセキュリティ
│   ├── rbac.py          # ロールベースアクセス制御
│   ├── sso.py           # シングルサインオン
│   ├── encryption.py    # 暗号化
│   ├── data_protection.py # データ保護
│   └── compliance.py    # コンプライアンスチェック
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

Apache-2.0 — [LICENSE](LICENSE) を参照。

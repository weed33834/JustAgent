# AutoShip CLI

ターミナルで完結するデリバリーパイプライン——コード整形、検証、コミット、公開を一つのCLIで。

[English](README.md) · [中文](README.zh.md) · [日本語](README.ja.md)

## 概要

AutoShip はデリバリーパイプラインを4つのステージに分割し、それぞれを独立して実行・設定できます。

1. **Clean** — Ruff によるコード整形、インポート整理、デッドコード除去
2. **Verify** — テストスイートとリンターの実行
3. **Commit** — 指定した LLM による conventional commit メッセージの自動生成
4. **Upload** — PyPI、Docker Hub、または任意のレジストリへの公開

全ステージをまとめて実行することも、必要なステージだけ個別に実行することも可能です。

## インストール

```bash
pip install autoship
```

AI によるコミットメッセージ生成やセキュリティスキャンが必要な場合：

```bash
pip install "autoship[ai,security]"
```

Python 3.11 以上、Git 2.30 以上を推奨。

## クイックスタート

```bash
cd プロジェクトディレクトリ
autoship init
autoship ship
```

ステージごとの実行：

```bash
autoship clean
autoship verify
autoship commit
autoship upload
```

## 設定

プロジェクトルートに `autoship.toml` を配置します：

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

設定値内の `${VAR}` は実行時に環境変数へ展開されます。全設定項目は `autoship.toml.example` を参照してください。

## AI バックエンド

AutoShip は [LiteLLM](https://github.com/BerriAI/litellm) をモデルゲートウェイとして使用しており、LiteLLM が対応する全てのプロバイダ（OpenAI、Anthropic、Ollama、OpenRouter、Azure、vLLM、LM Studio、llama.cpp 他、100以上）をそのまま利用できます。

`autoship.toml` に一つまたは複数のバックエンドを設定：

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

ゲートウェイはリトライ、レート制限、プロバイダ間の自動フェイルオーバーを処理します。

## コマンド一覧

| コマンド | 説明 |
|----------|------|
| `autoship init` | カレントディレクトリに初期化 |
| `autoship clean` | ソースの整形と lint |
| `autoship verify` | テストスイートの実行 |
| `autoship commit` | コミットの生成と作成 |
| `autoship upload` | アーティファクトのビルドと公開 |
| `autoship ship` | 全ステージを順次実行 |
| `autoship fix` | AI によるコード修正提案 |
| `autoship lsp` | 言語サーバーの起動 |
| `autoship plugin` | プラグイン管理 |

## 開発

```bash
git clone https://github.com/MS33834/autoship-cli.git
cd autoship-cli
pip install -e ".[dev]"
```

テスト・lint・型チェック：

```bash
pytest tests/ -v
ruff check src/ tests/
mypy src/
```

## 技術スタック

| レイヤー | コンポーネント |
|----------|----------------|
| CLI フレームワーク | Typer |
| プラグインシステム | Pluggy |
| AI ゲートウェイ | LiteLLM |
| 設定 / スキーマ | Pydantic v2 + pydantic-settings |
| ターミナル出力 | Rich |
| ログ | Structlog（Rich コンソール + JSON ファイル） |
| LSP サーバー | pygls |
| キャッシュ | diskcache |
| メトリクス | prometheus-client |
| HTTP クライアント | httpx |
| Lint / 整形 | Ruff |
| セキュリティスキャン | Semgrep |

## ミラーリポジトリ

- [GitCode](https://gitcode.com/badhope/autoship-cli) — 中国本土からのクローンが高速です

## ライセンス

MIT — 詳細は [LICENSE](LICENSE) を参照。

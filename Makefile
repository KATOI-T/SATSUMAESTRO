# SATSUMAESTRO 開発用ショートカット
# 使い方: `make <target>`（例: make setup / make check / make run ARGS="--version"）
# すべて uv 経由で .venv 内のツールを実行する。

.DEFAULT_GOAL := help

# ── 環境構築 ──────────────────────────────────────────────
.PHONY: setup
setup: ## 依存を同期し（dev 込み）pre-commit フックを導入
	uv sync
	uv run pre-commit install
	@echo "セットアップ完了。'cp .env.example .env' で資格情報を設定してください。"

.PHONY: sync
sync: ## 依存を同期（uv.lock で再現）
	uv sync

# ── 品質チェック ──────────────────────────────────────────
.PHONY: lint
lint: ## ruff で lint
	uv run ruff check .

.PHONY: format
format: ## ruff でコード整形（書き換え）
	uv run ruff format .

.PHONY: fmt-check
fmt-check: ## 整形済みか確認（書き換えない）
	uv run ruff format --check .

.PHONY: type
type: ## mypy で型チェック（strict）
	uv run mypy satsumaestro

.PHONY: test
test: ## pytest
	uv run pytest

.PHONY: check
check: lint fmt-check type test ## lint + format + type + test を一括実行

# 擬似プロバイダ（tests/smoke/fake_ollama.py）に繋いだ本物の REPL を pty で動かし、
# plan → 承認 → 段 → 完走 → 削除 の画面を検査する。GPU 不要・約 30 秒。
# `uv run pytest` は addopts で飛ばすので、-m smoke を明示する。check とは別に全 PR で回す。
.PHONY: smoke
smoke: ## 擬似プロバイダ＋pty の REPL 煙テスト（make check とは別・GPU 不要）
	uv run pytest -m smoke tests/smoke -p no:cacheprovider --durations=3

.PHONY: pre-commit
pre-commit: ## pre-commit を全ファイルに実行
	uv run pre-commit run --all-files

# ── 実行 ──────────────────────────────────────────────────
# モデルは MODEL 変数で指定（未指定なら環境変数 OLLAMA_MODEL を採用）。
# 空のままなら --model を付けず、設定 provider.default_model に委ねる。
MODEL ?= $(OLLAMA_MODEL)
_MODEL_ARG := $(if $(strip $(MODEL)),--model "$(strip $(MODEL))",)

.PHONY: run
run: ## CLI を実行（例: make run ARGS="--version"）
	uv run satsumaestro $(ARGS)

# 注: エージェント機能（ツール実行）には tool calling 対応モデルが必要
#     （例: llama3.1:8b / llama3.3:70b。qwen2.5-coder 等は tool 非対応）。
.PHONY: repl
repl: ## 対話 REPL を起動（例: make repl MODEL=llama3.1:8b）
	uv run satsumaestro repl $(_MODEL_ARG)

.PHONY: prompt
prompt: ## 一発プロンプト（例: make prompt TEXT="フィボナッチを書いて" MODEL=llama3.1:8b）
	uv run satsumaestro prompt "$(TEXT)" $(_MODEL_ARG)

# ── 評価 ──────────────────────────────────────────────────
# 既定は従来のコード解析タスク。WEB_ROUTING=1 のときだけ専用経路へ切り替える。
EVAL_TARGET ?= evals/code_analysis
_EVAL_SELECTOR := $(if $(filter 1,$(WEB_ROUTING)),--web-routing,$(EVAL_TARGET))

.PHONY: eval
eval: ## 評価を実行（Web: make eval WEB_ROUTING=1 MODEL=gpt-oss:120b ARGS="--runs 1"）
	uv run satsumaestro eval $(_EVAL_SELECTOR) $(_MODEL_ARG) $(ARGS)

.PHONY: doctor
doctor: ## 設定・プロバイダ・鍵・ポリシーの健全性を診断
	uv run satsumaestro doctor

.PHONY: version
version: ## バージョンを表示
	uv run satsumaestro --version

# ── web_search バックエンド ───────────────────────────────
# docker compose（v2 プラグイン）が無い環境ではスタンドアロンの docker-compose を使う。
# 既に 8080 を使っている場合は SATSUMA_SEARXNG_PORT で変更（例: make searxng-up SATSUMA_SEARXNG_PORT=8081）。
COMPOSE := $(shell docker compose version >/dev/null 2>&1 && echo "docker compose" || echo "docker-compose")

.PHONY: searxng-up
searxng-up: ## SearXNG 自前ホストを起動（web_search 用。docker/searxng/ 参照）
	$(COMPOSE) -f docker/searxng/compose.yaml up -d
	@echo "設定例: .satsuma/settings.local.json に {\"web_search\": {\"provider\": \"searxng\", \"base_url\": \"http://localhost:$${SATSUMA_SEARXNG_PORT:-8080}\"}}"
	@echo "疎通確認: uv run satsumaestro doctor --probe"

.PHONY: searxng-down
searxng-down: ## SearXNG 自前ホストを停止
	$(COMPOSE) -f docker/searxng/compose.yaml down

# ── 開発フロー計測（おまけ）────────────────────────────────
.PHONY: flow
flow: ## git履歴から開発フロー滞留/手戻りを可視化（例: make flow ARGS="main..HEAD"）
	python3 scripts/flow_report.py $(ARGS)

# ── 後片付け ──────────────────────────────────────────────
.PHONY: clean
clean: ## キャッシュを削除（.venv は残す）
	rm -rf .mypy_cache .ruff_cache .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

.PHONY: distclean
distclean: clean ## .venv も含めて削除
	rm -rf .venv

# ── ヘルプ ────────────────────────────────────────────────
.PHONY: help
help: ## このヘルプを表示
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

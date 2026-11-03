"""設定モデル。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from satsumaestro.i18n.schema import catalog
from satsumaestro.util.paths import AGENT_DIRNAME


class RetryConfig(BaseModel):
    """プロバイダのリトライ・キャッシュ監視の設定。"""

    model_config = ConfigDict(extra="forbid")

    max_retries: int = Field(default=8, ge=0)
    initial_backoff_s: float = Field(default=1.0, gt=0.0)
    max_backoff_s: float = Field(default=16.0, gt=0.0)
    tool_parse_max_retries: int = Field(default=1, ge=0)
    read_timeout_max_retries: int = Field(default=0, ge=0)
    request_timeout_s: float = Field(default=120.0, gt=0.0)
    extend_first_wait: bool = True
    cache_break_min_drop: int = Field(default=2000, ge=0)


ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]


class SamplingConfig(BaseModel):
    """サンプリング制御。``reasoning_effort`` の許容値はモデルごとに違う。"""

    model_config = ConfigDict(extra="forbid")

    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    reasoning_effort: ReasoningEffort | None = None

    def is_unset(self) -> bool:
        return self.temperature is None and self.top_p is None and self.reasoning_effort is None


ROLE_NAMES: tuple[str, ...] = ("summarize", "repair", "subagent", "decompose")


class ModelRoles(BaseModel):
    """役割別モデルルーティング。"""

    model_config = ConfigDict(extra="forbid")

    summarize: str | None = None
    repair: str | None = None
    subagent: str | None = None
    decompose: str | None = None
    sampling_overrides: dict[str, SamplingConfig] = Field(default_factory=dict)

    @field_validator("sampling_overrides")
    @classmethod
    def _known_role_keys(cls, v: dict[str, SamplingConfig]) -> dict[str, SamplingConfig]:
        unknown = sorted(set(v) - set(ROLE_NAMES))
        if unknown:
            raise ValueError(f"unknown role(s) {unknown}: expected one of {list(ROLE_NAMES)}")
        return v


class OllamaConfig(BaseModel):
    """Ollama ネイティブ API 連携。"""

    model_config = ConfigDict(extra="forbid")

    keep_alive: str | None = "30m"
    unload_on_exit: bool = True
    detect: Literal["auto", "on", "off"] = "auto"


class OpenAIConfig(BaseModel):
    """OpenAI 公式 API（``hosts``）向けの調整。``negotiate`` は xAI にも効く（Ollama は除く）。"""

    model_config = ConfigDict(extra="forbid")

    hosts: list[str] = Field(default_factory=lambda: ["api.openai.com"])
    negotiate: bool = True


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_model: str = ""
    fallback_model: str | None = None
    base_url: str | None = None
    retry: RetryConfig = Field(default_factory=RetryConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    roles: ModelRoles = Field(default_factory=ModelRoles)


class PermissionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_mode: Literal[
        "read-only", "workspace-write", "danger-full-access", "prompt", "allow"
    ] = "workspace-write"
    policy_file: str = f"{AGENT_DIRNAME}/policy.md"
    memory_file: str = f"{AGENT_DIRNAME}/memory.md"
    bash_auto_allow: Literal["off", "read-only"] = "read-only"
    secret_access: Literal["ask", "deny"] = "ask"
    secret_patterns: list[str] = Field(default_factory=list)
    secret_patterns_remove: list[str] = Field(default_factory=list)


class SandboxConfig(BaseModel):
    """bash 実行の隔離設定の既定値。"""

    model_config = ConfigDict(extra="forbid")

    filesystem_mode: Literal["off", "workspace-only", "allow-list"] = "workspace-only"
    isolate_network: bool = False
    allowed_mounts: list[str] = Field(default_factory=list)
    env_strip: list[str] = Field(default_factory=list)
    env_keep: list[str] = Field(default_factory=list)


class McpServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transport: Literal["stdio", "streamable-http"]
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    url: str | None = None
    env: dict[str, str] = Field(default_factory=dict)


class RagConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    embed_base_url: str | None = None
    embed_model: str = "bge-m3"
    db_path: str = f"{AGENT_DIRNAME}/rag.db"
    auto_inject: bool = False


class CondenseConfig(BaseModel):
    """要約の前の間引き（``engine/condense.py``）の設定。"""

    model_config = ConfigDict(extra="forbid")

    supersede_enabled: bool = True
    collapse_enabled: bool = True
    cluster_enabled: bool = True
    collapse_threshold: int = Field(default=4, ge=2)
    cluster_min_size: int = Field(default=2, ge=2)
    cluster_similarity_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    cluster_max_messages: int = Field(default=2000, ge=1)


class CompactionConfig(BaseModel):
    """コンパクション設定。"""

    model_config = ConfigDict(extra="forbid")

    keep_last: int = Field(default=6, ge=0)
    use_llm_summary: bool = True
    llm_summary_max_tokens: int = Field(default=1024, ge=1)
    trident: CondenseConfig = Field(default_factory=CondenseConfig)
    auto_context_fit: bool = True
    max_tool_result_share: float = Field(default=0.25, ge=0.0, le=1.0)
    context_window: int = Field(default=0, ge=0)


class ToolSchemaConfig(BaseModel):
    """ツールスキーマ規約と tool_call 破損リカバリの設定。"""

    model_config = ConfigDict(extra="forbid")

    strict_schema: bool = True
    strict_tools: bool = False
    repair_tool_calls: bool = True
    max_repairs_per_turn: int = Field(default=1, ge=0, le=3)


class WebSearchConfig(BaseModel):
    """web_search バックエンドの設定。"""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["duckduckgo", "searxng", "generic"] | None = None
    base_url: str | None = None
    timeout_s: float = Field(default=20.0, gt=0.0)
    max_results: int = Field(default=8, ge=1, le=50)
    preferred_domains: list[str] = Field(default_factory=list)

    @field_validator("provider", mode="before")
    @classmethod
    def _none_word(cls, value: object) -> object:
        return None if isinstance(value, str) and value.strip().lower() == "none" else value


class WebFetchConfig(BaseModel):
    """web_fetch の宛先と大きさの制限。"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    allow_private_hosts: bool = False
    max_bytes: int = Field(default=2_000_000, ge=1)
    max_redirects: int = Field(default=20, ge=0)
    confirm: Literal["first-use", "always", "never"] = "first-use"


class HookConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pre_tool_use: list[str] = Field(default_factory=list)
    post_tool_use: list[str] = Field(default_factory=list)
    post_tool_use_failure: list[str] = Field(default_factory=list)


class LspConfig(BaseModel):
    """LSP 統合の設定。"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    servers: dict[str, list[str]] = Field(default_factory=dict)
    request_timeout_s: float = Field(default=15.0, gt=0.0)
    confirm_start: bool = True


class TelemetryConfig(BaseModel):
    """ターンの出来事を記録帳に残す設定。"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    jsonl_file: str | None = None


class ToolsConfig(BaseModel):
    """ツール露出の設定。"""

    model_config = ConfigDict(extra="forbid")

    deferred_tool_loading: bool = True
    deferred_run_on_first_call: bool = True
    core: list[str] = Field(default_factory=lambda: ["todo_write"])
    syntax_check: bool = True
    syntax_reject_edits: bool = True
    edit_reindent: bool = True
    overwrite_guard: bool = True
    related_code_notes: bool = True
    mkdir_hint: bool = True


class GuidanceConfig(BaseModel):
    """モデルへの動的ガイダンス注入の設定。"""

    model_config = ConfigDict(extra="forbid")

    reminders: bool = True
    empty_final_retry: bool = True
    timeout_recovery: bool = True
    step_timeout_recoveries: int = Field(default=2, ge=1)
    tool_call_rewrite: bool = True
    repetition_abort: bool = True
    plan_steps: bool = True
    plan_exit_nudge: bool = True
    args_only_retry: bool = True
    no_change_report_retry: bool = True
    step_budget: bool = True
    step_budget_lines: int | None = Field(default=None, ge=1)
    freeze_git_context: bool = True
    permission_explain: bool = True


class LanguageConfig(BaseModel):
    """応答言語の制御。"""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["auto", "fixed", "off"] = "auto"
    fixed: str = "ja"
    repair: bool = True
    min_chars: int = Field(default=8, ge=1)
    ui: str | None = None

    @field_validator("fixed")
    @classmethod
    def _defined_language(cls, value: str) -> str:
        if value not in catalog().codes():
            from satsumaestro.i18n.text import unknown_language

            raise ValueError(unknown_language(value))
        return value

    @field_validator("ui")
    @classmethod
    def _screen_language(cls, value: str | None) -> str | None:
        from satsumaestro.i18n.text import screen_languages, unknown_screen_language

        if value is not None and value != "auto" and value not in screen_languages():
            raise ValueError(unknown_screen_language(value))
        return value

    @property
    def chosen_ui(self) -> str | None:
        """選んだ画面の言語（``auto`` と未設定は None）。"""
        return None if self.ui in (None, "auto") else self.ui


class RecoveryConfig(BaseModel):
    """自己回復レシピの設定。"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    max_attempts: int = Field(default=1, ge=0)


class SubagentConfig(BaseModel):
    """サブエージェント委譲の設定。"""

    model_config = ConfigDict(extra="forbid")

    max_depth: int = Field(default=2, ge=1)
    default_model: str = ""
    max_iterations: int = Field(default=32, ge=1)


class WorkerConfig(BaseModel):
    """Worker 制御プレーン（外部 satsumaestro 子プロセス）の設定。"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    trusted_roots: list[str] = Field(default_factory=list)
    auto_recover_prompt_misdelivery: bool = True
    ready_timeout_s: float = Field(default=45.0, gt=0.0)
    terminate_grace_s: float = Field(default=5.0, gt=0.0)
    result_timeout_s: float = Field(default=0.0, ge=0.0)
    max_workers: int = Field(default=4, ge=1)
    command: list[str] = Field(default_factory=list)


class OrchestrationConfig(BaseModel):
    """並列オーケストレーション（バックグラウンド Task / Team / Cron）の設定。"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    max_concurrency: int = Field(default=4, ge=1)
    stalled_after_s: int = Field(default=120, ge=1)
    state_file: str = f"{AGENT_DIRNAME}/orchestration-state.json"
    drain_on_exit: bool = True
    workers: WorkerConfig = Field(default_factory=WorkerConfig)


class DecompositionConfig(BaseModel):
    """自動タスク分解と並列振り分けの設定。"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    min_chars: int = Field(default=40, ge=0)
    min_width: int = Field(default=2, ge=2)
    max_subtasks: int = Field(default=6, ge=2, le=12)
    max_depth: int = Field(default=3, ge=1)
    gain_margin: float = Field(default=0.25, ge=0.0)
    overhead_calls: int = Field(default=1, ge=0)
    per_call_budget_s: float = Field(default=60.0, gt=0)
    provider_parallelism: int | Literal["auto"] = "auto"
    circuit_breaker_losses: int = Field(default=2, ge=1)
    capacity_cache_file: str = f"{AGENT_DIRNAME}/capacity.json"
    calibrate_parallelism: bool = False


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    permission: PermissionConfig = Field(default_factory=PermissionConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    mcp_servers: dict[str, McpServerConfig] = Field(default_factory=dict)
    rag: RagConfig = Field(default_factory=RagConfig)
    hooks: HookConfig = Field(default_factory=HookConfig)
    tool_schema: ToolSchemaConfig = Field(default_factory=ToolSchemaConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    lsp: LspConfig = Field(default_factory=LspConfig)
    web_search: WebSearchConfig = Field(default_factory=WebSearchConfig)
    web_fetch: WebFetchConfig = Field(default_factory=WebFetchConfig)
    guidance: GuidanceConfig = Field(default_factory=GuidanceConfig)
    language: LanguageConfig = Field(default_factory=LanguageConfig)
    recovery: RecoveryConfig = Field(default_factory=RecoveryConfig)
    subagent: SubagentConfig = Field(default_factory=SubagentConfig)
    orchestration: OrchestrationConfig = Field(default_factory=OrchestrationConfig)
    decomposition: DecompositionConfig = Field(default_factory=DecompositionConfig)
    show_thinking: bool = False
    stream_tool_args: bool = True
    planning_guidance: bool = False
    parallel_guidance: bool = False
    api_timeout_s: float = 600.0
    max_iterations: int = 50
    compact_token_threshold: int = 120_000
    compaction: CompactionConfig = Field(default_factory=CompactionConfig)

    def read_timeout_s(self) -> float:
        """1 回の API 呼び出しで「次のバイトが届くまで」に許す秒数。"""
        return min(self.provider.retry.request_timeout_s, self.api_timeout_s)

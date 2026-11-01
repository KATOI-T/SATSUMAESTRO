"""会話ドメインの pydantic モデル。"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.util.paths import AGENT_DIRNAME


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ThinkingBlock(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str | None = None


class ToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


class ToolResultBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    tool_name: str
    output: str
    is_error: bool = False


ContentBlock = Annotated[
    TextBlock | ThinkingBlock | ToolUseBlock | ToolResultBlock,
    Field(discriminator="type"),
]


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0


class WriteChunk(BaseModel):
    """``write_file`` / ``append_file`` 1 回分の分量の記録。"""

    tool: str
    path: str = ""
    ok: bool = True
    chars: int = 0
    lines: int = 0
    bytes: int = 0
    output_tokens: int = 0
    response_s: float = 0.0
    iteration: int = 0


class OmittedBody(BaseModel):
    """会話の中で縮めた書き込みの本文 1 つ。"""

    tool_use_id: str
    tool: str
    path: str = ""
    content: str = ""


class PlanStepRecord(BaseModel):
    """承認済み計画の 1 段の見積と実績。"""

    index: int
    total: int = 0
    estimated_lines: int | None = None
    budget_lines: int = 0
    actual_lines: int = 0
    max_write_lines: int = 0
    writes: int = 0
    max_response_s: float = 0.0
    timeouts: int = 0
    iterations: int = 0
    edits: int = 0
    edited_lines: int = 0

    @property
    def over_budget(self) -> bool:
        """1 回の書き込みが予算を超えた（契約の検算に落ちた）。"""
        return bool(self.budget_lines) and self.max_write_lines > self.budget_lines


class ConversationMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    blocks: list[ContentBlock]
    usage: TokenUsage | None = None


class TodoItem(BaseModel):
    """セッション永続の TODO 1 件。"""

    model_config = ConfigDict(populate_by_name=True)

    content: str = Field(description='The task in imperative form, e.g. "Add tests".')
    active_form: str = Field(
        default="",
        alias="activeForm",
        description='Optional: the same task in present-continuous form, e.g. "Adding tests".',
    )
    status: Literal["pending", "in_progress", "completed"] = Field(
        default="pending", description="Current state of this task."
    )

    @model_validator(mode="before")
    @classmethod
    def _content_from_known_variants(cls, data: Any) -> Any:
        """``content`` が無いときだけ、よく取り違えるキーをその値として読む。"""
        if not isinstance(data, dict) or str(data.get("content") or "").strip():
            return data
        for key in _CONTENT_VARIANTS:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return {**data, "content": value}
        return data


_CONTENT_VARIANTS = ("task", "description", "title", "text", "name")


MainMode = Literal["plan", "code"]
_MAIN_MODES: tuple[MainMode, ...] = ("plan", "code")

PLAN_DIR = f"{AGENT_DIRNAME}/plans"


def plan_file_relpath(session_id: str) -> str:
    """plan mode 中にモデルが書く計画ファイルの workspace 相対パス。"""
    return f"{PLAN_DIR}/{session_id}.md"


class ModeState(BaseModel):
    """REPL の実行モード。"""

    main: MainMode = "code"
    sub: PermissionMode = PermissionMode.WORKSPACE_WRITE

    @property
    def plan_active(self) -> bool:
        return self.main == "plan"

    def label(self) -> str:
        """/mode 表示・ログ用の完全形（``plan`` / ``code:workspace-write``）。"""
        return "plan" if self.main == "plan" else f"code:{self.sub.value}"

    def short_label(self) -> str:
        """プロンプト接頭辞用の省略形。"""
        if self.main == "plan" or self.sub is PermissionMode.WORKSPACE_WRITE:
            return self.main
        return f"code:{self.sub.value}"

    @classmethod
    def parse(cls, spec: str, default_sub: PermissionMode) -> ModeState:
        """``plan`` | ``code`` | ``code:<sub>``（``plan:<sub>`` も許容）を解釈する。"""
        text = spec.strip().lower()
        main_text, _, sub_text = text.partition(":")
        if main_text not in _MAIN_MODES:
            raise ValueError(f"unknown main mode {main_text!r} (expected plan | code)")
        main: MainMode = "plan" if main_text == "plan" else "code"
        sub = default_sub
        if sub_text:
            try:
                sub = PermissionMode(sub_text)
            except ValueError:
                choices = ", ".join(m.value for m in PermissionMode)
                raise ValueError(f"unknown sub mode {sub_text!r} (expected {choices})") from None
        return cls(main=main, sub=sub)


def resolve_initial_mode(
    saved: ModeState | None, spec: str | None, default_sub: PermissionMode
) -> ModeState:
    """起動時のモードを決める（優先順位: ``--mode`` 指定 > resume の保存値 > 既定）。"""
    if spec:
        return ModeState.parse(spec, default_sub)
    if saved is not None:
        return saved
    return ModeState(main="code", sub=default_sub)


class SessionCompaction(BaseModel):
    compacted_at_ms: int
    summary: str
    dropped_messages: int


class Session(BaseModel):
    version: int = 1
    session_id: str
    created_at_ms: int
    updated_at_ms: int
    workspace_root: str
    model: str
    messages: list[ConversationMessage] = Field(default_factory=list)
    prompt_history: list[str] = Field(default_factory=list)
    compaction: SessionCompaction | None = None
    todos: list[TodoItem] = Field(default_factory=list)
    mode: ModeState = Field(default_factory=ModeState)
    plan_steps: list[str] = Field(default_factory=list)
    plan_step_index: int = 0
    plan_step_estimates: list[int | None] = Field(default_factory=list)
    plan_budget_lines: int = 0
    language: str = ""
    git_context: str | None = None
    git_context_compaction: int = 0
    git_seen: str = ""
    last_system_digest: str = ""
    expanded_tools: list[str] = Field(default_factory=list)
    omitted_bodies: list[OmittedBody] = Field(default_factory=list)

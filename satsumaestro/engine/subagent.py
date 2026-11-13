"""サブエージェント基盤: プリセット・子 runtime 工場。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from satsumaestro.config.models import RuntimeConfig
from satsumaestro.engine.compaction import Compactor
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.i18n.state import LanguageState
from satsumaestro.prompt.builder import SystemPromptBuilder, format_tool_catalog
from satsumaestro.prompt.instructions import instruction_tokens
from satsumaestro.providers.base import Provider
from satsumaestro.providers.registry import expand_model_name
from satsumaestro.security.allowlist import PermissionResolver
from satsumaestro.security.policy import PermissionPolicy
from satsumaestro.session.models import Session
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools.builtins.orchestration_tools import ORCHESTRATION_TOOL_NAMES
from satsumaestro.tools.builtins.worker_tools import WORKER_TOOL_NAMES
from satsumaestro.tools.registry import ToolRegistry
from satsumaestro.util.proc import BackgroundProcessRegistry

if TYPE_CHECKING:
    from satsumaestro.mcp.manager import MCPManager
    from satsumaestro.rag.service import RAGService

AGENT_TOOL_NAME = "agent"
CHILD_EXCLUDED_TOOLS: frozenset[str] = (
    frozenset({AGENT_TOOL_NAME}) | ORCHESTRATION_TOOL_NAMES | WORKER_TOOL_NAMES
)

EXPLORE_TOOLS = frozenset(
    {"read_file", "glob_search", "list_directory", "grep_search", "web_fetch", "web_search"}
)
PLAN_TOOLS = EXPLORE_TOOLS | {"todo_write"}

_ROLE = (
    "# Delegated task\n"
    "Another agent handed you one task as its `{key}` helper. There is no user to ask, so "
    "decide with what you can see, and use only the tools listed for you. You have few steps: "
    "search with glob_search / grep_search before opening files one by one, and stop as soon "
    "as you can answer. Only your final message goes back to the other agent, so put the "
    "answer and the file paths it rests on in that message."
)
_ROLE_LOOK_ONLY = (
    " Look but do not change anything: report what you found, with file paths, and leave "
    "every file as it is."
)
_ROLE_PLAN = (
    " Look but do not change anything: return a numbered plan that names the files to edit, "
    "and leave every file as it is."
)


class SubagentPreset(BaseModel):
    """``subagent_type`` 1 種類分の子構成（ツール絞り込みとプロンプト差し替え）。"""

    model_config = ConfigDict(frozen=True)

    key: str
    system_suffix: str
    allowed_tools: frozenset[str] | None = None


_GENERAL = SubagentPreset(
    key="general-purpose", system_suffix=_ROLE.format(key="general-purpose"), allowed_tools=None
)
_EXPLORE = SubagentPreset(
    key="Explore",
    system_suffix=_ROLE.format(key="Explore") + _ROLE_LOOK_ONLY,
    allowed_tools=EXPLORE_TOOLS,
)
_PLAN = SubagentPreset(
    key="Plan", system_suffix=_ROLE.format(key="Plan") + _ROLE_PLAN, allowed_tools=PLAN_TOOLS
)

_STEMS: tuple[tuple[str, SubagentPreset], ...] = (("explor", _EXPLORE), ("plan", _PLAN))


def resolve_subagent_type(raw: str) -> SubagentPreset:
    """``subagent_type`` をプリセットにする。"""
    letters = "".join(ch for ch in raw.lower() if "a" <= ch <= "z")
    for stem, preset in _STEMS:
        if letters.startswith(stem):
            return preset
    return _GENERAL


class NullSessionStore(JSONLSessionStore):
    """子セッションをディスクへ保存しない no-op ストア。"""

    def save(self, s: Session) -> None:
        return None


class RuntimeFactory:
    """親の部材を共有して子 :class:`ConversationRuntime` を生む工場。"""

    def __init__(
        self,
        *,
        provider: Provider,
        model: str,
        config: RuntimeConfig,
        tools: ToolRegistry,
        policy: PermissionPolicy | PermissionResolver,
        hooks: HookRunner,
        cwd: Path,
        provider_builder: Callable[[str], Provider],
        mcp: MCPManager | None = None,
        rag: RAGService | None = None,
        background: BackgroundProcessRegistry | None = None,
        lsp: object | None = None,
        language: LanguageState | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.config = config
        self.tools = tools
        self.policy = policy
        self.hooks = hooks
        self.cwd = cwd
        self.provider_builder = provider_builder
        self.mcp = mcp
        self.rag = rag
        self.background = background
        self.lsp = lsp
        self.language = (
            language if language is not None else LanguageState.from_config(config.language)
        )
        self.env = env

    def use_provider(self, provider: Provider, model: str) -> None:
        """切り替えたプロバイダとモデルを、これから作る子に使う。"""
        self.provider = provider
        self.model = model

    def build_child(
        self, *, preset: SubagentPreset, depth: int, model: str = ""
    ) -> ConversationRuntime:
        """独立コンテキストの子 runtime を構築する。"""
        from satsumaestro.providers.factory import resolve_role_model, resolve_role_sampling

        roles = self.config.provider.roles
        resolved = expand_model_name(
            model or roles.subagent or self.config.subagent.default_model or self.model
        )
        provider = self.provider if resolved == self.model else self.provider_builder(resolved)
        updates: dict[str, object] = {"max_iterations": self.config.subagent.max_iterations}
        override = roles.sampling_overrides.get("subagent")
        if override is not None:
            updates["provider"] = self.config.provider.model_copy(update={"sampling": override})
        child_cfg = self.config.model_copy(update=updates)
        sum_model = resolve_role_model(self.config, "summarize", default=resolved)
        sum_provider = provider if sum_model == resolved else self.provider_builder(sum_model)
        child_tools = self.tools.filtered(preset.allowed_tools, exclude=CHILD_EXCLUDED_TOOLS)
        child_session = new_session(self.cwd, resolved)
        return ConversationRuntime(
            session=child_session,
            provider=provider,
            tools=child_tools,
            policy=self.policy,
            prompt_builder=SystemPromptBuilder(
                static_suffix=preset.system_suffix,
                instruction_tokens=instruction_tokens(child_cfg.compaction.context_window),
                tool_catalog=(
                    format_tool_catalog(
                        child_tools.deferred_catalog(),
                        run_on_first_call=child_cfg.tools.deferred_run_on_first_call,
                    )
                    if child_cfg.tools.deferred_tool_loading
                    else ""
                ),
            ),
            store=NullSessionStore(),
            hooks=self.hooks,
            config=child_cfg,
            compactor=Compactor.from_config(
                child_cfg,
                provider=sum_provider,
                model=sum_model,
                sampling=resolve_role_sampling(self.config, "summarize"),
            ),
            mcp=self.mcp,
            rag=self.rag,
            runtime_factory=self,
            depth=depth,
            background=self.background,
            lsp=self.lsp,
            language=self.language.for_child(),
        )

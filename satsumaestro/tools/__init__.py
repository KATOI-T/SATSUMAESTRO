"""ツール層: ツール仕様・レジストリ・dispatch・組み込みツール。"""

from __future__ import annotations

from collections.abc import Iterable

from satsumaestro.config.models import RuntimeConfig
from satsumaestro.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from satsumaestro.tools.builtins import web_fetch, web_search
from satsumaestro.tools.builtins.agent import Agent
from satsumaestro.tools.builtins.append_file import AppendFile
from satsumaestro.tools.builtins.bash import Bash
from satsumaestro.tools.builtins.bash_output import BashOutput
from satsumaestro.tools.builtins.delete_path import DeletePath
from satsumaestro.tools.builtins.edit_file import EditFile
from satsumaestro.tools.builtins.glob_search import GlobSearch
from satsumaestro.tools.builtins.grep_search import GrepSearch
from satsumaestro.tools.builtins.list_directory import ListDirectory
from satsumaestro.tools.builtins.lsp import Lsp
from satsumaestro.tools.builtins.notebook_edit import NotebookEdit
from satsumaestro.tools.builtins.plan_mode import (
    ENTER_PLAN_DESC_GUIDED,
    EnterPlanMode,
    ExitPlanMode,
)
from satsumaestro.tools.builtins.read_file import ReadFile
from satsumaestro.tools.builtins.todo_write import TodoWrite
from satsumaestro.tools.builtins.tool_search import ToolSearch
from satsumaestro.tools.builtins.web_fetch import WebFetch
from satsumaestro.tools.builtins.web_search import WebSearch
from satsumaestro.tools.builtins.write_file import WriteFile
from satsumaestro.tools.registry import ToolRegistry

_BUILTINS: tuple[type[Tool], ...] = (
    ReadFile,
    GlobSearch,
    GrepSearch,
    WriteFile,
    EditFile,
    Bash,
    WebFetch,
    WebSearch,
    TodoWrite,
    EnterPlanMode,
    ExitPlanMode,
    NotebookEdit,
    Agent,
    BashOutput,
    ToolSearch,
    Lsp,
    DeletePath,
    AppendFile,
    ListDirectory,
)


def unavailable_tools(config: RuntimeConfig) -> dict[str, str]:
    """設定で使えない組み込みツールと、それでも名前で呼ばれたときに返す文。"""
    off: dict[str, str] = {}
    if not config.web_fetch.enabled:
        off[WebFetch.spec.name] = web_fetch.TURNED_OFF
    if not web_search.is_ready(config.web_search):
        off[WebSearch.spec.name] = web_search.TURNED_OFF
    return off


def default_registry(
    *,
    strict_schema: bool = True,
    orchestration: bool = False,
    workers: bool = False,
    core_override: Iterable[str] = (),
    planning_guidance: bool = False,
    unavailable: Iterable[str] = (),
) -> ToolRegistry:
    """組み込みツールを登録した新しいレジストリを返す。"""
    core = set(core_override)
    skip = set(unavailable)
    descriptions: dict[str, str] = {}
    if planning_guidance:
        core.add(EnterPlanMode.spec.name)
        descriptions[EnterPlanMode.spec.name] = ENTER_PLAN_DESC_GUIDED
    if WebFetch.spec.name in skip:
        descriptions[WebSearch.spec.name] = web_search.DESCRIPTION_WITHOUT_FETCH
    registry = ToolRegistry(
        strict_schema=strict_schema, core_override=core, description_override=descriptions
    )
    for tool in _BUILTINS:
        if tool.spec.name not in skip:
            registry.register(tool)
    if orchestration:
        from satsumaestro.tools.builtins.orchestration_tools import ORCHESTRATION_TOOLS

        for tool in ORCHESTRATION_TOOLS:
            registry.register(tool)
    if workers:
        from satsumaestro.tools.builtins.worker_tools import WORKER_TOOLS

        for tool in WORKER_TOOLS:
            registry.register(tool)
    return registry


__all__ = [
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "default_registry",
    "unavailable_tools",
]

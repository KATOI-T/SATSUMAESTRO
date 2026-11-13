"""ツール仕様・実行文脈・Protocol。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from satsumaestro.config.models import RuntimeConfig
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.jail import WorkspaceJail
from satsumaestro.session.models import Session, TokenUsage
from satsumaestro.tools.read_ledger import ReadLedger
from satsumaestro.util.proc import BackgroundProcessRegistry


class ToolSpec(BaseModel):
    """ツールの静的メタ情報。"""

    name: str
    description: str
    input_schema: dict[str, Any]
    required_mode: PermissionMode
    parallel_safe: bool = False
    deferred: bool = False


class ToolResult(BaseModel):
    """ツール実行結果。"""

    output: str
    is_error: bool = False
    usage: TokenUsage | None = None
    flags: list[str] = Field(default_factory=list)


class ToolContext(BaseModel):
    """ツール実行時に注入する依存。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    workspace_root: Path
    jail: WorkspaceJail
    config: RuntimeConfig
    base_mode: PermissionMode | None = None
    session: Session | None = None
    mcp: object | None = None
    rag: object | None = None
    runtime_factory: object | None = None
    depth: int = 0
    orchestrator: object | None = None
    background: BackgroundProcessRegistry | None = None
    registry: object | None = None
    lsp: object | None = None
    reads: ReadLedger | None = None
    auto_allowed: bool = False
    env: Mapping[str, str] | None = None


@runtime_checkable
class Tool(Protocol):
    """組み込み・MCP ツールの共通 Protocol（1 ツール = 1 クラス）。"""

    spec: ClassVar[ToolSpec]
    Input: ClassVar[type[BaseModel]]

    async def run(self, inp: BaseModel, ctx: ToolContext) -> ToolResult: ...

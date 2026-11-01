"""ツール入出力のモデル追従性強化のテスト。"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel, Field

from satsumaestro.config.models import GuidanceConfig, HookConfig, RuntimeConfig, ToolSchemaConfig
from satsumaestro.engine.events import MessageStop, TextDelta
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.engine.tool_call_repair import (
    ToolCallRepairPolicy,
    ToolCallRepairState,
    hint_message,
)
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import ToolRegistry, default_registry
from satsumaestro.tools.base import ToolContext, ToolSpec
from satsumaestro.tools.builtins.read_file import ReadFile, _slice
from satsumaestro.tools.schema import (
    _MAX_PROPERTIES,
    SchemaLintError,
    lint_tool_schema,
    model_tool_schema,
)
from satsumaestro.util.errors import ProviderError
from tests.fixtures.mock_provider import ScriptedFailureProvider

_TEXT = "l1\nl2\nl3\nl4\nl5\n"


def _nodes(schema: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """スキーマ内の全ノード（``$defs`` / ``items`` の下も含む）。"""
    yield schema
    for key, value in schema.items():
        if key in ("properties", "$defs") and isinstance(value, dict):
            for sub in value.values():
                yield from _nodes(sub)
        elif key in ("anyOf", "oneOf", "allOf") and isinstance(value, list):
            for sub in value:
                yield from _nodes(sub)
        elif key in ("items", "additionalProperties") and isinstance(value, dict):
            yield from _nodes(value)


def _builtin_schemas() -> list[tuple[str, dict[str, Any]]]:
    return [(e["name"], e["input_schema"]) for e in default_registry().json_schemas()]


def test_schema_has_no_nullable() -> None:
    """破損の起点だった ``anyOf``（nullable）が全組み込みツールから消えている。"""
    for name, schema in _builtin_schemas():
        for node in _nodes(schema):
            assert "anyOf" not in node, name
            assert "oneOf" not in node, name
            assert not (isinstance(node.get("type"), list) and "null" in node["type"]), name


def test_every_property_has_description() -> None:
    for name, schema in _builtin_schemas():
        for node in _nodes(schema):
            for prop_name, prop in (node.get("properties") or {}).items():
                desc = prop.get("description")
                assert isinstance(desc, str) and desc.strip(), f"{name}.{prop_name}"


def test_property_count_within_limit() -> None:
    for name, schema in _builtin_schemas():
        assert len(schema.get("properties") or {}) <= _MAX_PROPERTIES, name


def test_schema_drops_titles_and_null_defaults() -> None:
    class Input(BaseModel):
        path: str = Field(description="p")
        limit: int | None = Field(default=None, description="l")

    schema = model_tool_schema(Input)
    assert "title" not in schema
    assert schema["properties"]["limit"] == {"description": "l", "type": "integer"}
    assert schema["required"] == ["path"]


def test_schema_folds_nullable_without_default() -> None:
    """既定値のない ``X | None`` も畳み、``required`` から外す。"""

    class Input(BaseModel):
        offset: int | None = Field(description="o")

    schema = model_tool_schema(Input)
    assert schema["properties"]["offset"]["type"] == "integer"
    assert "required" not in schema


def test_schema_inlines_nested_defs() -> None:
    """``todo_write`` のネスト構造は保ち、``$ref`` は実体を埋め込む。"""
    by_name = dict(_builtin_schemas())
    todos = by_name["todo_write"]
    items = todos["properties"]["todos"]["items"]
    assert "$defs" not in todos and "$ref" not in items
    assert items["type"] == "object" and items["required"] == ["content"]
    assert set(items["properties"]) == {"content", "activeForm", "status"}
    assert items["properties"]["content"]["description"].startswith("The task in imperative form")


def test_schema_keeps_self_reference() -> None:
    """自己参照だけは ``$ref`` のまま残し、``$defs`` も残す（展開すると無限に膨らむ）。"""

    class Node(BaseModel):
        label: str = Field(description="l")
        children: list[Node] = Field(default_factory=list, description="c")

    class Input(BaseModel):
        root: Node = Field(description="r")

    schema = model_tool_schema(Input)
    root = schema["properties"]["root"]
    assert root["type"] == "object" and set(root["properties"]) == {"label", "children"}
    assert root["properties"]["children"]["items"] == {"$ref": "#/$defs/Node"}
    assert "Node" in schema["$defs"]


def test_schema_keeps_property_named_title() -> None:
    """``title`` キーワード除去がプロパティ名 ``title`` を巻き込まない。"""

    class Input(BaseModel):
        title: str = Field(description="t")

    assert "title" in model_tool_schema(Input)["properties"]


class _NullableTool:
    """規約違反（nullable 露出）のツール。"""

    class Input(BaseModel):
        value: int | None = Field(default=None, description="v")

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="nullable_tool",
        description="violates the schema rules",
        input_schema=Input.model_json_schema(),
        required_mode=PermissionMode.READ_ONLY,
    )

    async def run(self, inp: BaseModel, ctx: ToolContext) -> Any: ...


def test_lint_rejects_nullable_schema() -> None:
    with pytest.raises(SchemaLintError, match="anyOf"):
        ToolRegistry().register(_NullableTool)  # type: ignore[arg-type]


def test_lint_rejects_missing_description() -> None:
    class Input(BaseModel):
        path: str

    with pytest.raises(SchemaLintError, match="description"):
        lint_tool_schema("t", model_tool_schema(Input))


def test_lint_rejects_too_many_properties() -> None:
    fields = {f"f{i}": (str, Field(description="d")) for i in range(_MAX_PROPERTIES + 1)}
    schema = {
        "type": "object",
        "properties": {k: {"type": "string", "description": "d"} for k in fields},
    }
    with pytest.raises(SchemaLintError, match="exceeds the limit"):
        lint_tool_schema("t", schema)


def test_lint_rejects_unknown_required() -> None:
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string", "description": "d"}},
        "required": ["a", "ghost"],
    }
    with pytest.raises(SchemaLintError, match="ghost"):
        lint_tool_schema("t", schema)


def test_lint_downgraded_for_mcp(caplog: pytest.LogCaptureFixture) -> None:
    """スキーマを制御できない外部ツールは warning で登録が通る（MCP 経路）。"""
    reg = ToolRegistry()
    with caplog.at_level(logging.WARNING):
        reg.register(_NullableTool, strict=False)  # type: ignore[arg-type]
    assert reg.get("nullable_tool") is not None
    assert "nullable_tool" in caplog.text


def _legacy_slice(text: str, offset: int | None, limit: int | None) -> str:
    """番兵化より前の実装（等価性の基準）。"""
    if offset is None and limit is None:
        return text
    lines = text.splitlines(keepends=True)
    start = (offset - 1) if offset and offset > 0 else 0
    end = (start + limit) if limit is not None else None
    return "".join(lines[start:end])


def test_read_file_sentinel_defaults() -> None:
    inp = ReadFile.Input(path="x.py")
    assert (inp.offset, inp.limit) == (1, 0)
    assert _slice(_TEXT, inp.offset, inp.limit) == _TEXT
    assert _slice(_TEXT, 3, 0) == "l3\nl4\nl5\n"
    assert _slice(_TEXT, 1, 2) == "l1\nl2\n"


def test_read_file_slice_equivalence() -> None:
    """番兵（0/1）と旧 ``None`` 表現が同じ出力になる。"""
    assert _slice(_TEXT, 1, 0) == _legacy_slice(_TEXT, None, None)
    for offset in (1, 2, 5, 9):
        assert _slice(_TEXT, offset, 0) == _legacy_slice(_TEXT, offset, None)
        for limit in (1, 2, 10):
            assert _slice(_TEXT, offset, limit) == _legacy_slice(_TEXT, offset, limit)
    assert _slice(_TEXT, 0, 2) == _legacy_slice(_TEXT, None, 2)


def test_read_file_rejects_negative_offset() -> None:
    with pytest.raises(ValueError):
        ReadFile.Input(path="x.py", offset=-1)


def _parse_failure() -> ProviderError:
    return ProviderError("error parsing tool call", status_code=500)


def _make_runtime(provider: object, tmp_path: Path, config: RuntimeConfig) -> ConversationRuntime:
    return ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=provider,  # type: ignore[arg-type]
        tools=default_registry(),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=config,
    )


async def test_repair_fires_once_on_parse_error(tmp_path: Path) -> None:
    provider = ScriptedFailureProvider([_parse_failure(), [TextDelta(text="ok"), MessageStop()]])
    rt = _make_runtime(provider, tmp_path, RuntimeConfig())
    summary = await rt.run_turn("hi")

    assert summary.final_text == "ok"
    assert provider.attempts == 2
    assert len(provider.requests[0].messages) == 1
    assert provider.requests[1].messages[-1] == hint_message()


async def test_repair_not_fired_for_other_errors(tmp_path: Path) -> None:
    """503 など他の失敗では発火しない（``is_tool_call_parse_error`` で判定するため）。"""
    provider = ScriptedFailureProvider(
        [ProviderError("service unavailable", status_code=503), [MessageStop()]]
    )
    rt = _make_runtime(provider, tmp_path, RuntimeConfig())
    with pytest.raises(ProviderError):
        await rt.run_turn("hi")
    assert provider.attempts == 1


async def test_repair_hint_not_persisted(tmp_path: Path) -> None:
    """誘導が ``session.messages`` に残らない（プロンプトキャッシュのプレフィクス保護）。"""
    provider = ScriptedFailureProvider([_parse_failure(), [TextDelta(text="ok"), MessageStop()]])
    rt = _make_runtime(provider, tmp_path, RuntimeConfig())
    await rt.run_turn("hi")

    texts = [b.text for m in rt.session.messages for b in m.blocks if hasattr(b, "text")]
    assert not any("malformed JSON arguments" in t for t in texts)
    assert [m.role for m in rt.session.messages] == ["user", "assistant"]


async def test_repair_gives_up_after_limit(tmp_path: Path) -> None:
    """上限を超えた破損はそのまま送出する（自己回復のエスカレーション経路へ渡す）。"""
    provider = ScriptedFailureProvider([_parse_failure()])
    rt = _make_runtime(provider, tmp_path, RuntimeConfig())
    with pytest.raises(ProviderError):
        await rt.run_turn("hi")
    assert provider.attempts == 3


async def test_repair_gives_up_after_limit_without_the_rewrite(tmp_path: Path) -> None:
    """書き直しの依頼を切れば、初回 + 誘導付きの再送 1 回で打ち切る（以前の動き）。"""
    provider = ScriptedFailureProvider([_parse_failure()])
    config = RuntimeConfig(guidance=GuidanceConfig(tool_call_rewrite=False))
    rt = _make_runtime(provider, tmp_path, config)
    with pytest.raises(ProviderError):
        await rt.run_turn("hi")
    assert provider.attempts == 2


async def test_repair_disabled_by_config(tmp_path: Path) -> None:
    provider = ScriptedFailureProvider([_parse_failure()])
    config = RuntimeConfig(tool_schema=ToolSchemaConfig(repair_tool_calls=False))
    rt = _make_runtime(provider, tmp_path, config)
    with pytest.raises(ProviderError):
        await rt.run_turn("hi")
    assert provider.attempts == 2
    assert all(hint_message() not in r.messages for r in provider.requests)


def test_repair_state_counts_per_section() -> None:
    state = ToolCallRepairState(ToolCallRepairPolicy(max_repairs_per_turn=1))
    failure = _parse_failure()
    assert state.should_repair(failure) is True
    state.window += 1
    assert state.should_repair(failure) is False
    state.new_section()
    assert state.should_repair(failure) is True

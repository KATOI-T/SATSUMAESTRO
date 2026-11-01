"""ツールレジストリ。"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

from satsumaestro.i18n.text import ui
from satsumaestro.tools.aliases import ALIAS_TABLE, alias_target, name_key, name_words
from satsumaestro.tools.base import Tool, ToolContext, ToolResult
from satsumaestro.tools.schema import SchemaLintError, lint_tool_schema
from satsumaestro.util.errors import ToolError

_log = logging.getLogger(__name__)


def _schema_entry(tool: type[Tool], description: str | None = None) -> dict[str, Any]:
    return {
        "name": tool.spec.name,
        "description": description or tool.spec.description,
        "input_schema": tool.spec.input_schema,
    }


if TYPE_CHECKING:
    from satsumaestro.engine.events import ToolUse


class ToolRegistry:
    """ツールクラスの登録簿と dispatch。"""

    def __init__(
        self,
        *,
        strict_schema: bool = True,
        core_override: Iterable[str] = (),
        description_override: Mapping[str, str] | None = None,
    ) -> None:
        self._tools: dict[str, type[Tool]] = {}
        self._strict = strict_schema
        self._core_override = set(core_override)
        self._description_override = dict(description_override or {})

    def _describe(self, name: str) -> str | None:
        """設定由来の説明文（無ければ ``None`` = spec の説明をそのまま使う）。"""
        return self._description_override.get(name)

    def register(self, tool: type[Tool], *, strict: bool | None = None) -> None:
        """ツールクラスを ``spec.name`` で登録する。"""
        enforce = self._strict if strict is None else strict
        try:
            lint_tool_schema(tool.spec.name, tool.spec.input_schema)
        except SchemaLintError as e:
            if enforce:
                raise
            _log.warning("tool schema lint (non-strict, registered anyway): %s", e)
        self._tools[tool.spec.name] = tool

    def filtered(
        self, allowed: Iterable[str] | None = None, *, exclude: Iterable[str] = ()
    ) -> ToolRegistry:
        """登録済みツールの部分集合を持つ**新しい**レジストリを返す。"""
        allowed_set = None if allowed is None else set(allowed)
        excluded = set(exclude)
        out = ToolRegistry(
            strict_schema=self._strict,
            core_override=self._core_override,
            description_override=self._description_override,
        )
        out._tools = {
            name: tool
            for name, tool in self._tools.items()
            if (allowed_set is None or name in allowed_set) and name not in excluded
        }
        return out

    def get(self, name: str) -> type[Tool] | None:
        """登録済みツールクラスを返す。"""
        return self._tools.get(name)

    def description(self, name: str) -> str:
        """モデルへ見せている説明文（設定の上書きを反映。未登録は空文字）。"""
        tool = self._tools.get(name)
        if tool is None:
            return ""
        return self._describe(name) or tool.spec.description

    def _require(self, name: str) -> type[Tool]:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(f"unknown tool: {name}")
        return tool

    def json_schemas(self) -> list[dict[str, Any]]:
        """各ツールの ``{name, description, input_schema}`` を返す（ApiRequest.tools 用）。"""
        return [_schema_entry(t, self._describe(n)) for n, t in self._tools.items()]

    def is_deferred(self, name: str) -> bool:
        """``name`` が遅延ツールか（``tools.core`` の強制コア化を反映）。"""
        tool = self._tools.get(name)
        if tool is None:
            return False
        return tool.spec.deferred and name not in self._core_override

    def core_names(self) -> list[str]:
        """コアツール名（登録順）。"""
        return [n for n in self._tools if not self.is_deferred(n)]

    def deferred_names(self) -> list[str]:
        """遅延ツール名（登録順）。"""
        return [n for n in self._tools if self.is_deferred(n)]

    def core_schemas(self) -> list[dict[str, Any]]:
        """コアツールの schema（常時露出分）。"""
        return [
            _schema_entry(t, self._describe(n))
            for n, t in self._tools.items()
            if not self.is_deferred(n)
        ]

    def schemas_for(self, expanded: set[str]) -> list[dict[str, Any]]:
        """コア + ``expanded`` に含まれる遅延ツールの schema（露出集合。登録順）。"""
        return [
            _schema_entry(t, self._describe(n))
            for n, t in self._tools.items()
            if not self.is_deferred(n) or n in expanded
        ]

    def deferred_catalog(self) -> list[dict[str, str]]:
        """遅延ツールの ``{name, description, required}``（schema なし。カタログ提示用）。"""
        return [
            {
                "name": t.spec.name,
                "description": self._describe(n) or t.spec.description,
                "required": ", ".join(
                    k for k, f in t.Input.model_fields.items() if f.is_required()
                ),
            }
            for n, t in self._tools.items()
            if self.is_deferred(n)
        ]

    def search(self, query: str, max_results: int = 5) -> list[str]:
        """遅延ツールを名前か語で探し、見つかったツール名を返す（多くて ``max_results`` 件）。"""
        limit = max(max_results, 1)
        catalog = self.deferred_catalog()
        text = query.strip()
        if text.lower().startswith("select:"):
            return _select(text[len("select:") :], [e["name"] for e in catalog])[:limit]
        required = [t[1:] for t in text.split() if t.startswith("+") and name_key(t)]
        optional = [t for t in text.split() if not t.startswith("+") and name_key(t)]
        if not required and not optional:
            return sorted(e["name"] for e in catalog)[:limit]
        ranked: list[tuple[int, str]] = []
        for entry in catalog:
            points = [_term_points(k, entry["name"], entry["description"]) for k in required]
            if not all(points):
                continue
            total = sum(points) + sum(
                _term_points(k, entry["name"], entry["description"]) for k in optional
            )
            if total:
                ranked.append((total, entry["name"]))
        ranked.sort(key=lambda pair: (-pair[0], pair[1]))
        return [name for _, name in ranked[:limit]]

    def resolve_allowed_tools(self, values: list[str]) -> set[str] | None:
        """``--allowed-tools`` の値を、登録してあるツール名の集合にする。"""
        if not values:
            return None
        by_key = {name_key(name): name for name in self._tools}
        aliases = {alias: tool for alias, tool in ALIAS_TABLE.items() if tool in self._tools}
        for alias, tool in aliases.items():
            by_key.setdefault(name_key(alias), tool)
        tokens = [token for value in values for token in value.replace(",", " ").split()]
        if not tokens:
            given = " ".join(values)
            raise ToolError(ui("tools.registry.allowed_tools_has_no_tool", given=given))
        unknown = [token for token in tokens if name_key(token) not in by_key]
        if unknown:
            names, known = ", ".join(unknown), ", ".join(sorted(self._tools))
            others = ", ".join(
                ui("tools.registry.alias_pair", alias=alias, tool=tool)
                for alias, tool in sorted(aliases.items())
            )
            raise ToolError(
                ui(
                    "tools.registry.allowed_tools_has_unknown_tool",
                    names=names,
                    known=known,
                    others=others,
                )
            )
        return {by_key[name_key(token)] for token in tokens}

    def validate_input(self, call: ToolUse) -> BaseModel:
        """``ToolUse.input`` をツールの ``Input`` へ型付きデシリアライズする。"""
        tool = self._require(call.name)
        try:
            return tool.Input.model_validate(call.input)
        except ValidationError as e:
            raise ToolError(describe_validation_error(call.name, tool.Input, e)) from e

    async def run(self, name: str, inp: BaseModel, ctx: ToolContext) -> ToolResult:
        """ツールを 1 回実行する。"""
        tool = self._require(name)
        return await tool().run(inp, ctx)


def describe_validation_error(name: str, model: type[BaseModel], err: ValidationError) -> str:
    """入力検証エラーを、欠けた項目名と項目の一覧つきの短い文にする。"""
    missing: list[str] = []
    others: list[str] = []
    for item in err.errors():
        field = ".".join(str(p) for p in item.get("loc", ())) or "(input)"
        if item.get("type") == "missing":
            missing.append(field)
        else:
            others.append(f"field `{field}`: {item.get('msg', 'invalid value')}")
    fields = model.model_fields
    required = [k for k, f in fields.items() if f.is_required()]
    optional = [k for k, f in fields.items() if not f.is_required()]
    head = f"invalid input for tool {name}"
    if missing:
        names = ", ".join(f"`{m}`" for m in missing)
        head += f": missing required field{'s' if len(missing) > 1 else ''} {names}"
    lines = [head + ("." if missing else ":"), *others]
    expected = f"Required fields: {', '.join(required) or '(none)'}."
    if optional:
        expected += f" Optional: {', '.join(optional)}."
    lines.append(expected)
    return "\n".join(lines)


def _select(listed: str, names: list[str]) -> list[str]:
    """``select:`` の名前の並びを、書いた順のツール名にする（重複と知らない名前は落とす）。"""
    by_key = {name_key(name): name for name in names}
    picked: list[str] = []
    for part in listed.split(","):
        found = by_key.get(name_key(part)) or by_key.get(name_key(alias_target(part) or ""))
        if found is not None and found not in picked:
            picked.append(found)
    return picked


def _term_points(term: str, name: str, description: str) -> int:
    """検索の語 1 つが、ツール 1 つにどれだけ当たるか（当たらなければ 0）。"""
    key = name_key(term)
    words = name_words(name)
    if key == name_key(name):
        return 10
    if key in words:
        return 5
    if any(word.startswith(key) for word in words) or key in name_key(name):
        return 3
    described = name_words(description)
    run = name_words(term)
    if len(run) > 1:
        hit = any(described[i : i + len(run)] == run for i in range(len(described)))
    else:
        hit = key in described or (len(key) >= 4 and any(w.startswith(key) for w in described))
    return 1 if hit else 0

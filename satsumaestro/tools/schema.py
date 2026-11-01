"""ツールスキーマの生成規約と静的検査。"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel

from satsumaestro.util.errors import ToolError

_log = logging.getLogger(__name__)

_MAX_PROPERTIES = 8

_MAP_KEYWORDS = frozenset({"properties", "$defs", "definitions", "patternProperties"})
_LIST_KEYWORDS = frozenset({"anyOf", "oneOf", "allOf", "prefixItems"})
_NODE_KEYWORDS = frozenset({"items", "additionalProperties", "not", "contains"})


class SchemaLintError(ToolError):
    """スキーマ健全性検査の違反（strict では起動時に落とす）。"""


def model_tool_schema(model: type[BaseModel]) -> dict[str, Any]:
    """``Input`` から「モデルが埋めやすい」JSON Schema を生成する。"""
    return _inline_refs(_rewrite(model.model_json_schema()))


_REF_PREFIX = "#/$defs/"


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """``$ref`` を ``$defs`` の実体で置き換える（ネストは保ったまま間接参照だけを無くす）。"""
    defs = schema.get("$defs")
    if not isinstance(defs, dict):
        return schema

    def walk(node: Any, active: frozenset[str]) -> Any:
        if isinstance(node, list):
            return [walk(v, active) for v in node]
        if not isinstance(node, dict):
            return node
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith(_REF_PREFIX):
            name = ref[len(_REF_PREFIX) :]
            target = defs.get(name)
            if isinstance(target, dict) and name not in active:
                rest = {k: v for k, v in node.items() if k != "$ref"}
                return walk({**target, **rest}, active | {name})
            return node
        return {k: walk(v, active) for k, v in node.items()}

    out = {k: walk(v, frozenset()) for k, v in schema.items() if k != "$defs"}
    if _has_ref(out):
        out["$defs"] = defs
    return out


def _has_ref(node: Any) -> bool:
    if isinstance(node, dict):
        return "$ref" in node or any(_has_ref(v) for v in node.values())
    if isinstance(node, list):
        return any(_has_ref(v) for v in node)
    return False


def lint_tool_schema(name: str, schema: dict[str, Any]) -> None:
    """登録時の静的検査。"""
    props = schema.get("properties") or {}
    if not isinstance(props, dict):
        raise SchemaLintError(f"{name}: properties must be an object")
    if len(props) > _MAX_PROPERTIES:
        raise SchemaLintError(
            f"{name}: {len(props)} properties exceeds the limit of {_MAX_PROPERTIES}; "
            f"fold mutually exclusive arguments into one"
        )
    required = schema.get("required") or []
    if isinstance(required, list):
        unknown = [r for r in required if r not in props]
        if unknown:
            raise SchemaLintError(f"{name}: required refers to unknown properties: {unknown}")

    for owner, prop_name, prop in _properties_of(schema):
        where = f"{owner}.{prop_name}" if owner else prop_name
        desc = prop.get("description")
        if not isinstance(desc, str) or not desc.strip():
            raise SchemaLintError(
                f"{name}: property {where} has no description; "
                f"give every argument a Field(description=...)"
            )

    for node in _nodes_of(schema):
        for keyword in ("anyOf", "oneOf"):
            if keyword in node:
                raise SchemaLintError(
                    f"{name}: schema exposes {keyword}; use a concrete type with a sentinel "
                    f"default instead of a nullable/union argument"
                )
        if _null_in_type(node.get("type")):
            raise SchemaLintError(f"{name}: schema exposes a nullable type")

    if not strict_eligible(schema):
        _log.debug("%s: has optional arguments, so strict is not set even with strict_tools", name)


def strict_eligible(schema: dict[str, Any]) -> bool:
    """OpenAI strict モードの前提（全オブジェクトで全プロパティが required）を満たすか。"""
    for node in _nodes_of(schema):
        props = node.get("properties")
        if not isinstance(props, dict):
            if node.get("type") == "object" or "additionalProperties" in node:
                return False
            continue
        if node.get("additionalProperties"):
            return False
        required = node.get("required") or []
        if not isinstance(required, list) or set(required) != set(props):
            return False
    return True


def _rewrite(node: dict[str, Any]) -> dict[str, Any]:
    """1 ノードを再帰的に整形する。"""
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key == "title":
            continue
        if key in _MAP_KEYWORDS and isinstance(value, dict):
            out[key] = {k: _rewrite(v) if isinstance(v, dict) else v for k, v in value.items()}
        elif key in _LIST_KEYWORDS and isinstance(value, list):
            out[key] = [_rewrite(v) if isinstance(v, dict) else v for v in value]
        elif key in _NODE_KEYWORDS and isinstance(value, dict):
            out[key] = _rewrite(value)
        else:
            out[key] = value
    if "properties" in out:
        out.pop("description", None)
    _prune_required(out, node)
    return _fold_nullable(out)


def _prune_required(out: dict[str, Any], original: dict[str, Any]) -> None:
    """nullable だったプロパティを ``required`` から外す（省略可能性の表現を一本化）。"""
    props = original.get("properties")
    required = out.get("required")
    if not isinstance(props, dict) or not isinstance(required, list):
        return
    nullable = {k for k, v in props.items() if isinstance(v, dict) and _is_nullable(v)}
    kept = [r for r in required if r not in nullable]
    if kept:
        out["required"] = kept
    else:
        out.pop("required", None)


def _fold_nullable(node: dict[str, Any]) -> dict[str, Any]:
    """``anyOf[X, null]`` を ``X`` へ畳み、``default: null`` を落とす。"""
    for keyword in ("anyOf", "oneOf"):
        variants = node.get(keyword)
        if not isinstance(variants, list):
            continue
        non_null = [v for v in variants if not _is_null_schema(v)]
        if len(non_null) == 1 and len(non_null) < len(variants):
            merged = {k: v for k, v in node.items() if k != keyword}
            for k, v in non_null[0].items():
                merged.setdefault(k, v)
            node = merged
    node_type = node.get("type")
    if isinstance(node_type, list):
        rest = [t for t in node_type if t != "null"]
        if len(rest) == 1:
            node = {**node, "type": rest[0]}
    if "default" in node and node["default"] is None:
        node = {k: v for k, v in node.items() if k != "default"}
    return node


def _is_null_schema(value: Any) -> bool:
    return isinstance(value, dict) and value.get("type") == "null"


def _is_nullable(prop: dict[str, Any]) -> bool:
    if _null_in_type(prop.get("type")):
        return True
    for keyword in ("anyOf", "oneOf"):
        variants = prop.get(keyword)
        if isinstance(variants, list) and any(_is_null_schema(v) for v in variants):
            return True
    return False


def _null_in_type(node_type: Any) -> bool:
    return isinstance(node_type, list) and "null" in node_type


def _nodes_of(schema: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """スキーマ内の全ノードを深さ優先で辿る。"""
    yield schema
    for key, value in schema.items():
        if key in _MAP_KEYWORDS and isinstance(value, dict):
            for sub in value.values():
                if isinstance(sub, dict):
                    yield from _nodes_of(sub)
        elif key in _LIST_KEYWORDS and isinstance(value, list):
            for sub in value:
                if isinstance(sub, dict):
                    yield from _nodes_of(sub)
        elif key in _NODE_KEYWORDS and isinstance(value, dict):
            yield from _nodes_of(value)


def _properties_of(schema: dict[str, Any]) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """``(所有者, プロパティ名, スキーマ)`` を列挙する。"""
    for name, prop in (schema.get("properties") or {}).items():
        if isinstance(prop, dict):
            yield "", name, prop
    for def_name, definition in (schema.get("$defs") or {}).items():
        if not isinstance(definition, dict):
            continue
        for name, prop in (definition.get("properties") or {}).items():
            if isinstance(prop, dict):
                yield def_name, name, prop

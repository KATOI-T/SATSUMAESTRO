"""LSP の action と、番号から名前への表。"""

from __future__ import annotations

from enum import StrEnum

from satsumaestro.tools.aliases import name_words


class LspRequest(StrEnum):
    DIAGNOSTICS = "diagnostics"
    HOVER = "hover"
    DEFINITION = "definition"
    REFERENCES = "references"
    COMPLETION = "completion"
    SYMBOLS = "symbols"
    FORMAT = "format"


_FILLER_WORDS = frozenset(
    {"text", "document", "workspace", "publish", "goto", "go", "to", "find", "get", "show", "list"}
)
_FORMS: dict[str, LspRequest] = {
    **dict.fromkeys(("diagnostics", "diagnostic"), LspRequest.DIAGNOSTICS),
    **dict.fromkeys(("hover",), LspRequest.HOVER),
    **dict.fromkeys(("definition", "definitions"), LspRequest.DEFINITION),
    **dict.fromkeys(("references", "reference", "refs", "usages"), LspRequest.REFERENCES),
    **dict.fromkeys(("completion", "completions", "complete"), LspRequest.COMPLETION),
    **dict.fromkeys(("symbols", "symbol", "outline"), LspRequest.SYMBOLS),
    **dict.fromkeys(("format", "formatting", "fmt"), LspRequest.FORMAT),
}


def action_for(raw: str) -> LspRequest | None:
    """書かれた action を :class:`LspRequest` にする。"""
    words = [w for w in name_words(raw) if w not in _FILLER_WORDS]
    return _FORMS.get("".join(words))


SEVERITY_NAMES: dict[int, str] = {1: "error", 2: "warning", 3: "information", 4: "hint"}

SYMBOL_KIND_NAMES: dict[int, str] = {
    1: "file", 2: "module", 3: "namespace", 4: "package", 5: "class", 6: "method",
    7: "property", 8: "field", 9: "constructor", 10: "enum", 11: "interface",
    12: "function", 13: "variable", 14: "constant", 15: "string", 16: "number",
    17: "boolean", 18: "array", 19: "object", 20: "key", 21: "null",
    22: "enum-member", 23: "struct", 24: "event", 25: "operator", 26: "type-parameter",
}  # fmt: skip

COMPLETION_KIND_NAMES: dict[int, str] = {
    1: "text", 2: "method", 3: "function", 4: "constructor", 5: "field", 6: "variable",
    7: "class", 8: "interface", 9: "module", 10: "property", 11: "unit", 12: "value",
    13: "enum", 14: "keyword", 15: "snippet", 16: "color", 17: "file", 18: "reference",
    19: "folder", 20: "enum-member", 21: "constant", 22: "struct", 23: "event",
    24: "operator", 25: "type-parameter",
}  # fmt: skip

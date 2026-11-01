"""存在しないツール名の呼び出しを、表記ゆれの解決と候補の提示で 1 回で直せるようにする。"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field

from satsumaestro.tools.aliases import alias_target, name_key, name_words
from satsumaestro.tools.registry import ToolRegistry

NO_AUTOLOAD = frozenset({"enter_plan_mode"})
MAX_CANDIDATES = 3
UNKNOWN_TOOL_LOOP_LIMIT = 3
_CLOSE_CUTOFF = 0.75
_TRAILING = "?!.:;"

_INTENT: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("search", "find", "lookup", "grep", "query"), ("grep_search", "glob_search")),
    (
        ("list", "listdir", "ls", "readdir", "dir", "tree", "printtree"),
        ("list_directory", "glob_search"),
    ),
    (("open", "cat", "view", "show", "readfile", "getfile"), ("read_file",)),
    (("git", "run", "exec", "shell", "command", "sys", "sysinfo", "test"), ("bash",)),
    (("create", "save", "newfile", "writefile"), ("write_file",)),
    (("replace", "modify", "patch", "update"), ("edit_file",)),
    (("remove", "delete", "rm", "unlink"), ("delete_path",)),
)
_GREP_LIKE = {
    "include",
    "ignore_case",
    "context_before",
    "context_after",
    "multiline",
    "output_mode",
    "case_sensitive",
    "regex",
}
_ARG_RENAMES: dict[str, tuple[str, ...]] = {
    "query": ("pattern", "command"),
    "regex": ("pattern",),
    "max_results": ("limit",),
    "topn": ("limit",),
    "file": ("path",),
    "file_path": ("path",),
    "filename": ("path",),
    "directory": ("path",),
    "dir": ("path",),
    "text": ("content",),
    "cmd": ("command",),
}


@dataclass(frozen=True)
class Candidate:
    """候補のツール 1 件。"""

    name: str
    purpose: str
    arg_hint: str
    deferred: bool


@dataclass(frozen=True)
class Suggestion:
    candidates: list[Candidate] = field(default_factory=list)
    available: list[str] = field(default_factory=list)
    empty_name: bool = False


def _strip(name: str) -> str:
    return name.strip().rstrip(_TRAILING).strip()


def resolve_variant(name: str, registry: ToolRegistry) -> str | None:
    """同じツールの表記ゆれなら正規名を返す（大文字小文字・区切り・末尾の記号・既知の別名）。"""
    raw = _strip(name)
    if not raw:
        return None
    key = name_key(raw)
    names = [*registry.core_names(), *registry.deferred_names()]
    hits = {n for n in names if name_key(n) == key}
    alias = alias_target(raw)
    if alias is not None and registry.get(alias) is not None:
        hits.add(alias)
    return hits.pop() if len(hits) == 1 else None


def _purpose(registry: ToolRegistry, name: str) -> str:
    text = registry.description(name)
    first = text.split(". ")[0].strip().rstrip(".")
    return first if len(first) <= 100 else first[:100] + " …"


def _arg_hint(registry: ToolRegistry, name: str, keys: list[str]) -> str:
    tool = registry.get(name)
    if tool is None:
        return ""
    fields = tool.Input.model_fields
    notes: list[str] = []
    for key in keys:
        if key in fields:
            continue
        renamed = next((r for r in _ARG_RENAMES.get(key, ()) if r in fields), None)
        notes.append(
            f"use `{renamed}` instead of `{key}`" if renamed else f"`{key}` is not a parameter"
        )
    required = [k for k, f in fields.items() if f.is_required()]
    parts = [f"parameters: {', '.join(fields)}"] if fields else []
    if required:
        parts.append(f"required: {', '.join(required)}")
    parts.extend(notes)
    return "; ".join(parts)


def suggest(
    name: str, inp: dict[str, object], registry: ToolRegistry, exposed: set[str]
) -> Suggestion:
    """存在しない ``name`` に対する候補（最大 :data:`MAX_CANDIDATES` 件）と使えるツールの一覧。"""
    available = [n for n in [*registry.core_names(), *registry.deferred_names()] if n in exposed]
    raw = _strip(name)
    if not raw:
        return Suggestion(available=available, empty_name=True)
    key = name_key(raw)
    names = [*registry.core_names(), *registry.deferred_names()]
    picked: list[str] = []

    def add(candidate: str) -> None:
        if candidate in names and candidate not in picked:
            picked.append(candidate)

    words = {*name_words(raw), key}
    for verbs, tools in _INTENT:
        if words & set(verbs):
            for t in tools:
                add(t)
    if not picked:
        keyed = {name_key(n): n for n in names}
        for close in difflib.get_close_matches(
            key, list(keyed), n=MAX_CANDIDATES, cutoff=_CLOSE_CUTOFF
        ):
            add(keyed[close])
        for n in names:
            nk = name_key(n)
            if n == "tool_search" and "tool" not in key:
                continue
            if len(key) >= 3 and (key in nk or nk in key):
                add(n)
            elif len(key) >= 4 and nk.startswith(key[:4]):
                add(n)
    picked = _rank_by_args(picked, inp)
    input_keys = list(inp)
    candidates = [
        Candidate(
            name=n,
            purpose=_purpose(registry, n),
            arg_hint=_arg_hint(registry, n, input_keys),
            deferred=n not in exposed,
        )
        for n in picked[:MAX_CANDIDATES]
    ]
    return Suggestion(candidates=candidates, available=available)


def _rank_by_args(picked: list[str], inp: dict[str, object]) -> list[str]:
    """grep_search と glob_search が両方あるとき、引数の形でどちらを先にするか決める。"""
    pair = ("grep_search", "glob_search")
    if not all(n in picked for n in pair):
        return picked
    text = inp.get("pattern", inp.get("query"))
    glob_first = not (_GREP_LIKE & set(inp)) and glob_like(text)
    order = iter(("glob_search", "grep_search") if glob_first else pair)
    return [next(order) if n in pair else n for n in picked]


def glob_like(value: object) -> bool:
    """``pattern`` がファイル名の glob に見えるか。"""
    if not isinstance(value, str):
        return False
    return any(c in value for c in "*?[") and not any(c in value for c in "\\^$|()+")


_MALFORMED = re.compile(r"[\s<>]")


def malformed_name(name: str) -> bool:
    """名前の中に空白・改行・``<`` / ``>`` があるか（前後の空白は表記ゆれとして別に扱う）。"""
    return bool(_MALFORMED.search(name.strip()))


def malformed_call_message(name: str, registry: ToolRegistry, available: list[str]) -> str:
    """形の壊れた呼び出しへの返答。"""
    head = _MALFORMED.split(name.strip(), maxsplit=1)[0]
    meant = head if registry.get(head) is not None else resolve_variant(head, registry)
    lines = [
        f"malformed tool call: the tool name {json.dumps(name)} contains spaces or markup, "
        "so the call could not be read and nothing was run.",
        "Call the tool again with only its exact name as the name and its arguments as JSON.",
    ]
    if meant is not None:
        lines.append(f"If you meant {meant}, call {meant} with its arguments.")
    lines.append(f"Available tools: {', '.join(available)}.")
    return "\n".join(lines)


def echoes_its_name(name: str, inp: dict[str, object]) -> bool:
    """引数が、ツール名そのものの鍵 1 つで値が空か（``{"run_turn": ""}``）。"""
    return list(inp) == [name] and inp[name] in ("", None, {}, [])


def unknown_tool_insist_message(name: str, available: list[str], *, repeated: bool) -> str:
    """同じ存在しない名前を続けて呼んだとき（か、名前だけの空の呼び出し）の返答。"""
    head = f"unknown tool: {name} (called again)." if repeated else f"unknown tool: {name}."
    return "\n".join(
        [
            head,
            f"There is no tool named {name}, and calling it again will not change that. "
            "Do the next step directly with one of the available tools "
            "(for example read_file, write_file, edit_file or bash).",
            f"If you call {name} again in this turn, the turn will be stopped.",
            f"Available tools: {', '.join(available)}.",
        ]
    )


def unknown_tool_stop_message(name: str, calls: int) -> str:
    """同じ存在しない名前を上限の回数まで続けて呼び、ターンを止めるときの返答。"""
    return f"unknown tool: {name} ({calls} calls in a row). This turn is stopped."


def unknown_tool_message(name: str, suggestion: Suggestion, loaded: list[str]) -> str:
    """モデルへ返す文。"""
    shown = name if name.strip() else "(empty)"
    lines = [f"unknown tool: {shown}."]
    if suggestion.empty_name:
        lines.append("The tool name was empty. Call one of the available tools by its exact name.")
    elif suggestion.candidates:
        lines.append("Did you mean one of these? (not run; call it yourself with fixed arguments)")
        for c in suggestion.candidates:
            status = (
                " — loaded now; you can call it from your next turn" if c.name in loaded else ""
            )
            hint = f" ({c.arg_hint})" if c.arg_hint else ""
            lines.append(f"- {c.name}: {c.purpose}{status}{hint}")
    else:
        lines.append("No similar tool name was found.")
    lines.append(
        f"Available tools: {', '.join(suggestion.available)}. Use tool_search to find others."
    )
    return "\n".join(lines)


def autoload_message(name: str, registry: ToolRegistry) -> str:
    """遅延ツールを直接呼ばれ、読み込んだときの文（今回の呼び出しは実行していない）。"""
    return (
        f"tool {name} was not loaded yet, so this call was not run. It is loaded now; call it "
        f"again from your next turn ({_arg_hint(registry, name, [])})."
    )

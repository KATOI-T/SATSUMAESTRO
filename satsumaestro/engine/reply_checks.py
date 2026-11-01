"""ツール呼び出しの無い応答が「回答」か「やったつもり」かを見分ける純関数。"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_CANDIDATES = 3

_CALL_NAME_KEYS = ("name", "tool", "tool_name")
_CALL_ARG_KEYS = ("arguments", "parameters", "input", "args")


def strip_code_fence(s: str) -> str:
    """先頭のコードフェンスを除去する（``json`` 以外の言語タグ・前後空行にも耐える）。"""
    if not s.startswith("```"):
        return s
    body = s[3:].strip()
    if body.endswith("```"):
        body = body[:-3].strip()
    m = re.match(r"[A-Za-z][\w+.-]*\s", body)
    if m is not None and not body.startswith("{"):
        body = body[m.end() :].strip()
    return body


def parse_whole_json(text: str) -> Any:
    """本文**全体**が 1 つの JSON オブジェクトならそれを返す。"""
    s = strip_code_fence(text.strip())
    if not s.startswith("{"):
        return None
    try:
        obj, end = json.JSONDecoder().raw_decode(s)
    except ValueError:
        return None
    return obj if not s[end:].strip() else None


@dataclass(frozen=True)
class BareToolArgs:
    """本文が引数だけの JSON だったときの判定結果。"""

    keys: tuple[str, ...]
    tools: tuple[str, ...]


def detect_bare_tool_args(
    text: str,
    schemas: Iterable[Mapping[str, Any]],
    *,
    read_only: Callable[[str], bool] | None = None,
) -> BareToolArgs | None:
    """本文がツール名の無い**引数だけの JSON** なら、合うツールの候補を返す。"""
    obj = parse_whole_json(text)
    if not isinstance(obj, dict) or not obj:
        return None
    if "tool_calls" in obj or (
        any(k in obj for k in _CALL_NAME_KEYS) and any(k in obj for k in _CALL_ARG_KEYS)
    ):
        return None
    keys = set(obj)
    ranked: list[tuple[int, str]] = []
    for schema in schemas:
        name = schema.get("name")
        input_schema = schema.get("input_schema") or {}
        props = input_schema.get("properties") or {}
        required = set(input_schema.get("required") or ())
        if not isinstance(name, str) or not props:
            continue
        if keys <= set(props) and required <= keys:
            ranked.append((len(props) - len(keys), name))
    if not ranked:
        return None
    names = [name for _, name in sorted(ranked)]
    if read_only is not None:
        safe = [name for name in names if read_only(name)]
        if safe:
            names = safe
    return BareToolArgs(keys=tuple(obj), tools=tuple(names[:MAX_CANDIDATES]))


_BACKTICK_RE = re.compile(r"`([^`\n]+)`")
_LINE_SUFFIX_RE = re.compile(r":\d+(?::\d+)?$")
_EXT_RE = re.compile(r"\.[A-Za-z0-9]{1,8}$")
_COMPLETION_RE = re.compile(
    r"^(?:[ \t]*(?:[-*+>]|\d+[.)]|#+)[ \t]*)*(?:\*\*)?"
    r"(?:Implemented|Created|Added|Updated|Modified|Wrote|Rewrote|Fixed|Removed|Deleted|"
    r"Refactored|Renamed)\b"
    r"|(?:実装|作成|追加|更新|修正|変更|削除|書き込み)(?:しました|した。)",
    re.MULTILINE,
)


def mentioned_workspace_paths(text: str) -> list[str]:
    """本文がバッククォートで囲んだ、作業ディレクトリの相対パスらしい語。"""
    out: list[str] = []
    for raw in _BACKTICK_RE.findall(text):
        token = _LINE_SUFFIX_RE.sub("", raw.strip())
        if token.startswith("./"):
            token = token[2:]
        if (
            not token
            or any(ch.isspace() for ch in token)
            or "://" in token
            or token.startswith(("/", "~", "-"))
            or ".." in token.split("/")
            or any(ch in token for ch in '*?[<>|"')
        ):
            continue
        if "/" not in token and not _EXT_RE.search(token):
            continue
        if token not in out:
            out.append(token)
    return out


def claims_completion(text: str) -> bool:
    """本文が作業の完了を報告する形か（補助の判定。語の一覧は ``_COMPLETION_RE`` の 1 か所）。"""
    return _COMPLETION_RE.search(text) is not None


@dataclass(frozen=True)
class NoChangeVerdict:
    """変更系ツールの試行が無いのに作業の完了を報告した本文の判定結果。"""

    paths: tuple[str, ...]
    missing: tuple[str, ...]


def no_change_report(
    text: str,
    *,
    cwd: Path,
    written: Iterable[Path],
    since: float,
) -> NoChangeVerdict | None:
    """変更系ツールの試行が 0 回のターンで、本文が作業の完了を報告しているか。"""
    if not claims_completion(text):
        return None
    paths = mentioned_workspace_paths(text)
    if not paths:
        return None
    root = cwd.resolve()
    resolved: dict[str, Path] = {}
    for p in paths:
        try:
            full = (root / p).resolve()
        except OSError:
            continue
        if full != root and root not in full.parents:
            continue
        resolved[p] = full
    if not resolved:
        return None
    existing = [p for p, full in resolved.items() if full.exists()]
    missing = [p for p, full in resolved.items() if not full.exists()]
    if existing and not missing:
        written_set = {w.resolve() if not w.is_absolute() else w for w in written}

        def plausible(p: str) -> bool:
            full = resolved[p]
            if full in written_set:
                return True
            try:
                return full.stat().st_mtime >= since
            except OSError:
                return False

        if all(plausible(p) for p in existing):
            return None
    return NoChangeVerdict(paths=tuple(resolved), missing=tuple(missing))

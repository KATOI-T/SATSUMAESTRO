"""セッション記録へ書く前に秘密を伏せる純関数。"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from functools import lru_cache
from typing import Any

import regex

from satsumaestro.security.secret_env import PROVIDER_KEY_VARS
from satsumaestro.util.environ import ENVIRON

_log = logging.getLogger(__name__)

REDACTED = "[redacted]"

_TIMEOUT_S = 1.0
_MIN_VALUE_LEN = 8

_PLACEHOLDERS = frozenset(
    {"not", "unset", "none", "null", "missing", "required", "redacted", REDACTED}
)

_PEM = regex.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", regex.DOTALL
)
_ASSIGN = regex.compile(
    r"(?P<name>\b(?:[A-Za-z0-9]+_)*(?:API_?KEY|ACCESS_?TOKEN|AUTH_?TOKEN|REFRESH_?TOKEN|"
    r"CLIENT_?SECRET|SECRET_?KEY|PRIVATE_?KEY|PASSWORD|PASSWD)\b"
    r"|\b(?:[A-Za-z0-9]+_)+(?:TOKEN|SECRET)\b)"
    r"(?P<sep>[\"']?\s*[=:]\s*)(?P<quote>[\"']?)(?P<value>[^\s\"'`,;]{4,})",
    regex.IGNORECASE,
)
_AUTH = regex.compile(
    r"(?P<prefix>\bAuthorization\s*[:=]\s*(?:Bearer|Basic|Token)\s+)(?P<value>[^\s\"']+)",
    regex.IGNORECASE,
)
_QUERY = regex.compile(
    r"(?P<prefix>[?&](?:token|key|api_key|apikey|access_token|secret|password)=)"
    r"(?P<value>[^&\s\"'#]+)",
    regex.IGNORECASE,
)
_SHAPES = regex.compile(
    r"\b(?:sk-ant-[A-Za-z0-9_\-]{20,}|sk-[A-Za-z0-9_\-]{20,}|xai-[A-Za-z0-9_\-]{20,}"
    r"|gh[pousr]_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{22,}|AKIA[0-9A-Z]{16}"
    r"|xox[baprs]-[A-Za-z0-9\-]{10,}|AIza[0-9A-Za-z_\-]{35})"
)

_SHELL = regex.compile(r"[$`()&|;<>]")
_MIN_SHOWN_HEAD = 4

_STRUCTURAL_KEYS = frozenset(
    {"type", "role", "id", "tool_use_id", "tool_name", "name", "session_id", "model", "signature"}
)


def secret_values(env: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """ハーネスが読む鍵の、いまの環境での値（伏せる対象。長さの下限未満は除く）。"""
    source = env if env is not None else ENVIRON
    found = {
        value
        for name in PROVIDER_KEY_VARS
        if (value := source.get(name)) and len(value) >= _MIN_VALUE_LEN
    }
    return tuple(sorted(found, key=len, reverse=True))


def _sub(pattern: regex.Pattern[str], text: str, repl: Any) -> str:
    try:
        return pattern.sub(repl, text, timeout=_TIMEOUT_S)
    except TimeoutError:
        _log.warning("secret redaction timed out: %s", pattern.pattern[:40])
        return text


def _whole(value: str) -> str:
    return REDACTED


def _before_shell(value: str) -> str:
    """値の、最初のシェルの記号から後ろは実行される物なので残す（前が短ければ伏せない）。"""
    cut = _SHELL.search(value)
    if cut is None:
        return REDACTED
    head = value[: cut.start()]
    if len(head) < _MIN_SHOWN_HEAD or head.lower() in _PLACEHOLDERS:
        return value
    return REDACTED + value[cut.start() :]


def _assign_repl(m: regex.Match[str], hide: Callable[[str], str]) -> str:
    if m.group("value").lower() in _PLACEHOLDERS:
        return m.group(0)
    return f"{m.group('name')}{m.group('sep')}{m.group('quote')}{hide(m.group('value'))}"


@lru_cache(maxsize=4096)
def _redact_cached(text: str, values: tuple[str, ...], shown: bool = False) -> str:
    hide = _before_shell if shown else _whole
    out = text
    for value in values:
        out = out.replace(value, REDACTED)
    out = _sub(_PEM, out, REDACTED)
    out = _sub(_ASSIGN, out, lambda m: _assign_repl(m, hide))
    out = _sub(_AUTH, out, lambda m: f"{m.group('prefix')}{hide(m.group('value'))}")
    out = _sub(_QUERY, out, lambda m: f"{m.group('prefix')}{hide(m.group('value'))}")
    return _sub(_SHAPES, out, REDACTED)


def _value_key(values: Iterable[str]) -> tuple[str, ...]:
    key = {v for v in values if len(v) >= _MIN_VALUE_LEN}
    return tuple(sorted(key, key=len, reverse=True))


def redact_secrets(text: str, *, values: Iterable[str] = ()) -> str:
    """``text`` の秘密を ``[redacted]`` に置き換えた文字列を返す。"""
    return _redact_cached(text, _value_key(values))


def redact_shown(text: str, *, values: Iterable[str] = ()) -> str:
    """画面と確認に出す文字列を伏せる（秘密らしい名前の値でもシェルの記号から後ろは残す）。"""
    return _redact_cached(text, _value_key(values), True)


def screen_redactor(live: bool) -> Callable[[str], str] | None:
    """``redaction.live`` なら、画面に出す文字列を呼んだ時点の鍵の値で伏せる関数。"""
    if not live:
        return None
    return lambda text: redact_shown(text, values=secret_values())


def redact_shown_input(payload: Mapping[str, Any], *, values: Iterable[str] = ()) -> dict[str, Any]:
    """画面と確認に出すツールの入力を伏せた写し（キーの名前では飛ばさない）。"""
    vals = tuple(values)

    def walk(node: Any) -> Any:
        if isinstance(node, str):
            return redact_shown(node, values=vals)
        if isinstance(node, Mapping):
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return {k: walk(v) for k, v in payload.items()}


def redact_rows(rows: list[dict[str, Any]], *, values: Iterable[str] = ()) -> list[dict[str, Any]]:
    """JSONL の行（``model_dump`` 済みの dict）の文字列値をすべて伏せた写しを返す。"""
    vals = tuple(values)

    def walk(node: Any, key: str | None) -> Any:
        if isinstance(node, str):
            if key in _STRUCTURAL_KEYS:
                return node
            return redact_secrets(node, values=vals)
        if isinstance(node, dict):
            return {k: walk(v, k) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v, None) for v in node]
        return node

    return [walk(row, None) for row in rows]

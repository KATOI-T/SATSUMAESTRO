"""セッション記録へ書く前に秘密を伏せる純関数。"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from functools import lru_cache
from typing import Any

import regex

from satsumaestro.security.secret_env import PROVIDER_KEY_VARS
from satsumaestro.util.environ import ENVIRON

_log = logging.getLogger(__name__)

REDACTED = "[redacted]"

_TIMEOUT_S = 1.0
_MIN_VALUE_LEN = 8
_PIECE_MIN = 4

_PLACEHOLDERS = frozenset(
    {"not", "unset", "none", "null", "missing", "required", "redacted", REDACTED}
)

_PEM = regex.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", regex.DOTALL
)
_ASSIGN_HEAD = (
    r"(?P<name>\b(?:[A-Za-z0-9]+_)*(?:API_?KEY|ACCESS_?TOKEN|AUTH_?TOKEN|REFRESH_?TOKEN|"
    r"CLIENT_?SECRET|SECRET_?KEY|PRIVATE_?KEY|PASSWORD|PASSWD)\b"
    r"|\b(?:[A-Za-z0-9]+_)+(?:TOKEN|SECRET)\b)"
    r"(?P<sep>[\"']?\s*[=:]\s*)(?P<quote>[\"']?)"
)
_AUTH_HEAD = r"(?P<prefix>\bAuthorization\s*[:=]\s*(?:Bearer|Basic|Token)\s+)"
_QUERY_HEAD = r"(?P<prefix>[?&](?:token|key|api_key|apikey|access_token|secret|password)=)"
_ASSIGN = regex.compile(_ASSIGN_HEAD + r"(?P<value>[^\s\"'`,;]{4,})", regex.IGNORECASE)
_AUTH = regex.compile(_AUTH_HEAD + r"(?P<value>[^\s\"']+)", regex.IGNORECASE)
_QUERY = regex.compile(_QUERY_HEAD + r"(?P<value>[^&\s\"'#]+)", regex.IGNORECASE)
_SHAPES = regex.compile(
    r"\b(?:sk-ant-[A-Za-z0-9_\-]{20,}|sk-[A-Za-z0-9_\-]{20,}|xai-[A-Za-z0-9_\-]{20,}"
    r"|gh[pousr]_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{22,}|AKIA[0-9A-Z]{16}"
    r"|xox[baprs]-[A-Za-z0-9\-]{10,}|AIza[0-9A-Za-z_\-]{35})"
)

_ROW = regex.compile("[^\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029]+")
_ALNUM = regex.compile("[A-Za-z0-9]+")
_VARIABLE = regex.compile(r"\$(?:[A-Za-z_][A-Za-z0-9_]*|[0-9@*#?$!-])")
_RUNS_BARE = frozenset("&|;<>()")
_COMMAND_VALUES = (
    (regex.compile(_ASSIGN_HEAD + r"(?=[^\s\"',;])", regex.IGNORECASE), "\"',;`", 4, True),
    (regex.compile(_AUTH_HEAD + r"(?=[^\s\"'])", regex.IGNORECASE), "\"'", 1, False),
    (regex.compile(_QUERY_HEAD + r"(?=[^&\s\"'#])", regex.IGNORECASE), "&\"'#", 1, False),
)

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


def _read_quotes(
    text: str, start: int, end: int, quote: str = "", escaped: bool = False
) -> tuple[str, bool]:
    """``text[start:end]`` を読んだ後の、開いている引用符と ``\\`` の続き。"""
    for ch in text[start:end]:
        if escaped:
            escaped = False
        elif quote == "'":
            if ch == "'":
                quote = ""
        elif ch == "\\":
            escaped = True
        elif quote and ch == quote:
            quote = ""
        elif not quote and ch in "'\"":
            quote = ch
    return quote, escaped


def _closing(text: str, i: int, closer: str, opener: str = "") -> int | None:
    """開き記号の直後 ``i`` から、対応する閉じ記号の直後の位置（閉じていなければ ``None``）。"""
    depth, quote = 1, ""
    while i < len(text):
        ch = text[i]
        if quote == "'":
            if ch == "'":
                quote = ""
        elif ch == "\\":
            i += 1
        elif quote:
            if ch == quote:
                quote = ""
        elif ch in "'\"":
            quote = ch
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return i + 1
        elif ch == opener:
            depth += 1
        i += 1
    return None


def _expansion(text: str, i: int) -> tuple[int, int | None] | None:
    """``text[i]`` からの展開の、開き記号の終わりと全体の終わり（閉じなければ ``None``）。"""
    for opener, closer, nest in (("$(", ")", "("), ("${", "}", "{"), ("`", "`", "")):
        if text.startswith(opener, i):
            return i + len(opener), _closing(text, i + len(opener), closer, nest)
    m = _VARIABLE.match(text, i)
    return (m.end(), m.end()) if m else None


def _shown_value(text: str, start: int, stops: str, quote: str) -> tuple[str, int]:
    """コマンドの値を、展開は残して文字だけを伏せた形と、値の終わりの位置で返す。"""
    parts: list[str] = []
    i = head = start
    while i < len(text):
        ch = text[i]
        found = None if quote == "'" else _expansion(text, i)
        if found is not None:
            opened, end = found
            if i > head:
                parts.append(REDACTED)
            if end is None:
                rest = text[opened:].rstrip("\"'")
                parts += [text[i:opened], REDACTED if rest else "", text[opened + len(rest) :]]
                return "".join(parts), len(text)
            parts.append(text[i:end])
            i = head = end
        elif ch.isspace() or ch in stops or (not quote and ch in _RUNS_BARE):
            break
        else:
            i += 2 if ch == "\\" and quote != "'" else 1
    i = min(i, len(text))
    if i > head:
        parts.append(REDACTED)
    return "".join(parts), i


def _hide_values(
    pattern: regex.Pattern[str], text: str, stops: str, shortest: int, placeholders: bool
) -> str:
    """コマンドの中の値を伏せる（展開と実行される記号は残す）。"""
    try:
        matches = list(pattern.finditer(text, timeout=_TIMEOUT_S))
    except TimeoutError:
        _log.warning("secret redaction timed out: %s", pattern.pattern[:40])
        return text
    parts: list[str] = []
    last = read = 0
    state = ("", False)
    for m in matches:
        start = m.end()
        if m.start() < last:
            continue
        state = _read_quotes(text, read, start, *state)
        read = start
        shown, end = _shown_value(text, start, stops, state[0])
        raw = text[start:end]
        if len(raw) < shortest or (placeholders and raw.lower() in _PLACEHOLDERS):
            continue
        parts += [text[last:start], shown]
        last = end
    return "".join(parts) + text[last:]


def _assign_repl(m: regex.Match[str]) -> str:
    if m.group("value").lower() in _PLACEHOLDERS:
        return m.group(0)
    return f"{m.group('name')}{m.group('sep')}{m.group('quote')}{REDACTED}"


@lru_cache(maxsize=4096)
def _redact_cached(text: str, values: tuple[str, ...], command: bool = False) -> str:
    out = text
    for value in values:
        out = out.replace(value, REDACTED)
    out = _sub(_PEM, out, REDACTED)
    if command:
        for pattern, stops, shortest, placeholders in _COMMAND_VALUES:
            out = _hide_values(pattern, out, stops, shortest, placeholders)
    else:
        out = _sub(_ASSIGN, out, _assign_repl)
        out = _sub(_AUTH, out, lambda m: f"{m.group('prefix')}{REDACTED}")
        out = _sub(_QUERY, out, lambda m: f"{m.group('prefix')}{REDACTED}")
    return _sub(_SHAPES, out, REDACTED)


def _value_key(values: Iterable[str]) -> tuple[str, ...]:
    key = {v for v in values if len(v) >= _MIN_VALUE_LEN}
    return tuple(sorted(key, key=len, reverse=True))


def redact_secrets(text: str, *, values: Iterable[str] = ()) -> str:
    """``text`` の秘密を ``[redacted]`` に置き換えた文字列を返す。"""
    return _redact_cached(text, _value_key(values))


def redact_command(command: str, *, values: Iterable[str] = ()) -> str:
    """画面と確認に出す bash のコマンドを伏せる（展開と実行される記号は残す）。"""
    return _redact_cached(command, _value_key(values), True)


def redact_lines(text: str, *, values: Iterable[str] = ()) -> list[str]:
    """伏せた文字列を、元と同じ数の行で返す（複数行にわたる秘密は各行を伏せる）。"""
    masked = text
    for value in _value_key(values):
        masked = masked.replace(value, _ROW.sub(REDACTED, value))
    masked = _sub(_PEM, masked, lambda m: _ROW.sub(REDACTED, m.group(0)))
    rows = masked.splitlines()
    lines = redact_secrets(masked).splitlines()
    return lines if len(lines) == len(rows) else [redact_secrets(row) for row in rows]


def redact_shown_input(
    tool: str, payload: Mapping[str, Any], *, values: Iterable[str] = ()
) -> dict[str, Any]:
    """画面と確認に出すツールの入力を伏せた写し（キーの名前では飛ばさない）。"""
    vals = tuple(values)

    def walk(node: Any) -> Any:
        if isinstance(node, str):
            return redact_secrets(node, values=vals)
        if isinstance(node, Mapping):
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    shown = {k: walk(v) for k, v in payload.items()}
    command = payload.get("command")
    if tool == "bash" and isinstance(command, str):
        shown["command"] = redact_command(command, values=vals)
    return shown


def _hidden_pieces(real: str, shown: str) -> list[str]:
    """``shown`` が ``[redacted]`` で隠した ``real`` の部分（各行と、長い英数字の並びも）。"""
    parts = shown.split(REDACTED)
    head, tail = parts[0], parts[-1]
    pos, stop = len(head), len(real) - len(tail)
    if len(parts) == 1 or stop < pos or not (real.startswith(head) and real.endswith(tail)):
        return []
    pieces: list[str] = []
    for part in parts[1:-1]:
        at = real.find(part, pos + 1, stop)
        if at < 0:
            break
        pieces.append(real[pos:at])
        pos = at + len(part)
    pieces.append(real[pos:stop])
    lines = [p for piece in pieces for p in (piece, *piece.splitlines()) if len(p) >= _PIECE_MIN]
    words = [w for piece in pieces for w in _ALNUM.findall(piece) if len(w) >= _MIN_VALUE_LEN]
    return lines + words


def shown_input_pieces(
    tool: str, payload: Mapping[str, Any], *, values: Iterable[str] = ()
) -> list[str]:
    """画面と確認でツールの入力から伏せた部分（同じ値を説明の文からも伏せるため）。"""
    pieces: list[str] = []

    def walk(real: Any, shown: Any) -> None:
        if isinstance(real, str) and isinstance(shown, str):
            pieces.extend(_hidden_pieces(real, shown))
        elif isinstance(real, Mapping) and isinstance(shown, Mapping):
            for key, value in real.items():
                walk(value, shown.get(key))
        elif isinstance(real, list) and isinstance(shown, list):
            for value, seen in zip(real, shown, strict=False):
                walk(value, seen)

    walk(payload, redact_shown_input(tool, payload, values=values))
    return pieces


def redact_pieces(text: str, pieces: Iterable[str], *, values: Iterable[str] = ()) -> str:
    """``text`` から ``pieces``（英数字の語の途中は除く）と秘密を伏せる。"""
    for piece in sorted(set(pieces), key=len, reverse=True):
        before = "(?<![A-Za-z0-9])" if _ALNUM.match(piece[0]) else ""
        after = "(?![A-Za-z0-9])" if _ALNUM.match(piece[-1]) else ""
        text = _sub(regex.compile(before + regex.escape(piece) + after), text, REDACTED)
    return redact_secrets(text, values=values)


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

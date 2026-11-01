"""未完の JSON 断片から文字列値を取り出す。"""

from __future__ import annotations

import json


def _scan_string_body(buf: str, start: int) -> str:
    """``buf[start:]`` を JSON 文字列本体として走査し、**完全なエスケープ単位まで**返す。"""
    out: list[str] = []
    i = start
    n = len(buf)
    while i < n:
        c = buf[i]
        if c == '"':
            break
        if c != "\\":
            out.append(c)
            i += 1
            continue
        if i + 1 >= n:
            break
        if buf[i + 1] == "u":
            if i + 6 > n:
                break
            out.append(buf[i : i + 6])
            i += 6
        else:
            out.append(buf[i : i + 2])
            i += 2
    return "".join(out)


def _find_value_start(buf: str, key: str) -> int | None:
    """``"key"`` の値（文字列）の開始位置（開き引用符の次）を返す。"""
    needle = json.dumps(key)
    n = len(buf)
    at = buf.find(needle)
    while at >= 0:
        i = at + len(needle)
        while i < n and buf[i].isspace():
            i += 1
        if i < n and buf[i] == ":":
            i += 1
            while i < n and buf[i].isspace():
                i += 1
            if i < n and buf[i] == '"':
                return i + 1
            if i < n:
                return None
        at = buf.find(needle, at + 1)
    return None


def partial_json_string(buf: str, key: str) -> str | None:
    """未完の JSON 断片 ``buf`` から ``key`` の文字列値を**途中まで**取り出す。"""
    start = _find_value_start(buf, key)
    if start is None:
        return None
    fragment = _scan_string_body(buf, start)
    try:
        value = json.loads(f'"{fragment}"')
    except ValueError:
        return None
    return value if isinstance(value, str) else None

"""画面の文（:func:`ui`）。"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from typing import Any

from satsumaestro.i18n.schema import FALLBACK, catalog
from satsumaestro.i18n.state import LanguageState, locale_language

_MESSAGES = Path(__file__).with_name("messages")
PLURAL_FORMS = frozenset({"one", "other"})

Message = str | Mapping[str, str]

_owner: LanguageState | None = None
_screen: str | None = None
_pinned: str | None = None


def use_language(owner: LanguageState | None) -> None:
    """会話の言語の持ち主を、画面の言語の出所にする（会話を組み立てた所で 1 回）。"""
    global _owner
    _owner = owner


def use_screen_language(code: str | None) -> None:
    """持ち主の無い所の画面の言語を ``code`` にする（設定を読んだ所で。``None`` は環境の言語）。"""
    global _screen
    _screen = code


def ui_language() -> str:
    """いまの画面の言語のコード。"""
    if _pinned is not None:
        return _pinned
    if _owner is not None:
        return _owner.ui_language
    if _screen is not None:
        return _screen
    return locale_language(os.environ)


@cache
def messages(code: str) -> dict[str, Message]:
    """言語ごとの画面の文をキーで引ける形にする（``messages/<code>.toml``。無ければ空）。"""
    path = _MESSAGES / f"{code}.toml"
    if not path.is_file():
        return {}
    flat: dict[str, Message] = {}
    _flatten(tomllib.loads(path.read_text(encoding="utf-8")), "", flat)
    return flat


def _flatten(node: Mapping[str, Any], prefix: str, out: dict[str, Message]) -> None:
    for key, value in node.items():
        name = prefix + key
        if isinstance(value, dict) and not is_plural(value):
            _flatten(value, name + ".", out)
        else:
            out[name] = value


def is_plural(value: object) -> bool:
    """数で変わる文の表（``{one = …, other = …}``）か。"""
    return (
        isinstance(value, dict)
        and "other" in value
        and set(value) <= PLURAL_FORMS
        and all(isinstance(v, str) for v in value.values())
    )


def ui(key: str, /, **params: object) -> str:
    """いまの画面の言語の文（その言語に文が無ければ英語）を ``params`` で埋めて返す。"""
    message = messages(ui_language()).get(key)
    if message is None:
        message = messages(FALLBACK).get(key)
    if message is None:
        raise KeyError(f"no screen text for {key!r}")
    if not isinstance(message, str):
        message = message.get("one", message["other"]) if params.get("n") == 1 else message["other"]
    return message.format(**params)


def unknown_language(value: str) -> str:
    """定義に無い言語のコードを受けたときの文（設定の ``language.fixed`` と ``--lang``）。"""
    codes = ", ".join(catalog().codes())
    return ui("i18n.text.unknown_language_available", value=value, codes=codes)


def screen_languages() -> list[str]:
    """画面の文を持つ言語のコード（定義の順）。"""
    return [spec.code for spec in catalog().languages if spec.ui]


def unknown_screen_language(value: str) -> str:
    """画面の文を持たない言語を設定の ``language.ui`` に書いたときの文。"""
    codes = ", ".join(["auto", *screen_languages()])
    return ui("i18n.text.unknown_screen_language_available", value=value, codes=codes)


@contextmanager
def pinned_ui_language(code: str | None) -> Iterator[None]:
    """この中では画面の言語を ``code`` に固定する（``None`` は固定を外す）。"""
    global _pinned
    previous = _pinned
    _pinned = code
    try:
        yield
    finally:
        _pinned = previous

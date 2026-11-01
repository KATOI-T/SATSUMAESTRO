"""会話の言語の持ち主（:class:`LanguageState`）。"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from satsumaestro.i18n.schema import FALLBACK, catalog
from satsumaestro.util.errors import ConfigError
from satsumaestro.util.lang import resolve_turn_language

if TYPE_CHECKING:
    from satsumaestro.config.models import LanguageConfig
    from satsumaestro.session.models import Session

_log = logging.getLogger(__name__)

_LOCALE_VARS = ("LC_ALL", "LC_MESSAGES", "LANG")


def locale_language(environ: Mapping[str, str]) -> str:
    """環境の言語に当たる、画面の文を持つ言語のコード（当たらなければ英語）。"""
    value = next((environ[k] for k in _LOCALE_VARS if environ.get(k)), "")
    head = value.split(".", 1)[0].split("@", 1)[0].replace("-", "_").split("_", 1)[0].lower()
    for spec in catalog().languages:
        if spec.ui and head in (loc.lower() for loc in spec.locales):
            return spec.code
    return FALLBACK


class LanguageState:
    """1 つの会話（と、その子）の言語を持つ唯一の場所。"""

    def __init__(
        self,
        settings: LanguageConfig,
        *,
        startup: str = FALLBACK,
        parent: LanguageState | None = None,
    ) -> None:
        self._settings = settings.model_copy()
        self._seen = settings.model_copy()
        self._current = ""
        self._startup = startup
        self._parent = parent
        self._session: Session | None = None
        self._source: Callable[[], LanguageConfig] | None = None

    @classmethod
    def from_config(
        cls, settings: LanguageConfig, environ: Mapping[str, str] | None = None
    ) -> LanguageState:
        """設定と環境の言語から作る（会話の頭で 1 回）。"""
        return cls(settings, startup=locale_language(os.environ if environ is None else environ))

    @property
    def mode(self) -> str:
        """``auto`` / ``fixed`` / ``off``。"""
        if self._parent is None:
            return self._settings.mode
        if self._parent.mode == "off":
            return "off"
        return "fixed" if self._parent.current else "auto"

    @property
    def fixed(self) -> str:
        """``mode=fixed`` のときの言語。"""
        if self._parent is not None:
            return self._parent.current or self._parent.fixed
        return self._settings.fixed

    @property
    def pinned(self) -> bool:
        """利用者の言葉に依らず決まった言語で答えるか（system prompt の言語の節が使う）。"""
        return self.mode == "fixed"

    @property
    def repair(self) -> bool:
        """答えの言語が違うとき、1 回だけ書き直させるか。"""
        return self._parent.repair if self._parent is not None else self._settings.repair

    @property
    def min_chars(self) -> int:
        """ラテン文字の判定に要る文字数。"""
        return self._parent.min_chars if self._parent is not None else self._settings.min_chars

    @property
    def current(self) -> str:
        """いまの言語（``""`` = 未決定、または言語を制御しない）。"""
        if self.mode == "off":
            return ""
        if self._parent is not None and self._parent.current:
            return self._parent.current
        return self._current

    @property
    def ui_setting(self) -> str | None:
        """設定の ``language.ui``（言語のコードか ``auto``。未設定は None）。"""
        if self._parent is not None:
            return self._parent.ui_setting
        return self._settings.ui

    @property
    def chosen_ui(self) -> str | None:
        """設定の ``language.ui`` で選んだ画面の言語（``auto`` と未設定は None）。"""
        if self._parent is not None:
            return self._parent.chosen_ui
        return self._settings.chosen_ui

    @property
    def ui_language(self) -> str:
        """画面に使う言語。"""
        chosen = self.chosen_ui
        if chosen is not None:
            return chosen
        code = self.current
        if not code and self._parent is not None:
            return self._parent.ui_language
        if not code:
            code = self.fixed if self.mode == "fixed" else self._startup
        spec = catalog().get(code)
        return code if spec is not None and spec.ui else FALLBACK

    def begin_turn(self, user_text: str) -> str:
        """ターンの頭で言語を決めて返す（設定ファイルの書き換えを読み、発話から判定する）。"""
        if self._parent is None:
            self._reload()
            self._current = resolve_turn_language(
                self.mode, self.fixed, self._current, user_text, min_chars=self.min_chars
            )
        elif self.mode == "auto":
            self._current = resolve_turn_language(
                "auto", self.fixed, self._current, user_text, min_chars=self.min_chars
            )
        self._write_session()
        return self.current

    def set_fixed(self, code: str) -> None:
        """この言語に固定する（``/lang``・``--lang``・評価）。"""
        if catalog().get(code) is None:
            raise ValueError(f"unknown language: {code!r}")
        self._settings = self._settings.model_copy(update={"mode": "fixed", "fixed": code})
        self._current = code
        self._write_session()

    def set_auto(self) -> None:
        """発話からの判定に戻す（次の発話から追従する）。"""
        self._settings = self._settings.model_copy(update={"mode": "auto"})

    def set_ui(self, value: str) -> str | None:
        """画面の言語を ``value``（言語のコードか ``auto``）にする（``/language``）。"""
        self._settings = self._settings.model_copy(update={"ui": value})
        if self._source is None:
            return None
        try:
            stored = self._source().ui
        except ConfigError as exc:
            _log.warning("language settings were not read after /language: %s", exc)
            return None
        self._seen = self._seen.model_copy(update={"ui": stored})
        return stored if stored is not None and stored != value else None

    def watch(self, source: Callable[[], LanguageConfig]) -> None:
        """ターンの頭で、設定ファイルの ``language`` の項目を ``source`` から読み直す。"""
        self._source = source

    def bind(self, session: Session) -> None:
        """再開のための保存先。"""
        self._session = session
        if self._parent is None and not self._current and catalog().get(session.language):
            self._current = session.language
        self._write_session()

    def for_child(self) -> LanguageState:
        """子の会話の持ち主（親を読むだけで、値の写しを持たない）。"""
        return LanguageState(self._settings, startup=self._startup, parent=self)

    def _reload(self) -> None:
        if self._source is None:
            return
        try:
            fresh = self._source()
        except ConfigError as exc:
            _log.warning("language settings were not reloaded: %s", exc)
            return
        if fresh == self._seen:
            return
        self._seen = fresh.model_copy()
        self._settings = fresh.model_copy()

    def _write_session(self) -> None:
        if self._session is not None:
            self._session.language = self.current

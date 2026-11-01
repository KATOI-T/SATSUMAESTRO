"""言語の定義（``languages.toml``）の型と読み込み。"""

from __future__ import annotations

import re
import tomllib
from functools import cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from satsumaestro.util.errors import ConfigError

FALLBACK = "en"

DEFAULT_PATH = Path(__file__).with_name("languages.toml")

CodeRange = tuple[int, int]


def _hex(r: CodeRange) -> str:
    return f"U+{r[0]:04X}-U+{r[1]:04X}"


def _check_ranges(ranges: tuple[CodeRange, ...]) -> None:
    for lo, hi in ranges:
        if not 0 <= lo <= hi <= 0x10FFFF:
            raise ValueError(f"invalid character range: {_hex((lo, hi))}")


def _check_patterns(patterns: tuple[str, ...]) -> None:
    for pattern in patterns:
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"invalid regular expression: {pattern!r}: {exc}") from exc


def _covers(ranges: tuple[CodeRange, ...], o: int) -> bool:
    return any(lo <= o <= hi for lo, hi in ranges)


class DetectSpec(BaseModel):
    """発話から言語を判定する方法（種類の説明は ``languages.toml`` の頭）。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["script", "latin"]
    ranges: tuple[CodeRange, ...] = ()
    min_count: int = Field(default=1, ge=1)
    min_ratio: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _shape(self) -> DetectSpec:
        if self.kind == "script" and not self.ranges:
            raise ValueError("script detection needs ranges")
        if self.kind == "latin" and self.ranges:
            raise ValueError(
                "latin detection takes no ranges (it counts the letters in U+0000-U+024F)"
            )
        _check_ranges(self.ranges)
        return self

    def covers(self, o: int) -> bool:
        """文字コード ``o`` がこの言語の文字か（``script`` の判定だけが使う）。"""
        return _covers(self.ranges, o)


class LanguageSpec(BaseModel):
    """1 つの言語の定義。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(pattern=r"^[a-z]{2,3}$")
    name: str = Field(min_length=1)
    native_name: str = Field(min_length=1)
    locales: tuple[str, ...] = ()
    ui: bool = False
    detect: DetectSpec | None = None
    request_patterns: tuple[str, ...] = ()

    @field_validator("request_patterns")
    @classmethod
    def _compiles(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        _check_patterns(value)
        return value


class LanguageCatalog(BaseModel):
    """定義の全体。"""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    ignore_ranges: tuple[CodeRange, ...] = ()
    unsupported_ranges: tuple[CodeRange, ...] = ()
    unsupported_ratio: float = Field(default=0.2, gt=0.0, le=1.0)
    request_patterns: tuple[str, ...] = ()
    languages: tuple[LanguageSpec, ...] = Field(alias="language", min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> LanguageCatalog:
        codes = [spec.code for spec in self.languages]
        duplicated = sorted({code for code in codes if codes.count(code) > 1})
        if duplicated:
            raise ValueError(f"duplicate language codes: {', '.join(duplicated)}")
        fallback = self.get(FALLBACK)
        if fallback is None or not fallback.ui:
            raise ValueError(
                f"a definition of {FALLBACK} with ui = true is required (the default screen text)"
            )
        latin = [s.code for s in self.languages if s.detect and s.detect.kind == "latin"]
        if len(latin) > 1:
            raise ValueError(f"only one language can use latin detection: {', '.join(latin)}")
        _check_ranges(self.ignore_ranges)
        _check_ranges(self.unsupported_ranges)
        _check_patterns(self.request_patterns)
        owned = [
            (spec.code, r)
            for spec in self.languages
            if spec.detect is not None
            for r in spec.detect.ranges
        ]
        owned += [("ignore_ranges", r) for r in self.ignore_ranges]
        owned += [("unsupported_ranges", r) for r in self.unsupported_ranges]
        for i, (a, ra) in enumerate(owned):
            for b, rb in owned[i + 1 :]:
                if a != b and ra[0] <= rb[1] and rb[0] <= ra[1]:
                    raise ValueError(
                        f"detection ranges overlap: {_hex(ra)} of {a} and {_hex(rb)} of {b}"
                    )
        return self

    def get(self, code: str) -> LanguageSpec | None:
        """コードの定義（無ければ None）。"""
        return next((spec for spec in self.languages if spec.code == code), None)

    def codes(self) -> tuple[str, ...]:
        """定義にある言語のコード（定義の順）。"""
        return tuple(spec.code for spec in self.languages)

    def name(self, code: str) -> str:
        """モデルへの指示に書く言語名（定義に無いコードはそのまま返す）。"""
        spec = self.get(code)
        return spec.name if spec is not None else code

    def script_languages(self) -> tuple[LanguageSpec, ...]:
        """文字の範囲で判定する言語（定義の順）。"""
        return tuple(
            s for s in self.languages if s.detect is not None and s.detect.kind == "script"
        )

    def latin_language(self) -> LanguageSpec | None:
        """ラテン文字の数で判定する言語（多くて 1 つ）。"""
        return next(
            (s for s in self.languages if s.detect is not None and s.detect.kind == "latin"),
            None,
        )

    def ignored(self, o: int) -> bool:
        """判定に数えない文字か。"""
        return _covers(self.ignore_ranges, o)

    def unsupported(self, o: int) -> bool:
        """どの定義にも無い言語の文字か。"""
        return _covers(self.unsupported_ranges, o)

    def request_pattern(self) -> re.Pattern[str]:
        """答えの言語を指定した依頼に当たる正規表現（全言語の言い方をまとめたもの）。"""
        patterns = [*self.request_patterns]
        for spec in self.languages:
            patterns.extend(spec.request_patterns)
        return re.compile("|".join(f"(?:{p})" for p in patterns) or r"(?!)", re.IGNORECASE)


def load_catalog(path: Path) -> LanguageCatalog:
    """定義のファイルを読んで検証する。"""
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"cannot read the language definitions: {path}: {exc}") from exc
    try:
        return LanguageCatalog.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"invalid language definitions: {path}: {exc}") from exc


@cache
def catalog() -> LanguageCatalog:
    """同梱の定義（プロセスで 1 回だけ読む）。"""
    return load_catalog(DEFAULT_PATH)

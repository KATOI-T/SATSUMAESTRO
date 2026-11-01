"""応答言語の判定と言語指示の生成。"""

from __future__ import annotations

import re
from collections import Counter
from functools import cache

from satsumaestro.i18n.schema import DetectSpec, LanguageCatalog, catalog

UNDETERMINED = "und"

LANGUAGE_DIRECTIVE = (
    "\n\n(Reply in {name}. Code, paths, identifiers and command output stay as-is.)"
)

_STRIP_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL),
    re.compile(r"```.*?(?:```|\Z)", re.DOTALL),
    re.compile(r"~~~.*?(?:~~~|\Z)", re.DOTALL),
    re.compile(r"`[^`\n]*`"),
    re.compile(r"^[ \t]*>.*$", re.MULTILINE),
    re.compile(r"\b\w+://\S+"),
    re.compile(r"\]\([^)\s]*\)"),
    re.compile(r"[A-Za-z0-9_.\-]*[/\\][A-Za-z0-9_.\-/\\]*"),
    re.compile(r"[A-Za-z0-9]*[_:][A-Za-z0-9_:]+"),
    re.compile(r"[A-Za-z_][A-Za-z0-9_]*\(\)"),
    re.compile(r"[A-Za-z0-9_]+\.[A-Za-z0-9]{1,6}\b"),
)


@cache
def _requests_other_language() -> re.Pattern[str]:
    return catalog().request_pattern()


def language_name(code: str) -> str:
    """言語コードをプロンプトへ書く言語名にする（未知コードはそのまま返す）。"""
    return catalog().name(code)


def strip_non_prose(text: str) -> str:
    """言語判定の対象から非散文（コード・識別子・パス・URL・引用）を取り除く。"""
    for pattern in _STRIP_PATTERNS:
        text = pattern.sub(" ", text)
    return text


_LATIN = "<latin>"
_UNSUPPORTED = "<unsupported>"
_OTHER = "<other>"


def _char_kind(
    ch: str, langs: LanguageCatalog, scripts: list[tuple[str, DetectSpec]], latin: bool
) -> str:
    """1 文字の種類（``""`` = 母数に数えない）。"""
    if ch.isspace() or ch.isdigit():
        return ""
    o = ord(ch)
    if langs.ignored(o):
        return ""
    for code, detect in scripts:
        if detect.covers(o):
            return code
    if latin and ch.isalpha() and o < 0x0250:
        return _LATIN
    if langs.unsupported(o):
        return _UNSUPPORTED
    return _OTHER if ch.isalpha() else ""


def detect_language(
    text: str, *, min_chars: int = 8, languages: LanguageCatalog | None = None
) -> str:
    """散文の言語を、定義にある言語のコードか ``und`` で返す（スクリプト比率のみ）。"""
    langs = catalog() if languages is None else languages
    scripts = [(s.code, s.detect) for s in langs.script_languages() if s.detect is not None]
    latin_spec = langs.latin_language()
    counts: Counter[str] = Counter()
    for ch, n in Counter(text).items():
        kind = _char_kind(ch, langs, scripts, latin_spec is not None)
        if kind:
            counts[kind] += n
    total = sum(counts.values())

    if total == 0:
        return UNDETERMINED
    if counts[_UNSUPPORTED] / total >= langs.unsupported_ratio:
        return UNDETERMINED
    for code, detect in scripts:
        if counts[code] >= detect.min_count and counts[code] / total >= detect.min_ratio:
            return code
    if latin_spec is not None and counts[_LATIN] >= min_chars:
        return latin_spec.code
    return UNDETERMINED


def resolve_language(previous: str, detected: str) -> str:
    """判定結果を前回値と突き合わせる（判定不能なら前回値を維持する）。"""
    if detected in (UNDETERMINED, ""):
        return previous
    return detected


def resolve_turn_language(
    mode: str, fixed: str, previous: str, user_text: str, *, min_chars: int = 8
) -> str:
    """このターンで応答に使う言語を決める（``""`` = 制御しない）。"""
    if mode == "off":
        return ""
    if mode == "fixed":
        return fixed
    detected = detect_language(strip_non_prose(user_text), min_chars=min_chars)
    return resolve_language(previous, detected)


def requests_other_language(text: str) -> bool:
    """ユーザーが応答言語を明示的に指定しているか（修復の誤発火を防ぐガード）。"""
    return bool(_requests_other_language().search(text))


def with_language_directive(text: str, language: str) -> str:
    """モデルへ渡す文の末尾へ言語指示を足す（``language`` が空なら素通し）。"""
    if not language:
        return text
    return text + LANGUAGE_DIRECTIVE.format(name=language_name(language))

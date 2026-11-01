"""起動時に、画面の言語が設定に無ければ選んでもらう。"""

from __future__ import annotations

from rich.text import Text

from satsumaestro.i18n.schema import catalog
from satsumaestro.i18n.text import pinned_ui_language, screen_languages, ui

AUTO = "auto"


HEADING_STYLE = "bold cyan"
NUMBER_STYLE = "cyan"
CURRENT_STYLE = "green"


def native_name(code: str) -> str:
    """言語の自分の名前（``日本語``・``English``。定義に無いコードと ``auto`` はそのまま）。"""
    spec = catalog().get(code)
    return spec.native_name if spec is not None else code


def question_lines() -> list[Text]:
    """問いの前に出す見出しと選択肢（見出しは太字の cyan、選択肢は番号だけ cyan）。"""
    heads = []
    for code in screen_languages():
        with pinned_ui_language(code):
            heads.append(ui("cli.language_gate.choose_the_screen_language"))
    return [Text(" / ".join(heads), style=HEADING_STYLE), *choice_lines()]


def choice_lines(current: str | None = None) -> list[Text]:
    """選択肢の行（番号だけ cyan）。"""
    lines = []
    for number, code in enumerate(screen_languages(), 1):
        line = Text("  ").append(f"{number})", style=NUMBER_STYLE).append(f" {native_name(code)}")
        if code == current:
            line.append(" ✓", style=CURRENT_STYLE)
        lines.append(line)
    return lines


def question(default: str) -> str:
    """入力の手前に出す 1 行（選べる番号と、Enter で選ぶ番号）。"""
    codes = screen_languages()
    numbers = "/".join(str(n) for n in range(1, len(codes) + 1))
    mark = codes.index(default) + 1 if default in codes else 1
    return f"({numbers}) [{mark}]: "


def parse_answer(answer: str, default: str) -> str | None:
    """答えに当たる言語のコードか ``auto``（当たらなければ None で、もう一度聞く）。"""
    text = answer.strip()
    codes = screen_languages()
    if not text:
        return default if default in codes else codes[0]
    if text.lower() == AUTO:
        return AUTO
    if text.isdigit():
        number = int(text)
        return codes[number - 1] if 1 <= number <= len(codes) else None
    for code in codes:
        spec = catalog().get(code)
        names = {code.lower()}
        if spec is not None:
            names |= {spec.name.lower(), spec.native_name.lower()}
        if text.lower() in names:
            return code
    return None

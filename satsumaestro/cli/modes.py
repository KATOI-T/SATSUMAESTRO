"""モード表示: 色と省略形を REPL / スラッシュコマンドで共有する。"""

from __future__ import annotations

from prompt_toolkit.formatted_text import FormattedText

from satsumaestro.session.models import MainMode, ModeState

MAIN_MODE_COLORS: dict[MainMode, str] = {"plan": "green", "code": "blue"}


def mode_prompt_fragments(state: ModeState) -> FormattedText:
    """行入力プロンプト ``[plan] satsuma> `` / ``[code] satsuma> `` の prompt_toolkit 断片。"""
    color = MAIN_MODE_COLORS[state.main]
    fragments: list[tuple[str, str]] = [("", "["), (f"fg:ansi{color} bold", state.main)]
    if state.main == "code" and state.short_label() != "code":
        fragments.append(("", f":{state.sub.value}"))
    fragments.append(("", "] satsuma> "))
    return FormattedText(fragments)


def mode_markup(state: ModeState, *, full: bool = False) -> str:
    """Rich マークアップ付きのモード表記（``full=True`` でサブモードを常に出す）。"""
    color = MAIN_MODE_COLORS[state.main]
    label = state.label() if full else state.short_label()
    main, sep, sub = label.partition(":")
    return f"[{color}]{main}[/{color}]{sep}{sub}"

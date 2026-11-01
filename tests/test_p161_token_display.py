"""状態行のトークン表示を「出力」に限定し、ターン末尾に内訳の 1 行を出す。"""

from __future__ import annotations

import io

from rich.console import Console

from satsumaestro.cli.render import RichRenderer, _format_tokens, _rough_tokens
from satsumaestro.engine.events import (
    MessageStop,
    TextDelta,
    Thinking,
    ToolUse,
    ToolUseDelta,
    Usage,
)
from satsumaestro.session.models import TokenUsage, ToolResultBlock

WIDTH = 100


def _renderer(
    *, tty: bool = True, context_limit: int = 52_000, height: int = 40
) -> tuple[RichRenderer, io.StringIO]:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=tty, width=WIDTH, height=height, color_system=None)
    r = RichRenderer(console=console, context_limit=context_limit)
    if tty:
        r._refresh_live = lambda: None  # type: ignore[method-assign]
    return r, buf


def _status(r: RichRenderer) -> str:
    """状態行を非 TTY Console へ描いて文字列にする（テスト専用の覗き窓）。"""
    status = r._status_row()
    if status is None:
        return ""
    out = io.StringIO()
    Console(file=out, force_terminal=False, width=WIDTH).print(status)
    return out.getvalue()


def _last_line(buf: io.StringIO) -> str:
    lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def test_format_has_no_unit_and_uses_three_significant_digits() -> None:
    assert _format_tokens(0) == "0"
    assert _format_tokens(999) == "999"
    assert _format_tokens(1000) == "1K"
    assert _format_tokens(1040) == "1.04K"
    assert _format_tokens(1099) == "1.1K"
    assert _format_tokens(12345) == "12.3K"
    assert _format_tokens(120_000) == "120K"
    assert _format_tokens(999_499) == "999K"
    assert _format_tokens(999_500) == "1M"
    assert _format_tokens(1_200_000) == "1.2M"


def test_rough_estimate_counts_japanese_per_character() -> None:
    assert _rough_tokens(40, 0) == 10
    assert _rough_tokens(0, 40) == 40
    r, _ = _renderer()
    r.start_turn()
    r.on_assistant_event(TextDelta(text="あ" * 40))
    assert r._turn_tokens() == 40
    r.on_assistant_event(TextDelta(text="x" * 40 + "い" * 2))
    assert r._turn_tokens() == 52


def test_status_line_shows_output_only_without_an_estimate_mark() -> None:
    r, _ = _renderer()
    r.start_turn()
    assert "出力" not in _status(r)
    r.on_assistant_event(TextDelta(text="x" * 400))
    line = _status(r)
    assert "出力 100" in line and "~" not in line
    assert "入力" not in line and "文脈" not in line and "tok" not in line
    r.on_assistant_event(Usage(input_tokens=12_000, output_tokens=123))
    line = _status(r)
    assert "出力 123" in line
    assert "入力" not in line and "文脈" not in line


def test_status_line_hides_tokens_while_the_count_cannot_move() -> None:
    """ツールの実行中・次の応答の待ち・出し直しの待ちでは数が動かないので出さない。"""
    r, _ = _renderer()
    r.start_turn()
    r.on_assistant_event(TextDelta(text="hello"))
    r.on_assistant_event(Usage(input_tokens=1000, output_tokens=50))
    r.on_assistant_event(MessageStop())
    assert "出力 50" in _status(r)
    r.on_tool_start(ToolUse(id="1", name="read_file", input={"path": "a"}))
    assert "読み込み中" in _status(r) and "出力" not in _status(r)
    r.on_tool_result(ToolResultBlock(tool_use_id="1", tool_name="read_file", output="x"))
    assert "次の操作を考え中" in _status(r) and "出力" not in _status(r)
    r.on_assistant_event(Thinking(text="t" * 40))
    assert "思考を整理中" in _status(r) and "出力 60" in _status(r)
    r.on_assistant_event(TextDelta(text="again"))
    assert "応答生成中" in _status(r) and "出力 61" in _status(r)
    r.on_stream_aborted(timeout_recovery_s=120)
    assert "出し直しを待っています" in _status(r) and "出力" not in _status(r)


def test_counter_keeps_moving_in_the_second_message() -> None:
    """2 通目以降も本文が流れれば数字が進む（以前は次の Usage まで止まっていた）。"""
    r, _ = _renderer()
    r.start_turn()
    r.on_assistant_event(TextDelta(text="x" * 40))
    r.on_assistant_event(Usage(input_tokens=1000, output_tokens=50))
    r.on_assistant_event(MessageStop())
    r.on_tool_start(ToolUse(id="1", name="read_file", input={"path": "a"}))
    r.on_tool_result(ToolResultBlock(tool_use_id="1", tool_name="read_file", output="x"))
    assert r._turn_tokens() == 50
    r.on_assistant_event(TextDelta(text="y" * 80))
    assert r._turn_tokens() == 70
    assert "出力 70" in _status(r)
    r.on_assistant_event(Usage(input_tokens=1100, output_tokens=33))
    assert r._turn_tokens() == 83


def test_consecutive_usages_without_a_tool_call_are_summed() -> None:
    """ツールを挟まずに assistant メッセージが続く経路（出し直し）で 1 通目の実測が消えない。"""
    r, _ = _renderer()
    r.start_turn()
    r.on_assistant_event(TextDelta(text="first"))
    r.on_assistant_event(Usage(input_tokens=1000, output_tokens=40))
    r.on_assistant_event(MessageStop())
    r.on_assistant_event(TextDelta(text="second"))
    r.on_assistant_event(Usage(input_tokens=1050, output_tokens=7))
    r.on_assistant_event(MessageStop())
    assert r._turn_tokens() == 47
    assert r._in_total == 2050 and r._requests == 2 and r._ctx_last == 1057


def test_aborted_message_without_usage_stays_as_an_estimate() -> None:
    """Usage が来ずに終わった通（打ち切り）は概算として残る。"""
    r, _ = _renderer()
    r.start_turn()
    r.on_assistant_event(Thinking(text="t" * 40))
    r.on_assistant_event(ToolUseDelta(index=0, name="write_file", args_delta="a" * 40))
    r.on_stream_aborted()
    assert r._turn_tokens() == 20
    r.on_assistant_event(TextDelta(text="z" * 40))
    r.on_assistant_event(Usage(input_tokens=500, output_tokens=12))
    r.on_assistant_event(MessageStop())
    assert r._turn_tokens() == 32
    assert "出力 32" in _status(r)


def test_status_line_stays_on_one_row() -> None:
    r, _ = _renderer(height=10)
    r.start_turn()
    r.on_assistant_event(TextDelta(text="hello"))
    r.on_assistant_event(Usage(input_tokens=123_456, output_tokens=7_890))
    out = io.StringIO()
    Console(file=out, force_terminal=False, width=WIDTH, height=10).print(r._live_renderable())
    rows = out.getvalue().rstrip("\n").split("\n")
    assert "応答生成中" in rows[-1] and "出力 7.89K" in rows[-1]
    assert "出力" not in "\n".join(rows[:-1])


def test_end_turn_prints_the_breakdown_line() -> None:
    r, buf = _renderer(context_limit=52_000)
    r.start_turn()
    r.on_assistant_event(TextDelta(text="hello"))
    r.on_assistant_event(Usage(input_tokens=12_000, output_tokens=300))
    r.on_assistant_event(MessageStop())
    r.on_tool_start(ToolUse(id="1", name="read_file", input={"path": "a"}))
    r.on_tool_result(ToolResultBlock(tool_use_id="1", tool_name="read_file", output="x"))
    r.on_assistant_event(TextDelta(text="done"))
    r.on_assistant_event(Usage(input_tokens=13_000, output_tokens=1_100))
    r.on_assistant_event(MessageStop())
    r.end_turn()
    assert _last_line(buf) == "— 出力 1.4K · 入力 25K（2 回） · 文脈 14.1K/52K"
    assert "tok" not in buf.getvalue()


def test_end_turn_adds_delegated_usage_only_when_present() -> None:
    r, buf = _renderer()
    r.start_turn()
    r.on_assistant_event(TextDelta(text="hello"))
    r.on_assistant_event(Usage(input_tokens=1_000, output_tokens=50))
    r.on_assistant_event(MessageStop())
    r.end_turn(delegated=TokenUsage(input_tokens=40_100, output_tokens=5_600))
    assert _last_line(buf) == (
        "— 出力 50 · 入力 1K（1 回） · 委譲 入力 40.1K / 出力 5.6K · 文脈 1.05K/52K"
    )

    r2, buf2 = _renderer()
    r2.start_turn()
    r2.on_assistant_event(TextDelta(text="hello"))
    r2.on_assistant_event(Usage(input_tokens=1_000, output_tokens=50))
    r2.on_assistant_event(MessageStop())
    r2.end_turn(delegated=TokenUsage())
    assert "委譲" not in buf2.getvalue()


def test_end_turn_without_usage_prints_the_estimate_only() -> None:
    """Usage を返さないプロバイダ: 出力の概算だけで、入力・文脈は書かない。"""
    r, buf = _renderer()
    r.start_turn()
    r.on_assistant_event(TextDelta(text="x" * 400))
    r.on_assistant_event(MessageStop())
    r.end_turn()
    assert _last_line(buf) == "— 出力 100"


def test_end_turn_without_a_limit_shows_the_numerator_only() -> None:
    r, buf = _renderer(context_limit=0)
    r.start_turn()
    r.on_assistant_event(TextDelta(text="hello"))
    r.on_assistant_event(Usage(input_tokens=1_000, output_tokens=50))
    r.on_assistant_event(MessageStop())
    r.end_turn()
    assert _last_line(buf) == "— 出力 50 · 入力 1K（1 回） · 文脈 1.05K"


def test_end_turn_prints_nothing_when_nothing_was_counted() -> None:
    r, buf = _renderer()
    r.start_turn()
    r.end_turn()
    assert "出力" not in buf.getvalue()


def test_counters_reset_between_turns() -> None:
    r, buf = _renderer()
    r.start_turn()
    r.on_assistant_event(TextDelta(text="hello"))
    r.on_assistant_event(Usage(input_tokens=9_000, output_tokens=900))
    r.on_assistant_event(MessageStop())
    r.end_turn()
    r.start_turn()
    r.on_assistant_event(TextDelta(text="again"))
    r.on_assistant_event(Usage(input_tokens=10_000, output_tokens=10))
    r.on_assistant_event(MessageStop())
    r.end_turn()
    assert _last_line(buf) == "— 出力 10 · 入力 10K（1 回） · 文脈 10K/52K"


def test_non_tty_output_is_unchanged() -> None:
    """非 TTY（prompt のパイプ・評価）では末尾行を出さない（従来からの不変）。"""
    r, buf = _renderer(tty=False)
    r.start_turn()
    r.on_assistant_event(TextDelta(text="hello"))
    r.on_assistant_event(Usage(input_tokens=1_000, output_tokens=50))
    r.on_assistant_event(MessageStop())
    r.end_turn(delegated=TokenUsage(input_tokens=1, output_tokens=1))
    out = buf.getvalue()
    assert "hello" in out
    assert "出力" not in out and "入力" not in out and "文脈" not in out and "委譲" not in out

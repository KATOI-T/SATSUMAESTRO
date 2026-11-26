"""状態行: 光の流れるラベルと経過だけを 1 行で出し、合計時間はターン末尾の内訳に出す。"""

from __future__ import annotations

import io

import pytest
from rich.cells import cell_len
from rich.console import Console
from rich.spinner import Spinner
from rich.text import Text

from satsumaestro.cli import render as render_mod
from satsumaestro.cli.render import (
    _LOAD_STALL_S,
    _SHIMMER_BASE,
    _SHIMMER_GAP_S,
    _SHIMMER_RADIUS,
    _SHIMMER_SPEED,
    RichRenderer,
    _format_elapsed,
    _shimmer,
)
from satsumaestro.engine.events import MessageStop, TextDelta, ToolUse, Usage
from satsumaestro.i18n.text import pinned_ui_language
from satsumaestro.session.models import ToolResultBlock


class _Clock:
    """``time.monotonic`` の代わり。"""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    c = _Clock()
    monkeypatch.setattr(render_mod.time, "monotonic", c)
    return c


def _renderer(**kwargs: object) -> tuple[RichRenderer, io.StringIO]:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=100, color_system=None)
    r = RichRenderer(console=console, **kwargs)  # type: ignore[arg-type]
    r._refresh_live = lambda: None  # type: ignore[method-assign]
    return r, buf


def _status(r: RichRenderer, *, width: int = 120) -> str:
    status = r._status_row()
    assert status is not None
    out = io.StringIO()
    Console(file=out, force_terminal=False, width=width).print(status)
    return out.getvalue()


def _status_text(r: RichRenderer) -> Text:
    """状態行のスピナーに載る文（字ごとの明るさを見る）。"""
    status = r._status_row()
    assert isinstance(status, render_mod._StatusLine)
    console = Console(file=io.StringIO(), width=120)
    spinner = next(iter(status.__rich_console__(console, console.options)))
    assert isinstance(spinner, Spinner) and isinstance(spinner.text, Text)
    return spinner.text


def _levels(text: Text) -> list[int]:
    """字ごとの明るさ（色の赤の値）。色の無い字（dim）は灰と同じに数える。"""
    levels = [_SHIMMER_BASE[0]] * len(text.plain)
    for span in text.spans:
        style = span.style
        if not isinstance(style, str) and style.color is not None and style.color.triplet:
            for i in range(span.start, span.end):
                levels[i] = style.color.triplet.red
    return levels


def _lit(text: Text) -> str:
    """灰より明るく描かれた字だけを並べる。"""
    return "".join(
        ch for ch, v in zip(text.plain, _levels(text), strict=True) if v > _SHIMMER_BASE[0]
    )


def _period(label: str) -> float:
    return (cell_len(label) + 2 * _SHIMMER_RADIUS) / _SHIMMER_SPEED + _SHIMMER_GAP_S


def _last_line(buf: io.StringIO) -> str:
    lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _in_a_tool_after_output(r: RichRenderer, clock: _Clock) -> None:
    """ターンの 10 秒目に bash を始め、その 8 秒後（ターンの 18 秒目）の状態にする。"""
    r.start_turn()
    r.on_assistant_event(TextDelta(text="hello"))
    r.on_assistant_event(Usage(input_tokens=1000, output_tokens=50))
    r.on_assistant_event(MessageStop())
    clock.now += 10
    r.on_tool_start(ToolUse(id="1", name="bash", input={"command": "make test"}))
    clock.now += 8


# ── 並び ──


def test_one_row_with_the_label_and_the_elapsed_while_a_tool_runs(clock: _Clock) -> None:
    r, _ = _renderer()
    _in_a_tool_after_output(r, clock)
    line = _status(r)
    assert line.count("\n") == 1
    assert line.rstrip().endswith("コマンド実行中…  8s")
    assert "·" not in line


def test_no_tokens_while_the_reply_is_written(clock: _Clock) -> None:
    r, _ = _renderer()
    r.start_turn()
    clock.now += 3
    r.on_assistant_event(TextDelta(text="x" * 200))
    clock.now += 4
    assert _status(r).rstrip().endswith("応答生成中…  4s")


def test_no_interrupt_hint_and_no_turn_total(clock: _Clock) -> None:
    r, _ = _renderer()
    r.start_turn()
    line = _status(r)
    assert "考え中…" in line and "ctrl+c" not in line
    assert "トークン" not in line
    clock.now += 5
    r.on_phase("履歴を要約中…")
    clock.now += 3
    line = _status(r)
    assert "3s" in line and "計" not in line and "8s" not in line


def test_english_screen(clock: _Clock) -> None:
    with pinned_ui_language("en"):
        r, _ = _renderer()
        r.start_turn()
        r.on_assistant_event(TextDelta(text="x" * 200))
        clock.now += 4
        line = _status(r)
    assert line.rstrip().endswith("…  4s") and "tokens" not in line
    assert "total" not in line and "interrupt" not in line


def test_model_loading_line(clock: _Clock) -> None:
    r, _ = _renderer(loading_model="gpt-oss:120b")
    r.start_turn()
    clock.now += 4
    line = _status(r)
    assert "gpt-oss:120b をロード中…  4s" in line
    assert "━" in line  # 往復の帯はそのまま


# ── ターン末尾の合計時間 ──


def test_turn_total_is_printed_last_on_the_breakdown_line(clock: _Clock) -> None:
    r, buf = _renderer()
    _in_a_tool_after_output(r, clock)
    r.on_tool_result(ToolResultBlock(tool_use_id="1", tool_name="bash", output="ok"))
    clock.now += 105
    r.end_turn()
    assert _last_line(buf) == "  — 出力 50 · 入力 1K（1 回） · 文脈 1.05K · 計 2m03s"


def test_short_turn_has_no_total(clock: _Clock) -> None:
    r, buf = _renderer()
    r.start_turn()
    r.on_assistant_event(TextDelta(text="x" * 400))
    clock.now += 1.9
    r.end_turn()
    assert _last_line(buf) == "  — 出力 100"


def test_turn_total_alone_when_nothing_was_counted(clock: _Clock) -> None:
    r, buf = _renderer()
    r.start_turn()
    clock.now += 5
    r.end_turn()  # 1 イベントも来なかったターン（エラー・中断）
    assert _last_line(buf) == "  — 計 5s"


def test_english_turn_total(clock: _Clock) -> None:
    with pinned_ui_language("en"):
        r, buf = _renderer()
        r.start_turn()
        r.on_assistant_event(TextDelta(text="x" * 400))
        clock.now += 72
        r.end_turn()
    assert _last_line(buf) == "  — output 100 · total 1m12s"


# ── ラベルの上を流れる光 ──


def test_light_glides_from_left_to_right_then_rests_and_repeats() -> None:
    label = "abcdefgh"
    period = _period(label)
    frames = [_levels(_shimmer(label, k / 60)) for k in range(round(period * 60))]
    peaks = [
        max(range(len(label)), key=lambda i: f[i]) for f in frames if max(f) > _SHIMMER_BASE[0]
    ]
    assert peaks[0] == 0 and peaks[-1] == len(label) - 1
    assert peaks == sorted(peaks)  # 一番明るい字は左から右へ進むだけ
    assert frames[0] == [_SHIMMER_BASE[0]] * len(label)  # 左から入る前
    rest = [f for f in frames if max(f) == _SHIMMER_BASE[0]]
    assert len(rest) >= _SHIMMER_GAP_S * 60  # 抜けた後に間を置く
    assert _levels(_shimmer(label, 0.3 + period)) == _levels(_shimmer(label, 0.3))  # 繰り返す


def test_light_changes_smoothly_between_frames() -> None:
    label = "次の操作を考え中…"
    frames = [_levels(_shimmer(label, k / 60)) for k in range(round(_period(label) * 60))]
    pairs = zip(frames, frames[1:], strict=False)
    steps = [abs(a - b) for f, g in pairs for a, b in zip(f, g, strict=True)]
    assert max(steps) <= 60  # 60 fps の 1 コマで、どの字も一気に灰から白へ飛ばない
    assert max(max(f) for f in frames) >= 230  # 中心は灰より十分に明るい
    widths = {len(_lit(_shimmer(label, k / 60))) for k in range(round(_period(label) * 60))}
    assert max(widths) >= 3  # 光は 1 字より広い（なだらかな山形）


@pytest.mark.parametrize(
    ("setup", "label"),
    [
        (lambda r: None, "考え中…"),
        (
            lambda r: r.on_tool_start(ToolUse(id="1", name="bash", input={"command": "ls"})),
            "コマンド実行中…",
        ),
        (
            lambda r: (
                r.on_tool_start(ToolUse(id="1", name="bash", input={"command": "ls"})),
                r.on_tool_result(ToolResultBlock(tool_use_id="1", tool_name="bash", output="")),
            ),
            "次の操作を考え中…",
        ),
        (lambda r: r.on_assistant_event(TextDelta(text="x" * 40)), "応答生成中…"),
        (lambda r: r.on_phase("履歴を要約中…"), "履歴を要約中…"),
    ],
)
def test_every_label_has_the_light(clock: _Clock, setup: object, label: str) -> None:
    r, _ = _renderer()
    r.start_turn()
    setup(r)  # type: ignore[operator]
    base = clock.now + 3
    seen: set[str] = set()
    for k in range(180):  # 3 秒（どのラベルでも 1 周期より長い）
        clock.now = base + k / 60
        text = _status_text(r)
        assert text.plain.startswith(label)
        lit = _lit(text)
        assert lit == "" or lit in label  # 経過は光らない
        seen.add(lit)
    assert any(s.startswith(label[0]) for s in seen) and any(s.endswith(label[-1]) for s in seen)
    assert "" in seen


def test_loading_label_has_the_light_until_it_stalls(clock: _Clock) -> None:
    r, _ = _renderer(loading_model="gpt-oss:120b")
    r.start_turn()
    seen = set()
    for _ in range(120):
        seen.add(_lit(_status_text(r)))
        clock.now += 1 / 60
    assert seen != {""}
    clock.now += _LOAD_STALL_S
    for _ in range(120):
        assert _lit(_status_text(r)) == ""  # 停滞後は「進んでいる」ように見せない
        clock.now += 1 / 60


@pytest.mark.parametrize(
    ("seconds", "shown"),
    [
        (0, "0s"),
        (2.9, "2s"),
        (59.99, "59s"),
        (60, "1m00s"),
        (72, "1m12s"),
        (3599, "59m59s"),
        (3600, "1h00m"),
        (3725, "1h02m"),
    ],
)
def test_elapsed_format(seconds: float, shown: str) -> None:
    assert _format_elapsed(seconds) == shown

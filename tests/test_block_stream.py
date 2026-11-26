"""本文をまとまりごとに出す（``display.text``）。鍵・ネットワーク不要。

- block: 書きかけの段落は最下部に描かず（状態行だけ）、完成したら一度にスクロールバックへ出る
- block: まとまりが ``reveal_after_s`` を超えたら、文の切れ目まで最下部に見せる
  （書きかけの文は見せない）
- block: 長いコードブロックは ``reveal_after_s`` を超えたら完成した行から出る。
  最後のスクロールバックは live と同じ
- full: 応答の区切りまで何も出さない
- block / full: 生成中のファイルは見出しだけ（本文の末尾を出さない）
- 設定 ``display`` の既定と検証
"""

from __future__ import annotations

import io
import json
import random
import re

import pytest
from pydantic import ValidationError
from rich.console import Console

from satsumaestro.cli import render as render_mod
from satsumaestro.cli.render import RichRenderer, _sentence_prefix
from satsumaestro.config.models import DisplayConfig, RuntimeConfig, TextMode
from satsumaestro.engine.events import TextDelta, ToolUseDelta

_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
WIDTH = 70


class _Clock:
    """``time.monotonic`` の代わり（まとまりの経過秒を進める）。"""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    c = _Clock()
    monkeypatch.setattr(render_mod.time, "monotonic", c)
    return c


def _renderer(mode: TextMode, *, height: int = 24) -> tuple[RichRenderer, io.StringIO]:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=WIDTH, height=height, color_system=None)
    r = RichRenderer(console=console, text_mode=mode, reveal_after_s=3.0)
    # 最下部の再描画をスクロールバックの検査に混ぜない（確定の順序と中身だけを見る）。
    r._refresh_live = lambda: None  # type: ignore[method-assign]
    return r, buf


def _feed(r: RichRenderer, text: str, *, seed: int = 1) -> None:
    rng = random.Random(seed)
    i = 0
    while i < len(text):
        n = rng.randint(1, 7)
        r.on_assistant_event(TextDelta(text=text[i : i + n]))
        i += n


def _plain(s: str) -> str:
    return _ANSI.sub("", s)


def _bottom(r: RichRenderer) -> str:
    out = io.StringIO()
    Console(file=out, force_terminal=False, width=WIDTH, height=24).print(r._live_renderable())
    return out.getvalue()


# ── block: まとまりごとに出す ──


def test_unfinished_paragraph_is_not_drawn_at_the_bottom(clock: _Clock) -> None:
    r, buf = _renderer("block")
    r.start_turn()
    _feed(r, "最初の段落の一文目。二文目の途中")
    assert "最初の段落" not in _bottom(r)
    assert "最初の段落" not in _plain(buf.getvalue())
    assert "応答" in _bottom(r)  # 状態行は出ている


def test_finished_paragraph_goes_to_the_scrollback_at_once(clock: _Clock) -> None:
    r, buf = _renderer("block")
    r.start_turn()
    _feed(r, "最初の段落。\n\n次の段落の途中")
    out = _plain(buf.getvalue())
    assert "最初の段落。" in out
    assert "次の段落" not in out and "次の段落" not in _bottom(r)


def test_long_paragraph_is_shown_up_to_the_last_sentence_after_the_delay(clock: _Clock) -> None:
    r, _ = _renderer("block")
    r.start_turn()
    _feed(r, "一文目です。二文目")
    clock.now += 2.9
    assert "一文目" not in _bottom(r)
    clock.now += 0.2  # まとまりの始まりから 3 秒を超えた（描くたびに時刻を見る）
    bottom = _bottom(r)
    assert "一文目です。" in bottom and "二文目" not in bottom
    _feed(r, "も終わり。三")
    bottom = _bottom(r)
    assert "二文目も終わり。" in bottom and "三" not in bottom


def test_reveal_clock_restarts_for_the_next_block(clock: _Clock) -> None:
    r, buf = _renderer("block")
    r.start_turn()
    _feed(r, "一つ目。")
    clock.now += 5
    _feed(r, "\n\n二つ目の文。")
    assert "一つ目。" in _plain(buf.getvalue())
    assert "二つ目" not in _bottom(r)  # 新しいまとまりはまだ 3 秒経っていない
    clock.now += 3
    assert "二つ目の文。" in _bottom(r)


@pytest.mark.parametrize(
    ("text", "shown"),
    [
        ("一文目。二文目", "一文目。"),
        ("本当？　はい", "本当？"),
        ("First one. Second", "First one."),
        ("Done!\nNext", "Done!\n"),
        ("pi is 3.14 and", ""),
        ("ends with a period.", ""),  # 次の字が来るまで決めない
        ("- item 1\n- item 2\n- ite", "- item 1\n- item 2\n"),
        ("no boundary yet", ""),
    ],
)
def test_sentence_prefix(text: str, shown: str) -> None:
    assert _sentence_prefix(text) == shown


# ── block: コードブロック ──

CODE_REPLY = (
    "コードです。\n\n```python\n"
    + "\n".join(f"x{i} = {i}  # line-{i:03d}" for i in range(1, 61))
    + "\n```\n\n以上です。"
)


def _run(mode: TextMode, text: str, clock: _Clock, *, step: float = 0.0) -> str:
    r, buf = _renderer(mode)
    r.start_turn()
    rng = random.Random(3)
    i = 0
    while i < len(text):
        n = rng.randint(1, 7)
        r.on_assistant_event(TextDelta(text=text[i : i + n]))
        clock.now += step
        i += n
    r.end_turn()
    # ターン末尾の合計時間は流す速さで変わるので比べない。
    return re.sub(r" · 計 \S+", "", _plain(buf.getvalue()))


def test_long_code_block_starts_line_by_line_after_the_delay(clock: _Clock) -> None:
    r, buf = _renderer("block")
    r.start_turn()
    head, _, rest = CODE_REPLY.partition("line-030")
    _feed(r, head)
    assert "line-001" not in _plain(buf.getvalue())  # 3 秒経つまでは閉じるのを待つ
    clock.now += 3
    _feed(r, "line-030" + rest[:40])
    out = _plain(buf.getvalue())
    assert "line-001" in out and "line-029" in out
    assert "以上です" not in out


def test_streamed_code_block_ends_in_the_same_scrollback_as_live(clock: _Clock) -> None:
    assert _run("block", CODE_REPLY, clock, step=0.05) == _run("live", CODE_REPLY, clock)


def test_short_reply_ends_in_the_same_scrollback_as_live(clock: _Clock) -> None:
    text = "# 見出し\n\n段落です。\n\n- a\n- b\n\n```sh\necho hi\n```\n\n終わり。"
    assert _run("block", text, clock) == _run("live", text, clock)


# ── full: 区切りまで出さない ──


def test_full_mode_prints_nothing_until_the_segment_ends(clock: _Clock) -> None:
    r, buf = _renderer("full")
    r.start_turn()
    _feed(r, "最初の段落。\n\n次の段落。\n\n")
    clock.now += 10
    assert "最初の段落" not in _plain(buf.getvalue())
    assert "最初の段落" not in _bottom(r)
    r.end_turn()
    out = _plain(buf.getvalue())
    assert "最初の段落。" in out and "次の段落。" in out


# ── 生成中のファイル ──


@pytest.mark.parametrize("mode", ["block", "full"])
def test_tool_arguments_show_only_the_heading(mode: TextMode, clock: _Clock) -> None:
    r, _ = _renderer(mode)
    r.start_turn()
    args = json.dumps({"path": "a.css", "content": "body {}\n/*secret-line*/\n"})
    r.on_assistant_event(ToolUseDelta(index=0, name="write_file", args_delta=args))
    bottom = _bottom(r)
    assert "a.css" in bottom and "write_file" in bottom
    assert "secret-line" not in bottom


def test_live_mode_still_shows_the_written_lines(clock: _Clock) -> None:
    r, _ = _renderer("live")
    r.start_turn()
    args = json.dumps({"path": "a.css", "content": "body {}\n/*secret-line*/\n"})
    r.on_assistant_event(ToolUseDelta(index=0, name="write_file", args_delta=args))
    assert "secret-line" in _bottom(r)


# ── 設定 ──


def test_display_config_defaults_and_validation() -> None:
    cfg = RuntimeConfig()
    assert cfg.display.text == "stream" and cfg.display.reveal_after_s == 3.0
    assert RuntimeConfig.model_validate({"display": {"text": "live"}}).display.text == "live"
    assert RuntimeConfig.model_validate({"display": {"text": "block"}}).display.text == "block"
    with pytest.raises(ValidationError):
        DisplayConfig.model_validate({"text": "fast"})
    with pytest.raises(ValidationError):
        DisplayConfig.model_validate({"reveal_after_s": -1})
    with pytest.raises(ValidationError):
        DisplayConfig.model_validate({"speed": 1})

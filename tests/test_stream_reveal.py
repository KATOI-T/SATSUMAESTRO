"""文ごとの早送り（``display.text = "stream"``）。鍵・ネットワーク不要。

- 文の切れ目が届くまで、最下部に本文を出さない（状態行だけ）
- 届いた文は長さに依らず ``stream_sentence_s`` 秒で流れ、書きかけの文は出ない。コードは行ごと。
  0 なら流さずに文ごと出す
- 溜まった分も同じ秒数で流し切る
- まとまりは流し終わってから確定する
- 区切りでは残りを流し切ってから確定する（待つのは同じ秒数まで。最下部が無ければ待たない）
- 文の切れ目の来ない書きかけは ``_SENTENCE_WAIT_S`` を超えたら届いた分まで流す
- 最後のスクロールバックは流す速さに依らず 1 つの Markdown（full）と同じ。まとまりを分けて
  確定しても、間の 1 行が 1 つの Markdown と同じになる（live も）
- 最下部の領域は stream だけ 60 fps
"""

from __future__ import annotations

import io
import random
import re

import pytest
from rich.console import Console, RenderableType

from satsumaestro.cli import render as render_mod
from satsumaestro.cli.render import _SENTENCE_WAIT_S, RichRenderer
from satsumaestro.config.models import TextMode
from satsumaestro.engine.events import TextDelta

_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
WIDTH = 70
SENTENCE_S = 0.2  # 1 文を流す秒数（既定）


class _Clock:
    """``time.monotonic`` の代わり。``sleep`` は時刻を進めるだけ（待った秒を覚える）。"""

    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    c = _Clock()
    monkeypatch.setattr(render_mod.time, "monotonic", c)
    monkeypatch.setattr(render_mod.time, "sleep", c.sleep)
    return c


class _Region:
    """最下部の領域の代わり（流し切りを待つかどうかだけを見る）。"""

    def update(self, renderable: RenderableType) -> None:
        pass

    def stop(self) -> None:
        pass


def _renderer(
    mode: TextMode = "stream", *, sentence_s: float = SENTENCE_S
) -> tuple[RichRenderer, io.StringIO]:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=WIDTH, height=24, color_system=None)
    r = RichRenderer(console=console, text_mode=mode, stream_sentence_s=sentence_s)
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


# ── 流し方 ──


def test_nothing_is_drawn_until_a_sentence_ends(clock: _Clock) -> None:
    r, buf = _renderer()
    r.start_turn()
    _feed(r, "最初の文の途中")
    clock.now += 1.0
    bottom = _bottom(r)
    assert "最初" not in bottom and "応答" in bottom  # 状態行だけ
    assert "最初" not in _plain(buf.getvalue())


@pytest.mark.parametrize("length", [10, 60])
def test_a_finished_sentence_flows_over_the_sentence_time(length: int, clock: _Clock) -> None:
    """短い文も長い文も ``stream_sentence_s`` 秒かけて流れる。"""
    r, _ = _renderer()
    r.start_turn()
    _feed(r, "あ" * (length - 1) + "。次の文")
    assert "あ" not in _bottom(r)  # 届いた瞬間はまだ 0 字
    clock.now += SENTENCE_S / 2
    assert length // 2 - 2 <= _bottom(r).count("あ") <= length // 2
    clock.now += SENTENCE_S / 2 + 0.001
    bottom = _bottom(r)
    assert bottom.count("あ") == length - 1 and "。" in bottom
    assert "次の文" not in bottom  # 書きかけの文は出さない


def test_zero_seconds_shows_each_sentence_at_once(clock: _Clock) -> None:
    r, _ = _renderer(sentence_s=0)
    r.start_turn()
    _feed(r, "一文目です。二文目")
    bottom = _bottom(r)
    assert "一文目です。" in bottom and "二文目" not in bottom


def test_a_latin_sentence_ends_at_a_period_followed_by_a_space(clock: _Clock) -> None:
    r, _ = _renderer()
    r.start_turn()
    _feed(r, "First one. Second one is still")
    clock.now += 1.0
    bottom = _bottom(r)
    assert "First one." in bottom and "Second" not in bottom


def test_code_lines_flow_one_line_at_a_time(clock: _Clock) -> None:
    r, _ = _renderer()
    r.start_turn()
    _feed(r, "```python\nx_one = 1\ny_two = 2\nz_thr")
    clock.now += 1.0
    bottom = _bottom(r)
    assert "x_one = 1" in bottom and "y_two = 2" in bottom and "z_thr" not in bottom


def test_a_large_backlog_is_shown_within_the_sentence_time(clock: _Clock) -> None:
    r, _ = _renderer()
    r.start_turn()
    text = "長い文です。" * 200  # 1200 字が一度に溜まっても、遅れは 1 文の秒数まで
    _feed(r, text + "続")
    assert r._clock is not None
    clock.now += SENTENCE_S / 2
    assert r._clock.at(clock.now) < len(text)
    clock.now += SENTENCE_S / 2 + 0.001
    assert r._clock.at(clock.now) == len(text)


def test_a_long_unfinished_sentence_is_shown_after_waiting(clock: _Clock) -> None:
    """待ちは書きかけが出はじめた時から数え、途中の差分では延びない。"""
    r, _ = _renderer()
    r.start_turn()
    _feed(r, "句点の来ない長い文")
    clock.now += _SENTENCE_WAIT_S - 0.1
    r.on_assistant_event(TextDelta(text="が"))
    assert "句点" not in _bottom(r)  # まだ待つ
    clock.now += 0.1 + SENTENCE_S + 0.01
    assert "句点の来ない長い文が" in _bottom(r)
    r.on_assistant_event(TextDelta(text="続く"))
    clock.now += SENTENCE_S + 0.01
    assert "句点の来ない長い文が続く" in _bottom(r)


def test_unfinished_text_is_shown_after_the_wait_without_another_delta(clock: _Clock) -> None:
    """届くのが止まっても、待ちが尽きたら次の差分を待たずに書きかけを流す。"""
    r, _ = _renderer()
    r.start_turn()
    _feed(r, "前の文。届いたきり止まる書きかけ")
    clock.now += _SENTENCE_WAIT_S - 0.1
    bottom = _bottom(r)
    assert "前の文。" in bottom and "書きかけ" not in bottom
    clock.now += 0.1 + SENTENCE_S + 0.01
    assert "前の文。届いたきり止まる書きかけ" in _bottom(r)


def test_a_first_unfinished_sentence_is_shown_after_the_wait(clock: _Clock) -> None:
    """切れ目がまだ 1 つも無い書きかけも、待ちが尽きたら流す。"""
    r, _ = _renderer()
    r.start_turn()
    _feed(r, "最初の文が止まる")
    clock.now += _SENTENCE_WAIT_S - 0.1
    assert "最初" not in _bottom(r)
    clock.now += 0.1 + SENTENCE_S + 0.01
    assert "最初の文が止まる" in _bottom(r)


def test_a_sentence_end_wins_over_the_wait(clock: _Clock) -> None:
    """待ちが尽きた所に切れ目が届いたら、切れ目まで流す（後ろの書きかけは出さない）。"""
    r, _ = _renderer()
    r.start_turn()
    _feed(r, "前の文。書きかけ")
    clock.now += _SENTENCE_WAIT_S + 0.1
    r.on_assistant_event(TextDelta(text="の文。次の途"))
    clock.now += 1.0
    bottom = _bottom(r)
    assert "書きかけの文。" in bottom and "次の途" not in bottom


# ── 確定 ──


def test_a_block_is_committed_only_after_it_has_been_shown(clock: _Clock) -> None:
    r, buf = _renderer()
    r.start_turn()
    _feed(r, "最初の段落。\n\n次の段落の一文目。")
    assert "最初の段落" not in _plain(buf.getvalue())  # まだ流していない
    clock.now += 1.0
    r.on_assistant_event(TextDelta(text="二"))  # 確定は次に届いたとき
    out = _plain(buf.getvalue())
    assert "最初の段落。" in out and "次の段落" not in out
    bottom = _bottom(r)
    assert "次の段落の一文目。" in bottom and "最初の段落" not in bottom


def _visible(r: RichRenderer, buf: io.StringIO) -> list[str]:
    """スクロールバックと最下部の本文（状態行を除く）を、画面に並ぶ順に。"""
    lines = _plain(buf.getvalue()).split("\n")[:-1]
    bottom = _bottom(r).rstrip("\n").split("\n")[:-2]  # 状態行とその上の空き
    return [line.rstrip() for line in lines + bottom]


@pytest.mark.parametrize("step", [0.25, 0.5])  # 1 文の秒数より間を空けて 3 字ずつ届く
def test_text_does_not_move_when_it_is_committed(step: float, clock: _Clock) -> None:
    """流し終わった時点の画面（確定分＋最下部）が、最後のスクロールバックと同じ位置に並ぶ。

    文の途中（```` `` ```` まで打った開きフェンスなど）は 1 フレームに満たない間の形なので見ない。
    """
    text = (
        "最初の段落です。\n\n- 項目一\n- 項目二\n\n```sh\necho hi\n```\n\n"
        "次の段落です。二文目です。\n\n---\n\n終わり。"
    )
    r, buf = _renderer()
    r.start_turn()
    frames = []
    for i in range(0, len(text), 3):
        r.on_assistant_event(TextDelta(text=text[i : i + 3]))
        clock.now += step
        if r._clock is None or r._clock.at(clock.now) == r._clock.end:
            frames.append(_visible(r, buf))
    assert len(frames) > 10
    r.end_turn()
    final = [line.rstrip() for line in _plain(buf.getvalue()).split("\n")]
    for frame in frames:
        for k, line in enumerate(frame):
            if k < len(frame) - 1:
                assert line == final[k], (k, frame)
            else:
                assert final[k].startswith(line), (k, frame)  # 流している途中の行


def test_the_rest_flows_before_the_segment_ends(clock: _Clock) -> None:
    r, buf = _renderer()
    r._live = _Region()  # type: ignore[assignment]
    r.start_turn()
    _feed(r, "あ" * 59 + "。最後の文")  # 最後の文には切れ目が無い
    r.end_turn()
    assert clock.slept == [pytest.approx(SENTENCE_S)]  # 書きかけの 4 字も同じ秒数で流す
    out = _plain(buf.getvalue())
    assert "あ" * 30 in out and "最後の文" in out


def test_waiting_at_the_segment_end_is_capped(clock: _Clock) -> None:
    r, _ = _renderer()
    r._live = _Region()  # type: ignore[assignment]
    r.start_turn()
    _feed(r, "長い文です。" * 200)
    clock.now += SENTENCE_S / 2
    r.end_turn()
    assert clock.slept == [pytest.approx(SENTENCE_S / 2)]  # 流している残りの分だけ待つ


def test_no_wait_without_the_bottom_region(clock: _Clock) -> None:
    r, buf = _renderer()
    r.start_turn()
    _feed(r, "一文目。二文目")
    r.end_turn()
    assert clock.slept == []
    assert "一文目。二文目" in _plain(buf.getvalue())


SHORT = "# 見出し\n\n段落です。二文目。\n\n- a\n- b\n\n```sh\necho hi\n```\n\n終わり。"
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


@pytest.mark.parametrize("step", [0.0, 0.01, 0.2])
@pytest.mark.parametrize("text", [SHORT, CODE_REPLY], ids=["short", "tall-code"])
def test_final_scrollback_does_not_depend_on_timing(text: str, step: float, clock: _Clock) -> None:
    """流す速さでまとまりの切れ目が変わっても、最後は 1 つの Markdown（full）と同じ。"""
    full = _run("full", text, clock)
    assert _run("stream", text, clock, step=step) == full
    assert _run("live", text, clock) == full


_BLOCKS = {
    "paragraph": "段落です。二文目。",
    "h1": "# 見出し1",
    "h2": "## 見出し2",
    "list": "- a\n- b",
    "ordered": "1. one\n2. two",
    "code": "```sh\necho hi\n```",
    "table": "| a | b |\n|---|---|\n| 1 | 2 |",
    "quote": "> 引用です。",
    "rule": "---",
}


@pytest.mark.parametrize(("mode", "step"), [("live", 0.0), ("stream", 0.01), ("stream", 0.3)])
def test_blocks_printed_one_by_one_are_spaced_like_one_markdown(
    mode: TextMode, step: float, clock: _Clock
) -> None:
    """まとまりを分けて確定しても、間の 1 行が 1 つの Markdown と同じ（リスト・表・引用は自分で
    空け、区切り線の後ろは線が空ける）。"""
    for (x_name, x), (y_name, y) in [(a, b) for a in _BLOCKS.items() for b in _BLOCKS.items()]:
        if mode == "live" and x_name == y_name == "ordered":
            continue  # 届いた途中の行（``1``）でまとまりを切る（stream は行の終わりまで見る）
        text = f"始め。\n\n{x}\n\n{y}\n\n終わり。"
        assert _run(mode, text, clock, step=step) == _run("full", text, clock), (x_name, y_name)


def test_tall_code_is_committed_line_by_line_after_it_flows(clock: _Clock) -> None:
    r, buf = _renderer()
    r.start_turn()
    head, _, _rest = CODE_REPLY.partition("line-040")
    _feed(r, head)
    assert "line-001" not in _plain(buf.getvalue())  # まだ流していない
    clock.now += 1.0
    r.on_assistant_event(TextDelta(text="l"))
    out = _plain(buf.getvalue())
    assert "line-001" in out and "line-030" in out and "以上です" not in out


# ── 最下部の領域 ──


@pytest.mark.parametrize(("mode", "fps"), [("stream", 60.0), ("block", 12.5), ("live", 12.5)])
def test_only_stream_draws_the_bottom_at_60_fps(
    mode: TextMode, fps: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[float] = []

    class _Fake(_Region):
        def __init__(self, console: Console, *, fps: float, on_print: object) -> None:
            seen.append(fps)

        def start(self) -> None:
            pass

    monkeypatch.setattr(render_mod, "LiveRegion", _Fake)
    console = Console(file=io.StringIO(), force_terminal=True, width=WIDTH)
    RichRenderer(console=console, text_mode=mode).start_turn()
    assert seen == [fps]

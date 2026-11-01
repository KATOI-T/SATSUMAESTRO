"""長い生成中も画面が追随する。"""

from __future__ import annotations

import io
import json
import random
import re

from rich.console import Console

from satsumaestro.cli.render import RichRenderer, _fence_opener, _tall_open_fence
from satsumaestro.engine.events import TextDelta, ToolUse, ToolUseDelta

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
WIDTH = 70


def _plain(s: str) -> str:
    return _ANSI.sub("", s)


def _renderer(*, height: int, tty: bool = True) -> tuple[RichRenderer, io.StringIO]:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=tty, width=WIDTH, height=height, color_system=None)
    r = RichRenderer(console=console)
    if tty:
        r._refresh_live = lambda: None  # type: ignore[method-assign]
    return r, buf


def _feed(r: RichRenderer, text: str, *, seed: int = 1) -> None:
    """1〜7 字の乱数幅で流す（行・フェンス・空行の境界が断片の境界と揃わないように）。"""
    rng = random.Random(seed)
    i = 0
    while i < len(text):
        n = rng.randint(1, 7)
        r.on_assistant_event(TextDelta(text=text[i : i + n]))
        i += n


def _scrollback(text: str, *, height: int, seed: int = 1) -> str:
    r, buf = _renderer(height=height)
    r.start_turn()
    _feed(r, text, seed=seed)
    r.end_turn()
    return _plain(buf.getvalue())


def _live_text(r: RichRenderer, *, height: int) -> str:
    out = io.StringIO()
    Console(file=out, force_terminal=False, width=WIDTH, height=height).print(r._live_renderable())
    return out.getvalue()


CODE = "\n".join(f"<!-- line-{i:04d} -->" for i in range(1, 41))


def test_live_body_is_cut_to_the_tail_and_status_stays_on_its_own_row() -> None:
    r, _ = _renderer(height=10)
    r.start_turn()
    _feed(r, "\n".join(f"- item {i}" for i in range(30)))
    rows = _live_text(r, height=10).rstrip("\n").split("\n")
    assert rows[0].strip() == "⋮"
    assert "item 29" in rows[-2] and "item 23" not in "\n".join(rows)
    assert "応答生成中" in rows[-1]
    assert len(rows) == 8
    r.end_turn()


def test_live_body_shorter_than_the_screen_is_not_cut() -> None:
    r, _ = _renderer(height=40)
    r.start_turn()
    _feed(r, "- a\n- b\n- c")
    text = _live_text(r, height=40)
    assert "⋮" not in text and "a" in text and "c" in text
    r.end_turn()


def test_long_fence_reaches_the_scrollback_before_it_closes() -> None:
    r, buf = _renderer(height=10)
    r.start_turn()
    _feed(r, f"```html\n{CODE}\n<!-- partial")
    out = _plain(buf.getvalue())
    assert "line-0030" in out
    assert "line-0040" not in out
    live = _live_text(r, height=10)
    assert "line-0040" in live and "<!-- partial" in live
    r.end_turn()
    assert "line-0040" in _plain(buf.getvalue())


def test_long_fence_live_tail_is_code_not_markdown() -> None:
    r, _ = _renderer(height=10)
    r.start_turn()
    _feed(r, "```python\n" + "\n".join(f"x{i} = {i}" for i in range(20)) + "\n# comment line")
    live = _live_text(r, height=10)
    assert "# comment line" in live
    r.end_turn()


_SAMPLES = {
    "段落＋空行＋フェンス＋空行＋段落": f"Here is the file:\n\n```html\n{CODE}\n```\n\nDone.\n",
    "段落の直後にフェンス": f"Here is the file:\n```html\n{CODE}\n```\nDone.\n",
    "フェンスだけ（閉じに改行無し）": f"```html\n{CODE}\n```",
    "フェンス内の末尾の空行と行末の空白": f"```html\n{CODE}   \n\n\n```\n",
    "リストの後のフェンス": f"- a\n- b\n\n```html\n{CODE}\n```\n",
    "閉じフェンスの直後に本文": f"```html\n{CODE}\n```\nNext para\n\nMore.\n",
    "2 つの長いフェンス": f"```html\n{CODE}\n```\n\n```css\n{CODE}\n```\n",
    "閉じていないフェンスのまま終わる": f"```html\n{CODE}\n",
}


def test_streamed_scrollback_equals_unstreamed_scrollback() -> None:
    """流したとき（高さ 10）と流さなかったとき（高さ 1000）で、最終のスクロールバックが同じ。"""
    for name, text in _SAMPLES.items():
        for seed in (1, 2, 3):
            big = _scrollback(text, height=1000, seed=seed)
            small = _scrollback(text, height=10, seed=seed)
            assert "line-0001" in small and "line-0040" in small, name
            assert small == big, f"{name} (seed {seed})"


def test_short_fence_is_still_rendered_as_markdown() -> None:
    text = "```python\nprint(1)\nprint(2)\n```\n"
    assert _scrollback(text, height=10) == _scrollback(text, height=1000)
    r, _ = _renderer(height=10)
    r.start_turn()
    _feed(r, text[:-5])
    assert r._fence is None
    r.end_turn()


def test_indented_fence_in_a_list_is_not_streamed() -> None:
    body = "\n".join(f"  x{i} = {i}" for i in range(40))
    text = f"- item\n\n  ```python\n{body}\n  ```\n"
    r, buf = _renderer(height=10)
    r.start_turn()
    _feed(r, text)
    assert r._fence is None
    assert "x30" not in _plain(buf.getvalue())
    r.end_turn()
    assert _scrollback(text, height=10) == _scrollback(text, height=1000)


def test_segment_end_closes_a_streaming_fence() -> None:
    r, buf = _renderer(height=10)
    r.start_turn()
    _feed(r, f"```html\n{CODE}\n<!-- tail -->")
    assert r._fence is not None
    r.on_stream_aborted()
    assert r._fence is None
    out = _plain(buf.getvalue())
    assert "<!-- tail -->" in out and "応答が途中で切れました" in out
    assert out.index("<!-- tail -->") < out.index("応答が途中で切れました")
    r.end_turn()


def test_partial_closing_fence_at_segment_end_is_a_closer_not_code() -> None:
    r, buf = _renderer(height=10)
    r.start_turn()
    _feed(r, f"```html\n{CODE}\n```")
    r.end_turn()
    out = _plain(buf.getvalue())
    assert "```" not in out and "line-0040" in out


def test_open_fence_helpers() -> None:
    assert _fence_opener("```python\nx = 1\n") == (0, 10, "python")
    assert _fence_opener("text\n```\nx\n") == (5, 9, "text")
    assert _fence_opener("```python") is None
    assert _fence_opener("```python\nx\n```\n") is None
    assert _fence_opener("```python\nx\n```python\ny\n") == (0, 10, "python")
    assert _fence_opener("```\na\n```\n\n```css\nb\n") == (11, 18, "css")
    assert _fence_opener("  ```python\nx\n") is None
    assert _tall_open_fence("```\n" + "x\n" * 3, 3) == (0, 4, "text")
    assert _tall_open_fence("```\n" + "x\n" * 2, 3) is None


def _delta(r: RichRenderer, index: int, args: str, *, step: int = 23) -> None:
    for i in range(0, len(args), step):
        r.on_assistant_event(
            ToolUseDelta(index=index, name="write_file", args_delta=args[i : i + step])
        )


def _content(n: int) -> str:
    return "\n".join(f"/*f{n}-l{j:03d}*/" for j in range(1, 41)) + "\n"


def test_panel_follows_the_latest_call_and_summarizes_the_finished_ones() -> None:
    r, _ = _renderer(height=40)
    r.start_turn()
    _delta(r, 0, json.dumps({"path": "f1.css", "content": _content(1)}))
    live = _live_text(r, height=40)
    assert "write_file — 生成中 f1.css（40 行）" in live
    assert "/*f1-l040*/" in live
    _delta(r, 1, json.dumps({"path": "f2.css", "content": _content(2)}))
    _delta(r, 2, json.dumps({"path": "f3.css", "content": _content(3)}))
    live = _live_text(r, height=40)
    assert "write_file — 生成中 3 件目 f3.css（40 行）" in live
    assert "/*f3-l040*/" in live and "/*f1-l040*/" not in live
    assert "✓ f1.css（40 行）" in live and "✓ f2.css（40 行）" in live
    r.on_assistant_event(ToolUse(id="c1", name="write_file", input={}))
    assert "— 生成中" not in _live_text(r, height=40)
    r.end_turn()


def test_panel_shows_the_finished_calls_while_the_next_path_is_still_arriving() -> None:
    r, _ = _renderer(height=40)
    r.start_turn()
    _delta(r, 0, json.dumps({"path": "f1.css", "content": _content(1)}))
    r.on_assistant_event(ToolUseDelta(index=1, name="write_file", args_delta='{"pa'))
    live = _live_text(r, height=40)
    assert "✓ f1.css（40 行）" in live and "2 件目" in live


def test_panel_line_count_grows_with_the_content() -> None:
    r, _ = _renderer(height=40)
    r.start_turn()
    r.on_assistant_event(
        ToolUseDelta(index=0, name="edit_file", args_delta='{"path": "a.py", "new": "x\\ny\\nz')
    )
    assert "edit_file — 生成中 a.py（3 行）" in _live_text(r, height=40)


def test_non_tty_output_is_unchanged_for_long_fences() -> None:
    r, buf = _renderer(height=10, tty=False)
    r.start_turn()
    _feed(r, f"Intro\n\n```html\n{CODE}\n```\n\nDone.\n")
    r.end_turn()
    out = buf.getvalue()
    expected = io.StringIO()
    from rich.markdown import Markdown

    Console(file=expected, force_terminal=False, width=WIDTH, height=10, color_system=None).print(
        Markdown(f"Intro\n\n```html\n{CODE}\n```\n\nDone.\n")
    )
    assert expected.getvalue() in out

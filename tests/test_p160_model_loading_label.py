"""モデルロード待ちの状態行表示のテスト。"""

from __future__ import annotations

import io

import pytest
from rich.console import Console

import satsumaestro.cli.render as render_mod
from satsumaestro.cli.app import _unloaded_model
from satsumaestro.cli.render import RichRenderer
from satsumaestro.engine.context_fit import ContextFitResult
from satsumaestro.engine.events import TextDelta
from satsumaestro.providers.detect import ModelContextInfo
from satsumaestro.session.models import ToolResultBlock

_MODEL = "gpt-oss:120b"


def _renderer(loading_model: str | None) -> RichRenderer:
    console = Console(file=io.StringIO(), force_terminal=True, width=100)
    return RichRenderer(console=console, loading_model=loading_model)


def _status(r: RichRenderer) -> str:
    out = io.StringIO()
    Console(file=out, force_terminal=False, width=100).print(r._live_renderable())
    return out.getvalue()


def test_first_waiting_shows_loading_label() -> None:
    r = _renderer(_MODEL)
    r.start_turn()
    assert f"{_MODEL} をロード中…" in _status(r)
    assert "考え中" not in _status(r)


def test_first_event_returns_to_normal_labels() -> None:
    r = _renderer(_MODEL)
    r.start_turn()
    r.on_assistant_event(TextDelta(text="はい"))
    assert "ロード中" not in _status(r)
    r.on_tool_result(ToolResultBlock(tool_use_id="t1", tool_name="read_file", output="x"))
    text = _status(r)
    assert "考え中…" in text and "ロード中" not in text


def test_loading_label_not_carried_to_next_turn() -> None:
    r = _renderer(_MODEL)
    r.start_turn()
    r.end_turn()
    r.start_turn()
    text = _status(r)
    assert "考え中…" in text and "ロード中" not in text


def test_phase_label_takes_precedence() -> None:
    r = _renderer(_MODEL)
    r.start_turn()
    r.on_phase("会話を圧縮中…")
    assert "会話を圧縮中…" in _status(r)
    assert "ロード中" not in _status(r)


def test_no_loading_label_when_loaded() -> None:
    r = _renderer(None)
    r.start_turn()
    assert "考え中…" in _status(r)


def test_stalled_load_switches_label(monkeypatch: pytest.MonkeyPatch) -> None:
    r = _renderer(_MODEL)
    r.start_turn()
    base = r._phase_since
    monkeypatch.setattr(render_mod.time, "monotonic", lambda: base + 130.0)
    assert f"{_MODEL} をロード中…" in _status(r)
    monkeypatch.setattr(render_mod.time, "monotonic", lambda: base + 181.0)
    text = _status(r)
    assert "ロードが完了しません" in text and "ollama ps" in text


_BAR = "━"


def test_pulse_bar_shown_while_loading() -> None:
    r = _renderer(_MODEL)
    r.start_turn()
    assert _BAR in _status(r)
    r.on_assistant_event(TextDelta(text="はい"))
    assert _BAR not in _status(r)


def test_no_pulse_bar_when_loaded() -> None:
    r = _renderer(None)
    r.start_turn()
    assert _BAR not in _status(r)


def test_pulse_bar_hidden_after_stall(monkeypatch: pytest.MonkeyPatch) -> None:
    r = _renderer(_MODEL)
    r.start_turn()
    base = r._phase_since
    monkeypatch.setattr(render_mod.time, "monotonic", lambda: base + 181.0)
    assert _BAR not in _status(r)


def _fit(info: ModelContextInfo) -> ContextFitResult:
    return ContextFitResult(
        model=_MODEL, info=info, original_threshold=120_000, effective_threshold=120_000
    )


def test_unloaded_model_requires_confirmed_ps() -> None:
    assert _unloaded_model(_fit(ModelContextInfo(ps_ok=True, loaded=False))) == _MODEL
    assert _unloaded_model(_fit(ModelContextInfo(ps_ok=True, loaded=True))) is None
    assert _unloaded_model(_fit(ModelContextInfo(ps_ok=False, loaded=False))) is None
    assert _unloaded_model(None) is None

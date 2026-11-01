"""REPL がターンの失敗でセッションごと落ちないことのテスト。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from rich.console import Console

from satsumaestro.cli import repl as repl_mod
from satsumaestro.engine.turn import TurnSummary
from satsumaestro.session.models import TokenUsage
from satsumaestro.util.errors import MaxIterationsError, ProviderError, SatsumaError


class _FakeSession:
    """``PromptSession`` の代役。"""

    def __init__(self, lines: list[str]) -> None:
        self._lines = list(lines)

    async def prompt_async(self, *_: Any, **__: Any) -> str:
        if not self._lines:
            raise EOFError
        return self._lines.pop(0)


class _FakeRuntime:
    """``run_turn`` が指定の結果（例外 or サマリ）を順に返す最小のランタイム。"""

    def __init__(self, outcomes: list[Any]) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[str] = []
        self.session = _FakeSessionState()
        self.orchestrator = None
        self.cwd = Path("/work/project")

    def take_window_notes(self) -> list[str]:
        return []

    async def run_turn(self, line: str, *_: Any, **__: Any) -> TurnSummary:
        self.calls.append(line)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def approve_plan_from_text(self, _: str) -> bool:
        return False


class _FakeSessionState:
    mode = None
    model = "mock-model"
    plan_steps: list[str] = []
    plan_step_index = 0


class _FakeRender:
    def __init__(self) -> None:
        self.turns = 0
        self.ends = 0

    def start_turn(self) -> None:
        self.turns += 1

    def end_turn(self, *, delegated: object = None) -> None:
        self.ends += 1
        self.delegated = delegated


def _read_timeout() -> ProviderError:
    error = ProviderError("OpenAI 互換 API 呼び出しに失敗: ", retryable=True)
    error.__cause__ = httpx.ReadTimeout("timed out", request=httpx.Request("POST", "http://x"))
    return error


async def _run(
    monkeypatch: pytest.MonkeyPatch, lines: list[str], outcomes: list[Any]
) -> tuple[_FakeRuntime, str, _FakeRender]:
    monkeypatch.setattr(repl_mod, "PromptSession", lambda **_: _FakeSession(lines))
    monkeypatch.setattr(repl_mod, "SlashDispatcher", lambda *a, **k: _NoSlash())
    monkeypatch.setattr(repl_mod, "mode_prompt_fragments", lambda _: "> ")
    rt = _FakeRuntime(outcomes)
    console = Console(record=True, width=120)
    render = _FakeRender()
    await repl_mod.run_repl(rt, None, render, console)  # type: ignore[arg-type]
    return rt, console.export_text(), render


class _NoSlash:
    def is_command(self, _: str) -> bool:
        return False


def _summary(text: str) -> TurnSummary:
    return TurnSummary(final_text=text, stop_reason="end_turn", iterations=1, usage=TokenUsage())


async def test_provider_error_keeps_the_repl_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    rt, out, render = await _run(
        monkeypatch, ["大きい HTML を作って", "続けて"], [_read_timeout(), _summary("できました")]
    )
    assert rt.calls == ["大きい HTML を作って", "続けて"]
    assert "⚠ ターンを最後まで進められませんでした: モデルが応答しませんでした" in out
    assert "「続けて」で続きから再開できます" in out
    assert render.ends == 2


async def test_max_iterations_gets_the_same_reason_and_next_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, out, _ = await _run(monkeypatch, ["hi"], [MaxIterationsError(50, last_text="")])
    assert "⚠ ターンを最後まで進められませんでした: 反復の上限（50 回）に達しました" in out
    assert "reached max_iterations" not in out


async def test_provider_error_is_a_satsuma_error() -> None:
    assert issubclass(ProviderError, SatsumaError)
    assert not issubclass(ProviderError, MaxIterationsError)


async def test_unexpected_error_still_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ZeroDivisionError):
        await _run(monkeypatch, ["hi"], [ZeroDivisionError("bug")])


def test_repl_module_path_is_stable() -> None:
    assert hasattr(repl_mod, "PromptSession")
    assert Path(repl_mod.__file__ or "").name == "repl.py"

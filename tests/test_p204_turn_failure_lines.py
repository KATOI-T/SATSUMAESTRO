"""ターンが例外で終わったとき、REPL は落ちた理由と次の一手を日本語で出す。"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx
import pytest
from rich.console import Console

from satsumaestro.cli import repl as repl_mod
from satsumaestro.cli.errors_ui import (
    TurnFailure,
    failure_head,
    resume_hint,
    turn_failure,
)
from satsumaestro.util.errors import (
    MaxIterationsError,
    PlanExitLoopError,
    ProviderError,
    RepetitionLoopError,
    RetriesExhaustedError,
    SatsumaError,
    UnknownToolLoopError,
)
from tests.test_p163_timeout_recovery import _done, _runtime, _timeout
from tests.test_repl_turn_failure import _FakeRuntime, _FakeSession, _NoSlash


def _read_timeout(*, seconds: float = 0.0, recoveries: int = 0) -> ProviderError:
    error = ProviderError("OpenAI 互換 API 呼び出しに失敗: ", retryable=True)
    error.__cause__ = httpx.ReadTimeout("timed out", request=httpx.Request("POST", "http://x"))
    error.timeout_s = seconds
    error.timeout_recoveries = recoveries
    return error


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (MaxIterationsError(50), TurnFailure("反復の上限（50 回）に達しました")),
        (
            MaxIterationsError(8, step=(2, 3)),
            TurnFailure("反復の上限（8 回）に達しました（段 2/3 の中）"),
        ),
        (
            RetriesExhaustedError(2, 25.4, last_error="error parsing tool call: raw=…"),
            TurnFailure(
                "モデルの出力（ツール呼び出しの引数）が繰り返し壊れました", broken_output=True
            ),
        ),
        (
            _read_timeout(seconds=120.0, recoveries=1),
            TurnFailure("モデルが 120 秒応答しませんでした（分けて出し直す依頼の後も）"),
        ),
        (_read_timeout(), TurnFailure("モデルが応答しませんでした")),
        (
            ProviderError("OpenAI 互換 API 呼び出しに失敗: Connection error."),
            TurnFailure(
                "モデルとの通信に失敗しました（OpenAI 互換 API 呼び出しに失敗: Connection error.）"
            ),
        ),
        (
            PlanExitLoopError(3),
            TurnFailure("plan モードではないのに exit_plan_mode を 3 回続けて呼びました"),
        ),
        (
            UnknownToolLoopError("run_command", 3),
            TurnFailure("存在しないツール名 run_command を 3 回続けて呼びました"),
        ),
        (
            RepetitionLoopError(30, "同じ段落", recovered_before=True),
            TurnFailure("同じ内容の繰り返しが止まりませんでした（同じ段落が 30 回）"),
        ),
        (SatsumaError("設定が読めません"), TurnFailure("設定が読めません")),
    ],
)
def test_each_failure_gets_a_reason_and_a_next_step(
    exc: SatsumaError, expected: TurnFailure
) -> None:
    assert turn_failure(exc) == expected


def test_the_first_line_and_the_hint() -> None:
    failure = turn_failure(MaxIterationsError(50))
    assert failure.headline == f"{failure_head()}: 反復の上限（50 回）に達しました"
    assert failure.next_step == resume_hint()
    assert resume_hint().startswith("「続けて」で続きから再開できます。")


def test_the_original_words_go_to_the_log(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="satsumaestro.cli.errors_ui"):
        turn_failure(MaxIterationsError(50))
    assert "reached max_iterations=50 without end_turn" in caplog.text


async def test_a_failed_read_timeout_carries_the_seconds(tmp_path: Path) -> None:
    rt = _runtime([_timeout(), _timeout("again"), _done()], tmp_path)
    with pytest.raises(ProviderError) as excinfo:
        await rt.run_turn("300 行の HTML を作って")
    assert excinfo.value.timeout_s == rt.config.read_timeout_s()
    assert excinfo.value.timeout_recoveries == 1


async def _repl(
    monkeypatch: pytest.MonkeyPatch, outcome: BaseException, *, steps: list[str]
) -> str:
    monkeypatch.setattr(repl_mod, "PromptSession", lambda **_: _FakeSession(["作って"]))
    monkeypatch.setattr(repl_mod, "SlashDispatcher", lambda *a, **k: _NoSlash())
    monkeypatch.setattr(repl_mod, "mode_prompt_fragments", lambda _: "> ")
    rt = _FakeRuntime([outcome])
    rt.session.plan_steps = steps  # type: ignore[misc]
    rt.session.plan_step_index = 2 if steps else 0  # type: ignore[misc]
    console = Console(record=True, width=200)
    render = type("R", (), {"start_turn": lambda s: None, "end_turn": lambda s, **_: None})()
    await repl_mod.run_repl(rt, None, render, console)  # type: ignore[arg-type]
    return console.export_text()


async def test_repl_shows_the_reason_and_how_to_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    out = await _repl(monkeypatch, MaxIterationsError(50), steps=[])
    assert f"{failure_head()}: 反復の上限（50 回）に達しました" in out
    assert resume_hint() in out
    assert "reached max_iterations" not in out


async def test_inside_a_plan_the_step_line_is_the_next_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = await _repl(monkeypatch, MaxIterationsError(8, step=(2, 3)), steps=["a", "b", "c"])
    assert f"{failure_head()}: 反復の上限（8 回）に達しました（段 2/3 の中）" in out
    assert "計画の段 2/3 が未完です。「続けて」で再開、/clear で破棄" in out
    assert resume_hint() not in out

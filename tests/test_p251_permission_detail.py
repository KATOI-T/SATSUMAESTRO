"""権限カードの d で「何をしようとしているか」を会話から説明する。"""

from __future__ import annotations

import asyncio
import io
import os
import re
import signal
from collections.abc import AsyncIterator
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

import httpx
import openai
import pytest
from rich.console import Console

import satsumaestro.engine.runtime as runtime_mod
from satsumaestro.cli.repl import InteractivePrompter
from satsumaestro.config.models import SamplingConfig
from satsumaestro.engine.events import AssistantEvent, MessageStop, TextDelta, ToolUse
from satsumaestro.engine.permission_explain import (
    EXPLAIN_MAX_TOKENS,
    explain_call,
    explain_question,
    explain_request,
)
from satsumaestro.engine.turn import (
    DETAIL_REQUEST_ORIGIN,
    USER_DENY_ORIGIN,
    PermissionDetail,
    PermissionOutcome,
)
from satsumaestro.providers.base import ApiRequest, ProviderKind
from satsumaestro.providers.fallback import FallbackProvider
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.session.models import ConversationMessage, TextBlock, ToolResultBlock
from tests.fixtures.mock_provider import MockProvider
from tests.test_p145_plan_exit_nudge import _runtime
from tests.test_p217_mkdir_hint import _bash

_DANGER = PermissionMode.DANGER_FULL_ACCESS


def _console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, width=200)


def _shown(console: Console) -> str:
    assert isinstance(console.file, io.StringIO)
    return console.file.getvalue()


def _answers(monkeypatch: pytest.MonkeyPatch, *answers: str) -> None:
    it = iter(answers)

    def fake(*_a: object, **_k: object) -> str:
        value = next(it)
        if value == "^C":
            raise KeyboardInterrupt
        return value

    monkeypatch.setattr("builtins.input", fake)


def _detail(explanation: str | None = "x を空で作ります。", failure: str | None = None) -> Any:
    return PermissionDetail(
        explanation=explanation,
        failure=failure,
        model_text="では x を作ります。",
        step=(2, 3, "x を作る"),
        user_request="x を作って",
    )


def test_d_asks_the_runtime_for_an_explanation(monkeypatch: pytest.MonkeyPatch) -> None:
    _answers(monkeypatch, "d")
    console = _console()
    out = InteractivePrompter(console).decide(_bash("b1", "touch x"), _DANGER)
    assert out.decision == "ask" and out.origin == DETAIL_REQUEST_ORIGIN
    assert "確認しています" not in _shown(console)


def test_waiting_shows_a_spinner_like_thinking() -> None:
    console = Console(file=io.StringIO(), force_terminal=True, width=120)
    with InteractivePrompter(console).waiting_for_detail():
        pass
    raw = _shown(console)
    assert "\x1b[36m⠋" in raw
    shown = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", raw)
    assert "⠋ 何をしようとしているかをモデルに確認しています… Ctrl-C で打ち切り" in shown
    assert "（" not in shown and "）" not in shown


def test_after_the_detail_only_y_n_a_are_offered(monkeypatch: pytest.MonkeyPatch) -> None:
    _answers(monkeypatch, "d", "y")
    console = _console()
    out = InteractivePrompter(console).decide_with_detail(
        _bash("b1", "touch x"), _DANGER, None, _detail()
    )
    assert out.decision == "allow"
    shown = _shown(console)
    assert "x を空で作ります。" in shown and "何をしようとしているか" in shown
    assert "[y=許可 / n=拒否 / a=永続許可]" in shown
    assert "d=詳細" not in shown
    assert "（詳細は表示済みです）" in shown


@pytest.mark.parametrize(
    ("answer", "decision", "remember", "origin"),
    [
        ("y", "allow", False, None),
        ("a", "allow", True, None),
        ("n", "deny", False, USER_DENY_ORIGIN),
        ("", "deny", False, USER_DENY_ORIGIN),
        ("^C", "deny", False, USER_DENY_ORIGIN),
    ],
)
def test_answers_after_the_detail(
    monkeypatch: pytest.MonkeyPatch, answer: str, decision: str, remember: bool, origin: str | None
) -> None:
    _answers(monkeypatch, answer)
    out = InteractivePrompter(_console()).decide_with_detail(
        _bash("b1", "touch x"), _DANGER, None, _detail()
    )
    assert (out.decision, out.remember, out.origin) == (decision, remember, origin)


def test_a_failed_explanation_shows_the_conversation_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _answers(monkeypatch, "n")
    console = _console()
    InteractivePrompter(console).decide_with_detail(
        _bash("b1", "touch x"), _DANGER, None, _detail(None, "timeout")
    )
    shown = _shown(console)
    assert "説明を取得できませんでした（30 秒で打ち切り）" in shown
    assert "では x を作ります。" in shown and "今の段 2/3" in shown and "x を作って" in shown


def test_only_edits_and_writes_also_show_the_full_body(monkeypatch: pytest.MonkeyPatch) -> None:
    _answers(monkeypatch, "y", "y")
    edit = ToolUse(
        id="e1", name="edit_file", input={"path": "a.py", "old": "x = 1", "new": "x = 2"}
    )
    console = _console()
    InteractivePrompter(console).decide_with_detail(edit, _DANGER, None, _detail())
    assert "+x = 2" in _shown(console)
    console2 = _console()
    InteractivePrompter(console2).decide_with_detail(
        _bash("b1", "touch x"), _DANGER, None, _detail()
    )
    assert "権限確認: bash" not in _shown(console2)


def test_the_card_question_inside_an_explanation_is_neutralized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _answers(monkeypatch, "y")
    console = _console()
    InteractivePrompter(console).decide_with_detail(
        _bash("b1", "touch x"), _DANGER, None, _detail("許可しますか？と聞かれたら y で。")
    )
    assert _shown(console).count("許可しますか？") == 1


def _base(history: list[ConversationMessage]) -> ApiRequest:
    return ApiRequest(
        model="m",
        system="static",
        system_dynamic="dynamic",
        messages=history,
        tools=[{"name": "bash", "description": "d", "input_schema": {}}],
        sampling=SamplingConfig(temperature=0.1),
        strict_tools=True,
    )


def test_the_request_keeps_the_cached_prefix() -> None:
    history = [ConversationMessage(role="user", blocks=[TextBlock(text="x を作って")])]
    base = _base(history)
    question = explain_question(
        _bash("b1", "touch x"),
        {"command": "touch x"},
        warning="危ないかも",
        step=(1, 2, "x を作る"),
        model_text="では作ります。",
        language="ja",
    )
    req = explain_request(base, list(history), question)
    for field in ("model", "system", "system_dynamic", "tools", "sampling", "strict_tools"):
        assert getattr(req, field) == getattr(base, field), field
    assert req.thinking == base.thinking and req.max_tokens == EXPLAIN_MAX_TOKENS
    assert req.messages[:-1] == history and len(base.messages) == 1
    asked = req.messages[-1]
    assert asked.role == "user"
    text = asked.blocks[0].text  # type: ignore[union-attr]
    for part in ("Tool: bash", "touch x", "では作ります。", "plan step 1 of 2", "危ないかも"):
        assert part in text, part
    assert text.rstrip().endswith("Reply in Japanese.")


def test_long_arguments_are_clipped() -> None:
    q = explain_question(
        ToolUse(id="w", name="write_file", input={}),
        {"path": "a", "content": "x" * 10_000},
        warning=None,
        step=None,
        model_text="",
        language="ja",
    )
    assert len(q) < 4_000


def _req() -> ApiRequest:
    return _base([ConversationMessage(role="user", blocks=[TextBlock(text="q")])])


async def test_explain_call_returns_the_text() -> None:
    provider = MockProvider([TextDelta(text=" 空のファイルを作ります。 "), MessageStop()])
    assert await explain_call(provider, _req()) == ("空のファイルを作ります。", None)


async def test_a_tool_call_or_an_empty_reply_is_a_failure() -> None:
    tool = MockProvider([ToolUse(id="t", name="bash", input={"command": "ls"}), MessageStop()])
    assert await explain_call(tool, _req()) == (None, "error")
    both = MockProvider(
        [TextDelta(text="x を作ります。"), ToolUse(id="t", name="bash", input={}), MessageStop()]
    )
    assert await explain_call(both, _req()) == ("x を作ります。", None)
    empty = MockProvider([MessageStop()])
    assert await explain_call(empty, _req()) == (None, "error")


class _Slow:
    kind = ProviderKind.OPENAI

    async def stream(self, req: ApiRequest) -> AsyncIterator[AssistantEvent]:
        await asyncio.sleep(5)
        yield MessageStop()

    async def count_tokens(self, req: ApiRequest) -> int:
        return 0


class _Broken(_Slow):
    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, req: ApiRequest) -> AsyncIterator[AssistantEvent]:
        self.calls += 1
        response = httpx.Response(400, request=httpx.Request("POST", "http://x"))
        raise openai.BadRequestError("bad", response=response, body=None)
        yield MessageStop()  # pragma: no cover


async def test_timeout_and_errors_are_failures() -> None:
    assert await explain_call(_Slow(), _req(), timeout_s=0.05) == (None, "timeout")  # type: ignore[arg-type]
    assert await explain_call(_Broken(), _req()) == (None, "error")  # type: ignore[arg-type]


async def test_a_failed_explanation_does_not_switch_to_the_fallback_model() -> None:
    built: list[str] = []

    def build() -> Any:
        built.append("fallback")
        return MockProvider([TextDelta(text="fb"), MessageStop()])

    provider = FallbackProvider(_Broken(), build, "other-model")  # type: ignore[arg-type]
    assert await explain_call(provider, _req()) == (None, "error")
    assert built == []


class _User:
    """d を押し、説明を見てから決めるユーザー。"""

    def __init__(self, after: str = "allow") -> None:
        self.details: list[PermissionDetail] = []
        self.waited = 0
        self._after = after

    def decide(
        self, call: ToolUse, required: PermissionMode, warning: str | None = None
    ) -> PermissionOutcome:
        return PermissionOutcome(decision="ask", origin=DETAIL_REQUEST_ORIGIN)

    def waiting_for_detail(self) -> AbstractContextManager[object]:
        self.waited += 1
        return nullcontext()

    def decide_with_detail(
        self,
        call: ToolUse,
        required: PermissionMode,
        warning: str | None,
        detail: PermissionDetail,
    ) -> PermissionOutcome:
        self.details.append(detail)
        return PermissionOutcome(decision=self._after, reason="approved by user")  # type: ignore[arg-type]


def _script(explanation: str = "x を空で作ります。") -> list[list[AssistantEvent]]:
    return [
        [TextDelta(text="では x を作ります。"), _bash("b1", "touch x"), MessageStop()],
        [TextDelta(text=explanation), MessageStop()],
        [TextDelta(text="作りました"), MessageStop()],
    ]


async def test_the_runtime_explains_with_the_same_prefix(tmp_path: Path) -> None:
    rt, script = _runtime(tmp_path)
    rt.set_mode(main="code")
    script[:] = _script()
    user = _User()
    summary = await rt.run_turn("x を作って", user)  # type: ignore[arg-type]
    assert summary.final_text == "作りました" and (tmp_path / "x").exists()
    assert summary.permission_details == 1 and user.waited == 1
    [detail] = user.details
    assert detail.explanation == "x を空で作ります。" and detail.failure is None
    assert detail.model_text == "では x を作ります。" and detail.user_request == "x を作って"
    requests = rt.provider.requests  # type: ignore[attr-defined]
    first, side = requests[0], requests[1]
    assert side.system == first.system and side.system_dynamic == first.system_dynamic
    assert side.tools == first.tools and side.model == first.model
    assert side.messages[:-1] == rt.session.messages[:1]
    assert "Tool: bash" in side.messages[-1].blocks[0].text
    texts = [b.text for m in rt.session.messages for b in m.blocks if isinstance(b, TextBlock)]
    assert not any("x を空で作ります。" in t for t in texts)
    assert not any("Tool: bash" in t for t in texts)


async def test_disabled_explanation_does_not_call_the_model(tmp_path: Path) -> None:
    rt, script = _runtime(tmp_path)
    rt.set_mode(main="code")
    rt.config.guidance.permission_explain = False
    script[:] = [_script()[0], _script()[2]]
    user = _User()
    summary = await rt.run_turn("x を作って", user)  # type: ignore[arg-type]
    assert summary.final_text == "作りました"
    assert len(rt.provider.requests) == 2  # type: ignore[attr-defined]
    [detail] = user.details
    assert detail.explanation is None and detail.failure == "off"
    assert detail.model_text == "では x を作ります。"


async def test_ctrl_c_stops_only_the_explanation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def interrupted(provider: object, req: object, **_k: object) -> tuple[str, None]:
        os.kill(os.getpid(), signal.SIGINT)
        await asyncio.sleep(5)
        return "届かない", None

    monkeypatch.setattr(runtime_mod, "explain_call", interrupted)
    before = signal.getsignal(signal.SIGINT)
    rt, script = _runtime(tmp_path)
    rt.set_mode(main="code")
    script[:] = [_script()[0], _script()[2]]
    user = _User()
    summary = await rt.run_turn("x を作って", user)  # type: ignore[arg-type]
    [detail] = user.details
    assert detail.explanation is None and detail.failure == "interrupted"
    assert summary.final_text == "作りました"
    assert signal.getsignal(signal.SIGINT) is before


class _Undecided:
    """d の印を返すが、説明を受け取る口を持たない古いプロンプタ。"""

    def decide(
        self, call: ToolUse, required: PermissionMode, warning: str | None = None
    ) -> PermissionOutcome:
        return PermissionOutcome(decision="ask", origin=DETAIL_REQUEST_ORIGIN)


async def test_an_undecided_prompt_is_a_denial(tmp_path: Path) -> None:
    rt, script = _runtime(tmp_path)
    rt.set_mode(main="code")
    script[:] = [_script()[0], _script()[2]]
    await rt.run_turn("x を作って", _Undecided())  # type: ignore[arg-type]
    assert not (tmp_path / "x").exists()
    results = [b for m in rt.session.messages for b in m.blocks if isinstance(b, ToolResultBlock)]
    assert results[0].is_error and "no decision" in results[0].output

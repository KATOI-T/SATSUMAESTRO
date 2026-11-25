"""ターン中の Ctrl+C で、ターンだけを止める（REPL は続く）。"""

from __future__ import annotations

import asyncio
import io
import signal
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from satsumaestro.cli import repl as repl_mod
from satsumaestro.cli.interrupts import ctrl_c_raises, run_cancellable
from satsumaestro.cli.repl import InteractivePrompter
from satsumaestro.config.models import HookConfig, RuntimeConfig
from satsumaestro.engine.events import AssistantEvent, MessageStop, TextDelta, ToolUse
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.providers.base import ApiRequest, ProviderKind
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.models import ToolResultBlock
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from tests.fixtures.mock_provider import SequencedMockProvider


class _Outer:
    """ターンの外の SIGINT の受け手（``asyncio.run`` の受け手の代役）。届いた回数を数える。"""

    def __init__(self) -> None:
        self.hits = 0

    def __call__(self, _signum: int, _frame: Any) -> None:
        self.hits += 1


@pytest.fixture
def outer() -> Iterator[_Outer]:
    handler = _Outer()
    previous = signal.signal(signal.SIGINT, handler)
    try:
        yield handler
    finally:
        signal.signal(signal.SIGINT, previous)


async def test_ctrl_c_cancels_only_the_turn(outer: _Outer) -> None:
    started = asyncio.Event()
    cancelled = False

    async def turn() -> None:
        nonlocal cancelled
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled = True
            raise

    async def press() -> None:
        await started.wait()
        signal.raise_signal(signal.SIGINT)

    presser = asyncio.create_task(press())
    with pytest.raises(KeyboardInterrupt):
        await run_cancellable(turn())
    await presser
    assert cancelled
    assert outer.hits == 0  # ターンの外の受け手（REPL ごと止める側）には届かない
    assert signal.getsignal(signal.SIGINT) is outer


async def test_cancelling_the_caller_is_not_taken_as_ctrl_c(outer: _Outer) -> None:
    inner_cancelled = False

    async def turn() -> None:
        nonlocal inner_cancelled
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            inner_cancelled = True
            raise

    caller = asyncio.create_task(run_cancellable(turn()))
    await asyncio.sleep(0.01)
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert caller.cancelled() and inner_cancelled
    assert signal.getsignal(signal.SIGINT) is outer


async def test_the_result_comes_back_and_the_handler_is_restored(outer: _Outer) -> None:
    async def turn() -> int:
        assert signal.getsignal(signal.SIGINT) is not outer
        return 42

    assert await run_cancellable(turn()) == 42
    assert signal.getsignal(signal.SIGINT) is outer


def test_ctrl_c_in_a_line_input_raises_there(outer: _Outer) -> None:
    with pytest.raises(KeyboardInterrupt), ctrl_c_raises():
        signal.raise_signal(signal.SIGINT)
    assert outer.hits == 0
    assert signal.getsignal(signal.SIGINT) is outer


async def test_ctrl_c_at_a_card_denies_and_the_turn_goes_on(outer: _Outer) -> None:
    """カードの入力中の Ctrl+C は拒否の答えになり、ターンは取り消されない。"""
    console = Console(file=io.StringIO(), force_terminal=False)

    def pressing_ctrl_c(*_args: object, **_kwargs: object) -> str:
        signal.raise_signal(signal.SIGINT)  # 入力を待っている間に押された
        return "y"

    console.input = pressing_ctrl_c  # type: ignore[method-assign]
    prompter = InteractivePrompter(console)

    async def turn() -> str:
        answer = prompter._ask("Allow this? ")
        await asyncio.sleep(0.05)  # 取り消しが積まれていれば、ここで届く
        return answer

    assert await run_cancellable(turn()) == repl_mod._INTERRUPTED
    assert outer.hits == 0


def _runtime(provider: object, tmp_path: Path) -> ConversationRuntime:
    return ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=provider,  # type: ignore[arg-type]
        tools=default_registry(),
        policy=default_policy(PermissionMode.ALLOW),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=RuntimeConfig(),
    )


async def _cancel_when(task: asyncio.Task[Any], ready: Any) -> None:
    for _ in range(500):
        if ready():
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("the turn did not reach the expected point")
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_cancelled_turn_closes_the_running_call(tmp_path: Path) -> None:
    provider = SequencedMockProvider(
        [
            [ToolUse(id="t1", name="bash", input={"command": "sleep 30"}), MessageStop()],
            [TextDelta(text="never sent"), MessageStop()],
        ]
    )
    rt = _runtime(provider, tmp_path)
    task = asyncio.create_task(rt.run_turn("run it"))
    messages = rt.session.messages
    await _cancel_when(task, lambda: messages and messages[-1].role == "assistant")

    last = messages[-1]
    assert last.role == "tool"
    [block] = last.blocks
    assert isinstance(block, ToolResultBlock)
    assert block.tool_use_id == "t1" and block.is_error
    assert block.output.startswith("interrupted:")
    store = JSONLSessionStore()
    restored = store.load(store.path_for(tmp_path, rt.session.session_id))
    assert [m.role for m in restored.messages] == [m.role for m in messages]


class _SilentProvider:
    """ストリームを開いたまま何も返さない（最初の応答を待っている間の再現）。"""

    kind = ProviderKind.OPENAI

    async def stream(self, req: ApiRequest) -> AsyncIterator[AssistantEvent]:
        await asyncio.Event().wait()
        yield MessageStop()

    async def count_tokens(self, req: ApiRequest) -> int:
        return 1


async def test_cancelled_wait_adds_nothing(tmp_path: Path) -> None:
    rt = _runtime(_SilentProvider(), tmp_path)
    task = asyncio.create_task(rt.run_turn("run it"))
    messages = rt.session.messages
    await _cancel_when(task, lambda: messages and messages[-1].role == "user")
    assert [m.role for m in messages] == ["user"]

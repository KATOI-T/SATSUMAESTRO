"""入力欄で SIGTERM・SIGHUP を受けた REPL は、入力欄を閉じて ``/exit`` と同じく終わる。"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

import satsumaestro.cli.app as cli_app
from satsumaestro.cli import repl as repl_mod
from satsumaestro.engine.runtime import ConversationRuntime
from tests.test_model_not_set import _Lines, _patch_provider
from tests.test_p153_missing_model import _ollama, _route_http

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="SIGHUP が無い")

_SIGNALS = [pytest.param(signal.SIGTERM, id="SIGTERM"), pytest.param(signal.SIGHUP, id="SIGHUP")]


class _Waiting:
    """入力欄の代役（待ち始めたら ``signum`` を自分に送り、閉じられるまで待つ）。"""

    def __init__(self, signum: int) -> None:
        self.signum = signum
        self.app = self
        self.future: asyncio.Future[str] | None = None
        self.handlers: list[Any] = []

    @property
    def is_running(self) -> bool:
        return self.future is not None

    def exit(self, *, exception: BaseException) -> None:
        assert self.future is not None
        self.future.set_exception(exception)

    async def prompt_async(self, *_: Any, **__: Any) -> str:
        handler = signal.getsignal(self.signum)
        self.handlers.append(handler)
        if handler in (signal.SIG_DFL, signal.SIG_IGN):
            raise EOFError
        self.future = asyncio.get_running_loop().create_future()
        os.kill(os.getpid(), self.signum)
        return await self.future


async def _run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, session: Any) -> int | None:
    _patch_provider(monkeypatch)
    _route_http(monkeypatch, _ollama())
    rt, prompter, render = await cli_app._assemble_runtime(
        "gpt-oss:120b", tmp_path, interactive=True, trusted=True
    )
    monkeypatch.setattr(repl_mod, "PromptSession", lambda **_: session)
    monkeypatch.setattr(repl_mod, "mode_prompt_fragments", lambda _: "> ")
    try:
        return await repl_mod.run_repl(rt, prompter, render, Console(record=True))  # type: ignore[arg-type]
    finally:
        await cli_app._shutdown(rt.mcp, rt.rag, rt.orchestrator, rt.background, rt.lsp)


@pytest.mark.parametrize("signum", _SIGNALS)
async def test_a_signal_at_the_input_line_closes_it_and_ends_the_repl(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, signum: int
) -> None:
    session = _Waiting(signum)
    assert await _run(monkeypatch, tmp_path, session) == signum
    assert session.handlers[0] not in (signal.SIG_DFL, signal.SIG_IGN)
    assert signal.getsignal(signum) == signal.SIG_DFL


async def test_a_turn_runs_with_the_default_handlers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[Any] = []

    async def turn(*_: Any, **__: Any) -> Any:
        seen.extend(signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP))
        raise KeyboardInterrupt

    monkeypatch.setattr(ConversationRuntime, "run_turn", turn)
    assert await _run(monkeypatch, tmp_path, _Lines(["1+1"])) is None
    assert seen == [signal.SIG_DFL, signal.SIG_DFL]


async def test_an_ignored_signal_stays_ignored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    previous = signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        session = _Waiting(signal.SIGHUP)
        assert await _run(monkeypatch, tmp_path, session) is None
        assert session.handlers == [signal.SIG_IGN]
        assert signal.getsignal(signal.SIGHUP) == signal.SIG_IGN
    finally:
        signal.signal(signal.SIGHUP, previous)

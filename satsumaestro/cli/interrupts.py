"""REPL の場面ごとに Ctrl+C（SIGINT）の受け方を切り替える。"""

from __future__ import annotations

import asyncio
import signal
import threading
from collections.abc import Callable, Coroutine, Generator
from contextlib import contextmanager
from typing import Any


@contextmanager
def _sigint_handler(handler: Callable[[int, Any], object]) -> Generator[None]:
    """区間の間だけ SIGINT の受け手を ``handler`` にし、抜けるときに元へ戻す。"""
    previous: Any = None
    swapped = False
    # 主スレッドの外と、Python から取り出せない受け手（C の側で置かれた物）では差し替えない
    if threading.current_thread() is threading.main_thread():
        try:
            previous = signal.getsignal(signal.SIGINT)
            if previous is not None:
                signal.signal(signal.SIGINT, handler)
                swapped = True
        except (ValueError, OSError):
            swapped = False
    try:
        yield
    finally:
        if swapped:
            signal.signal(signal.SIGINT, previous)


@contextmanager
def ctrl_c_raises() -> Generator[None]:
    """同期の 1 行入力の間だけ、Ctrl+C をその場の ``KeyboardInterrupt`` にする。"""
    with _sigint_handler(signal.default_int_handler):
        yield


async def run_cancellable[T](coro: Coroutine[Any, Any, T]) -> T:
    """``coro`` を子のタスクで回し、Ctrl+C では子だけを取り消して ``KeyboardInterrupt`` にする。"""
    loop = asyncio.get_running_loop()
    task = asyncio.ensure_future(coro)

    def on_sigint(_signum: int, _frame: Any) -> None:
        loop.call_soon_threadsafe(task.cancel)

    with _sigint_handler(on_sigint):
        try:
            return await task
        except asyncio.CancelledError:
            current = asyncio.current_task()
            # 呼び出し側のタスク自身の取り消しは CancelledError のまま伝える
            if task.cancelled() and (current is None or current.cancelling() == 0):
                raise KeyboardInterrupt from None
            raise

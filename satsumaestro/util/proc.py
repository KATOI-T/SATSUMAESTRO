"""子プロセスの確実な停止・回収。"""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
from dataclasses import dataclass, field
from pathlib import Path

BACKGROUND_TERMINATE_GRACE_S = 2.0
COMMAND_PREVIEW_CHARS = 80
UNREAD_TAIL_BYTES = 16_384


@dataclass
class BackgroundTask:
    """バックグラウンド bash 1 件の台帳エントリ。"""

    task_id: str
    command: str
    proc: asyncio.subprocess.Process
    log_path: Path
    read_offset: int = 0
    notified: bool = False
    waiter: asyncio.Task[int] | None = field(default=None, repr=False)

    @property
    def running(self) -> bool:
        return self.proc.returncode is None

    @property
    def command_preview(self) -> str:
        head = self.command[:COMMAND_PREVIEW_CHARS]
        return head + ("…" if len(self.command) > COMMAND_PREVIEW_CHARS else "")


class BackgroundProcessRegistry:
    """バックグラウンド bash の台帳。"""

    def __init__(self, log_dir: Path) -> None:
        self._log_dir = log_dir
        self._tasks: dict[str, BackgroundTask] = {}
        self._seq = 0

    async def spawn(self, command: str, *, cwd: Path, env: dict[str, str]) -> BackgroundTask:
        """``command`` を ``/bin/sh -c`` でデタッチ起動し、台帳へ登録して返す。"""
        self._seq += 1
        task_id = f"bg-{self._seq}"
        self._log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self._log_dir / f"{task_id}.log"
        with log_path.open("wb") as fh:
            proc = await asyncio.create_subprocess_exec(
                "/bin/sh",
                "-c",
                command,
                cwd=cwd,
                env=env,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=fh,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
        task = BackgroundTask(task_id=task_id, command=command, proc=proc, log_path=log_path)
        task.waiter = asyncio.create_task(proc.wait())
        self._tasks[task_id] = task
        return task

    def get(self, task_id: str) -> BackgroundTask | None:
        return self._tasks.get(task_id)

    def known_ids(self) -> list[str]:
        return list(self._tasks)

    def read_new_output(self, task: BackgroundTask) -> str:
        """ログファイルの未読分を読み、既読オフセットを進める。"""
        try:
            with task.log_path.open("rb") as fh:
                fh.seek(task.read_offset)
                data = fh.read()
        except OSError:
            return ""
        task.read_offset += len(data)
        if len(data) <= UNREAD_TAIL_BYTES:
            return data.decode(errors="replace")
        omitted = len(data) - UNREAD_TAIL_BYTES
        tail = data[-UNREAD_TAIL_BYTES:].decode(errors="replace")
        return f"[... {omitted} bytes of earlier unread output omitted ...]\n{tail}"

    def take_completed_unnotified(self) -> list[BackgroundTask]:
        """未通知のまま終了したタスクを返し、通知済みへマークする（1 回制御）。"""
        done = [t for t in self._tasks.values() if not t.running and not t.notified]
        for t in done:
            t.notified = True
        return done

    async def aclose(self) -> None:
        """残存プロセスを SIGTERM（猶予後 SIGKILL）で停止し、ログ置き場を片付ける。"""
        for t in self._tasks.values():
            if t.running:
                await terminate_graceful(t.proc, grace_s=BACKGROUND_TERMINATE_GRACE_S)
            if t.waiter is not None:
                try:
                    await t.waiter
                except asyncio.CancelledError:
                    pass
        shutil.rmtree(self._log_dir, ignore_errors=True)


async def terminate_graceful(proc: asyncio.subprocess.Process, *, grace_s: float) -> None:
    """プロセスグループへ SIGTERM → ``grace_s`` 待ち → 残っていれば SIGKILL。"""
    if proc.returncode is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, AttributeError, OSError):
        try:
            proc.terminate()
        except ProcessLookupError:
            return
    try:
        await asyncio.wait_for(proc.wait(), timeout=grace_s)
    except TimeoutError:
        await terminate(proc)


async def terminate(proc: asyncio.subprocess.Process) -> None:
    """``proc`` とその子孫を停止し、ゾンビ化しないよう ``wait`` で回収する。"""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, AttributeError, OSError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass
    try:
        await proc.wait()
    except ProcessLookupError:
        pass

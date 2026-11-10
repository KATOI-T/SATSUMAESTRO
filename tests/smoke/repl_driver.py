"""本物の REPL を pty で駆動する。"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import pty
import re
import select
import shutil
import signal
import struct
import sys
import tempfile
import termios
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from satsumaestro.security.trust import TRUST_FILE, TrustStore, trust_root
from tests.smoke.fake_ollama import FAKE_MODEL, SCRIPTS_DIR, FakeOllama
from tests.smoke.transcript import Report, Transcript, check, turn_ends

REQUEST = "三つのファイル a.txt、b.txt、c.txt をそれぞれ 1 行で作成して"
DELETE_REQUEST = "作成したファイルを削除してください"
BANNER = r"SATSUMAESTRO REPL —"
CODE_PROMPT = r"\[code\] satsuma>"
PLAN_PROMPT = r"\[plan\] satsuma>"
CARD_MARKERS = ("許可しますか？", "プランを承認して", "Allow this?", "Approve the plan and start?")
TRUST_QUESTION = r"\(y/n/q\): "
LANGUAGE_QUESTION = r"\((?:\d/)*\d\) \[\d\]: "
_CPR_REQUEST = b"\x1b[6n"
_CPR_REPLY = b"\x1b[1;1R"
_SHIFT_TAB = b"\x1b[Z"


class DriverError(RuntimeError):
    """駆動の失敗。"""

    def __init__(self, message: str, transcript: Transcript) -> None:
        super().__init__(f"{message}\n--- transcript (tail) ---\n{transcript.tail()}")
        self.transcript = transcript
        self.requests: list[dict[str, Any]] = []


class DriverTimeout(DriverError):
    pass


class ReplExited(DriverError):
    pass


class ReplDriver:
    """pty で ``satsumaestro repl`` を 1 つ動かす。"""

    def __init__(
        self,
        *,
        workspace: Path,
        base_url: str,
        config_home: Path,
        python: str = sys.executable,
        model: str = FAKE_MODEL,
        cols: int = 120,
        rows: int = 40,
        raw_log: Path | None = None,
        answer: str | Sequence[str] = "y",
        trusted: bool = True,
        locale: str = "ja_JP.UTF-8",
        base_url_env: str = "OLLAMA_HOST",
        extra_env: Mapping[str, str] | None = None,
        command: Sequence[str] = ("repl",),
    ) -> None:
        self.workspace = workspace
        self.command = tuple(command)
        self.extra_env = dict(extra_env or {})
        self.base_url = base_url
        self.base_url_env = base_url_env
        self.config_home = config_home
        self.python = python
        self.model = model
        self.cols = cols
        self.rows = rows
        self.answer = answer
        self.trusted = trusted
        self.locale = locale
        self._sent = 0
        self.transcript = Transcript()
        self.exited = False
        self.status: int | None = None
        self._pid = -1
        self._fd = -1
        self._raw_log = raw_log.open("wb") if raw_log is not None else None
        self._answered = dict.fromkeys(CARD_MARKERS, 0)

    def _env(self) -> dict[str, str]:
        home = self.config_home.parent
        return {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(home),
            "SATSUMA_CONFIG_HOME": str(self.config_home),
            self.base_url_env: self.base_url,
            "SATSUMA_LOG_FILE": "off",
            "PROMPT_TOOLKIT_NO_CPR": "1",
            "TERM": "xterm-256color",
            "COLUMNS": str(self.cols),
            "LINES": str(self.rows),
            "LANG": "C.UTF-8",
            "LC_MESSAGES": self.locale,
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
            **self.extra_env,
        }

    def start(self) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.config_home.mkdir(parents=True, exist_ok=True)
        if self.trusted:
            TrustStore(self.config_home / TRUST_FILE).trust(trust_root(self.workspace))
        argv = [
            self.python,
            "-m",
            "satsumaestro.cli.app",
            *self.command,
            "--cwd",
            str(self.workspace),
            "--model",
            self.model,
        ]
        env = self._env()
        pid, fd = pty.fork()
        if pid == 0:
            os.chdir(self.workspace)
            os.execve(self.python, argv, env)
        self._pid, self._fd = pid, fd
        fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", self.rows, self.cols, 0, 0))

    def _reap(self, *, block: bool = False) -> None:
        if self.exited or self._pid < 0:
            return
        pid, status = os.waitpid(self._pid, 0 if block else os.WNOHANG)
        if pid == self._pid:
            self.exited = True
            self.status = status

    def pump(self, seconds: float) -> None:
        """``seconds`` のあいだ pty を読んで転写に足す。"""
        deadline = time.monotonic() + seconds
        while not self.exited:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            ready, _, _ = select.select([self._fd], [], [], min(0.2, remaining))
            if not ready:
                self._reap()
                continue
            try:
                data = os.read(self._fd, 65536)
            except OSError:
                data = b""
            if not data:
                self._reap(block=True)
                return
            self.transcript.append(data)
            if self._raw_log is not None:
                self._raw_log.write(data)
                self._raw_log.flush()
            if _CPR_REQUEST in data:
                self.send(_CPR_REPLY)

    def send(self, data: bytes) -> None:
        os.write(self._fd, data)

    def send_line(self, text: str) -> None:
        self.send(text.encode("utf-8") + b"\r")

    def wait_for(self, pattern: str, timeout: float = 30.0) -> float:
        """剥いだ転写に ``pattern`` が現れるまで待ち、見つけた時刻（``time.monotonic``）を返す。"""
        regex = re.compile(pattern)
        deadline = time.monotonic() + timeout
        while regex.search(self.transcript.plain) is None:
            if self.exited:
                raise ReplExited(
                    f"REPL exited (status={self.status}) before {pattern!r}", self.transcript
                )
            if time.monotonic() > deadline:
                raise DriverTimeout(
                    f"timeout ({timeout:.0f}s) waiting for {pattern!r}", self.transcript
                )
            self.pump(0.2)
        return time.monotonic()

    def shift_tab(self) -> None:
        """Shift+Tab で plan へ。"""
        self.send(_SHIFT_TAB)
        self.wait_for(r"satsuma>\S*plan")

    def answer_cards(self) -> None:
        """承認カード・権限カードの出現回数が答えた数を超えていたら ``y`` を送る。"""
        plain = self.transcript.plain
        for marker in CARD_MARKERS:
            while plain.count(marker) > self._answered[marker]:
                self._answered[marker] += 1
                time.sleep(0.3)
                self.send_line(self._next_answer())

    def _next_answer(self) -> str:
        if isinstance(self.answer, str):
            return self.answer
        answers = list(self.answer)
        value = answers[min(self._sent, len(answers) - 1)]
        self._sent += 1
        return value

    def run_turn(self, text: str, timeout: float = 30.0) -> None:
        """依頼を送り、ターンの終端（``— 出力 N …`` の行）が 1 つ増えるまでカードに答えつつ待つ。"""
        self.finish_turn(self.begin_turn(text), text, timeout)

    def begin_turn(self, text: str) -> int:
        """依頼を送る。"""
        base = turn_ends(self.transcript.plain)
        self.send_line(text)
        return base

    def finish_turn(self, base: int, text: str, timeout: float = 30.0) -> None:
        """:meth:`begin_turn` の続き。"""
        deadline = time.monotonic() + timeout
        while turn_ends(self.transcript.plain) <= base:
            if self.exited:
                raise ReplExited(
                    f"REPL exited (status={self.status}) during {text!r}", self.transcript
                )
            if time.monotonic() > deadline:
                raise DriverTimeout(
                    f"timeout ({timeout:.0f}s) during turn {text!r}", self.transcript
                )
            self.pump(0.2)
            self.answer_cards()
        self.pump(0.3)

    def run_command(self, line: str, expect: str, timeout: float = 10.0) -> None:
        """スラッシュコマンドを送り、出力に ``expect``（正規表現）が出るまで待つ。"""
        regex = re.compile(expect)
        before = len(regex.findall(self.transcript.plain))
        self.send_line(line)
        deadline = time.monotonic() + timeout
        while len(regex.findall(self.transcript.plain)) <= before:
            if self.exited:
                raise ReplExited(
                    f"REPL exited (status={self.status}) during {line!r}", self.transcript
                )
            if time.monotonic() > deadline:
                raise DriverTimeout(
                    f"timeout ({timeout:.0f}s) waiting for {expect!r} after {line!r}",
                    self.transcript,
                )
            self.pump(0.2)
        self.pump(0.3)

    def exit(self, timeout: float = 5.0) -> None:
        """``/exit`` で終える。"""
        if self._pid < 0:
            return
        try:
            if not self.exited:
                try:
                    self.send_line("/exit")
                except OSError:
                    pass
                deadline = time.monotonic() + timeout
                while not self.exited and time.monotonic() < deadline:
                    self.pump(0.2)
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    if self.exited:
                        break
                    try:
                        os.kill(self._pid, sig)
                    except ProcessLookupError:
                        pass
                    deadline = time.monotonic() + 2.0
                    while not self.exited and time.monotonic() < deadline:
                        self.pump(0.2)
                self._reap(block=True)
        finally:
            if self._fd >= 0:
                os.close(self._fd)
                self._fd = -1
            if self._raw_log is not None:
                self._raw_log.close()
                self._raw_log = None
            self._pid = -1


@dataclass
class ScenarioResult:
    transcript: Transcript
    report: Report
    requests: list[dict[str, Any]]
    elapsed: float
    workspace: Path
    turns: list[str] = field(default_factory=list)
    unloads: list[str] = field(default_factory=list)


def run_scenario(
    script: Path,
    *,
    workspace: Path,
    home: Path,
    python: str = sys.executable,
    model: str = FAKE_MODEL,
    settings: dict[str, Any] | None = None,
    turns: Sequence[str] = (REQUEST, DELETE_REQUEST),
    plan: bool = True,
    commands: Sequence[tuple[str, str]] = (),
    answer: str | Sequence[str] = "y",
    raw_log: Path | None = None,
    during: Callable[[int, ReplDriver, FakeOllama], None] | None = None,
    rows: int = 40,
    cols: int = 120,
    trust_answer: str | None = None,
    locale: str = "ja_JP.UTF-8",
    screen_answer: str = "auto",
    window: int = 131072,
    loaded_window: int | None = None,
    flavor: str = "ollama",
    before: Callable[[int, ReplDriver], None] | None = None,
    env: Mapping[str, str] | None = None,
    command: Sequence[str] = ("repl",),
) -> ScenarioResult:
    """擬似サーバと REPL を起動し、plan → 依頼（→ 削除）を流して転写と検査結果を返す。"""
    workspace.mkdir(parents=True, exist_ok=True)
    if settings:
        state = workspace / ".satsuma"
        state.mkdir(exist_ok=True)
        (state / "settings.local.json").write_text(json.dumps(settings), encoding="utf-8")
    started = time.monotonic()
    fake = FakeOllama(
        script,
        model=model,
        workspace=workspace,
        window=window,
        loaded_window=loaded_window,
        flavor=flavor,
    )
    fake.bind()
    driver = ReplDriver(
        workspace=workspace,
        base_url=fake.base_url,
        base_url_env=fake.base_url_env,
        config_home=home / ".satsuma",
        python=python,
        model=model,
        answer=answer,
        raw_log=raw_log,
        rows=rows,
        cols=cols,
        trusted=trust_answer is None,
        locale=locale,
        extra_env=env,
        command=command,
    )
    try:
        driver.start()
        fake.serve()
        driver.wait_for(f"{LANGUAGE_QUESTION}|{TRUST_QUESTION}|{BANNER}")
        if re.search(LANGUAGE_QUESTION, driver.transcript.plain):
            driver.send_line(screen_answer)
        if trust_answer is not None:
            driver.wait_for(TRUST_QUESTION)
            driver.send_line(trust_answer)
        driver.wait_for(BANNER)
        driver.wait_for(CODE_PROMPT)
        if plan:
            driver.shift_tab()
        for line, expect in commands:
            driver.run_command(line, expect)
        for index, text in enumerate(turns):
            if before is not None:
                before(index, driver)
            base = driver.begin_turn(text)
            if during is not None:
                during(index, driver, fake)
            driver.finish_turn(base, text)
    except DriverError as e:
        e.requests = list(fake.requests)
        raise
    finally:
        driver.exit()
        fake.stop()
    return ScenarioResult(
        transcript=driver.transcript,
        report=check(driver.transcript.text),
        requests=list(fake.requests),
        elapsed=time.monotonic() - started,
        workspace=workspace,
        turns=list(turns),
        unloads=fake.unloads,
    )


def _python_for(repo: Path | None) -> str:
    if repo is None:
        return sys.executable
    for candidate in (repo / ".venv" / "bin" / "python", repo / ".venv" / "Scripts" / "python.exe"):
        if candidate.exists():
            return str(candidate)
    raise SystemExit(f"{repo} に .venv がありません（uv sync してください）")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="擬似プロバイダ＋pty で本物の REPL を流し、転写を検査する"
    )
    ap.add_argument("--script", type=Path, default=SCRIPTS_DIR / "plan_three_steps.jsonl")
    ap.add_argument(
        "--repo", type=Path, default=None, help="REPL を動かす側の worktree（.venv を使う）"
    )
    ap.add_argument("--model", default=FAKE_MODEL)
    ap.add_argument(
        "--settings",
        default=None,
        help="settings.local.json の中身（例: '{\"max_iterations\": 8}'）",
    )
    ap.add_argument(
        "--turn", action="append", default=None, help="依頼文（複数可。既定は作成→削除）"
    )
    ap.add_argument("--no-delete", action="store_true", help="削除のターンを流さない")
    ap.add_argument("--no-plan", action="store_true", help="Shift+Tab を送らず code のまま流す")
    ap.add_argument(
        "--workspace", type=Path, default=None, help="作業ディレクトリ（既定は一時ディレクトリ）"
    )
    ap.add_argument("--out", type=Path, default=None, help="剥いだ転写の出力先（既定は標準出力）")
    ap.add_argument("--raw", type=Path, default=None, help="生バイト列の出力先")
    ap.add_argument("--log", type=Path, default=None, help="擬似サーバの要求ログ（JSONL）の出力先")
    args = ap.parse_args(argv)

    turns = args.turn or ([REQUEST] if args.no_delete else [REQUEST, DELETE_REQUEST])
    settings = json.loads(args.settings) if args.settings else None
    temp = tempfile.mkdtemp(prefix="satsuma-smoke-") if args.workspace is None else None
    root = Path(temp) if temp is not None else args.workspace
    workspace = root / "ws" if temp is not None else root
    home = root / "home" if temp is not None else root.parent / f"{root.name}-home"
    requests: list[dict[str, Any]] = []
    try:
        result = run_scenario(
            args.script,
            workspace=workspace,
            home=home,
            python=_python_for(args.repo),
            model=args.model,
            settings=settings,
            turns=turns,
            plan=not args.no_plan,
            raw_log=args.raw,
        )
        transcript, report, requests = result.transcript, result.report, result.requests
        elapsed = result.elapsed
    except DriverError as e:
        transcript, report, requests = e.transcript, check(e.transcript.text), e.requests
        report.problems.insert(0, f"driver: {str(e).splitlines()[0]}")
        elapsed = float("nan")
    finally:
        if args.log is not None:
            args.log.write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in requests),
                encoding="utf-8",
            )
        if temp is not None:
            shutil.rmtree(temp, ignore_errors=True)
    if args.out is not None:
        args.out.write_text(transcript.text + "\n", encoding="utf-8")
    else:
        print(transcript.text)
    print(f"--- elapsed {elapsed:.1f}s ---", file=sys.stderr)
    for problem in report.problems:
        print(f"問題: {problem}", file=sys.stderr)
    if not report.problems:
        print("検査: 合格", file=sys.stderr)
    return 1 if report.problems else 0


if __name__ == "__main__":
    sys.exit(main())

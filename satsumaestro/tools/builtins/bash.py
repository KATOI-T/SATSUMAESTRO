"""bash 組み込みツール。"""

from __future__ import annotations

import asyncio
import os
import shlex
from asyncio.subprocess import DEVNULL, PIPE, STDOUT
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.sandbox import (
    AUTO_ALLOWED_GIT_CONFIG,
    FilesystemMode,
    bash_environment,
    isolation_for,
    mount_paths,
    with_git_config,
)
from satsumaestro.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from satsumaestro.tools.bash_validation import validate_bash
from satsumaestro.tools.schema import model_tool_schema

_MAX_OUTPUT_CHARS = 16_000
_KEEP_HEAD = 6_000
_KEEP_TAIL = _MAX_OUTPUT_CHARS - _KEEP_HEAD

_EXIT1_IS_ANSWER = frozenset({"grep", "egrep", "fgrep", "rg", "diff", "cmp", "test", "["})
_ASSIGNMENT_OR_WRAPPER = frozenset({"command", "builtin", "env", "time", "nice", "nohup"})


def benign_exit(command: str, returncode: int | None) -> bool:
    """非 0 終了が正常な情報か（``ls | grep x`` の 1 = 一致なし 等）。"""
    if returncode != 1:
        return False
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        tokens = list(lexer)
    except ValueError:
        return False
    last: list[str] = []
    skip_next = False
    for tok in tokens:
        if skip_next:
            skip_next = False
            continue
        if tok and all(c in "();<>|&" for c in tok):
            if tok == "|":
                last = []
            elif "<" in tok or ">" in tok:
                skip_next = not tok.endswith("&")
                if tok.endswith("&") and last and last[-1].isdigit():
                    last.pop()
            else:
                return False
            continue
        last.append(tok)
    words = [w for w in last if not ("=" in w and not w.startswith("-"))]
    while words and os.path.basename(words[0]) in _ASSIGNMENT_OR_WRAPPER:
        words = words[1:]
    return bool(words) and os.path.basename(words[0]) in _EXIT1_IS_ANSWER


_TEST_RUNNERS = frozenset(
    {"pytest", "py.test", "unittest", "tox", "nox", "jest", "vitest", "mocha", "ava", "rspec",
     "phpunit", "ctest"}
)  # fmt: skip
_TEST_SUBCOMMANDS: dict[str, frozenset[str]] = {
    "cargo": frozenset({"test", "nextest"}),
    "go": frozenset({"test"}),
    "make": frozenset({"test", "check"}),
    "deno": frozenset({"test"}),
    "dotnet": frozenset({"test"}),
    "swift": frozenset({"test"}),
    "mvn": frozenset({"test"}),
    "gradle": frozenset({"test"}),
    "playwright": frozenset({"test"}),
}
_PACKAGE_MANAGERS = frozenset({"npm", "pnpm", "yarn", "bun"})
_PREFIXES = frozenset({"env", "time", "nice", "nohup", "command", "exec", "sudo"})
_PREFIXES_WITH_VALUE = frozenset({"timeout"})
_RUNNER_PREFIXES = ({"uv", "run"}, {"poetry", "run"}, {"pdm", "run"}, {"pipenv", "run"},
                    {"pnpm", "exec"}, {"npm", "exec"}, {"bun", "x"})  # fmt: skip


def _simple_commands(command: str) -> list[list[str]]:
    """``&&`` ``||`` ``;`` ``|`` ``&`` で切った単純コマンドの語の並び。"""
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        tokens = list(lexer)
    except ValueError:
        return []
    out: list[list[str]] = [[]]
    for tok in tokens:
        if tok and all(c in ";&|()" for c in tok):
            out.append([])
        else:
            out[-1].append(tok)
    return [words for words in out if words]


def _head(words: list[str]) -> list[str]:
    """代入と前置きを外した、実際に走るコマンドからの語の並び（小文字・パスは名前だけ）。"""
    rest = [os.path.basename(w).lower() for w in words if not ("=" in w and not w.startswith("-"))]
    while rest:
        if rest[0] in _PREFIXES:
            rest = rest[1:]
        elif rest[0] in _PREFIXES_WITH_VALUE:
            rest = rest[2:]
        elif rest[0] in {"npx", "bunx", "pnpx"}:
            rest = rest[1:]
        elif len(rest) >= 2 and {rest[0], rest[1]} in _RUNNER_PREFIXES:
            rest = rest[2:]
        elif len(rest) >= 3 and rest[0].startswith("python") and rest[1] == "-m":
            rest = rest[2:]
        else:
            break
    return rest


def _runs_tests(command: str) -> bool:
    """単純コマンドのどれかがテストの実行か（タイムアウトの文を変えるためだけに使う）。"""
    for words in _simple_commands(command):
        head = _head(words)
        if not head:
            continue
        first, second = head[0], head[1] if len(head) > 1 else ""
        if first in _TEST_RUNNERS:
            return True
        if second in _TEST_SUBCOMMANDS.get(first, ()):
            return True
        if first in _PACKAGE_MANAGERS:
            script = head[2] if second == "run" and len(head) > 2 else second
            if script in {"test", "t"} or script.startswith("test:"):
                return True
    return False


def _clip_output(text: str) -> str:
    """上限超過の出力を先頭＋末尾で保持し、省略量をマーカで明示する。"""
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    omitted = len(text) - _KEEP_HEAD - _KEEP_TAIL
    return (
        text[:_KEEP_HEAD] + f"\n\n[... output truncated: {omitted} characters omitted "
        f"(total exceeded {_MAX_OUTPUT_CHARS}) ...]\n\n" + text[-_KEEP_TAIL:]
    )


async def _kill_and_reap(proc: asyncio.subprocess.Process) -> None:
    """``proc`` を kill して回収する。"""
    try:
        proc.kill()
    except ProcessLookupError:
        pass
    await proc.wait()


MAX_TIMEOUT_S = 600.0


class Bash(Tool):
    """workspace 内でのシェルコマンド実行。"""

    class Input(BaseModel):
        command: str = Field(
            description="Shell command to run with /bin/sh from the workspace root."
        )
        timeout_s: float = Field(
            default=120.0,
            gt=0.0,
            description=(
                "Seconds to wait before killing the command (at most 600; larger values are "
                "lowered to 600)."
            ),
        )
        filesystem_mode: Literal["off", "workspace-only", "allow-list"] = Field(
            default="workspace-only",
            description=(
                "Filesystem isolation mode: 'workspace-only' (default) keeps writes jailed "
                "to the workspace, 'allow-list' additionally permits allowed_mounts, "
                "'off' lifts the advisory restriction (trusted use only)."
            ),
        )
        isolate_network: bool = Field(
            default=False,
            description=(
                "Best-effort network isolation: neutralize proxy variables in the child "
                "process environment (no OS-level isolation)."
            ),
        )
        allowed_mounts: list[str] = Field(
            default_factory=list,
            description=(
                "Extra paths (absolute or workspace-relative) permitted when "
                "filesystem_mode is 'allow-list'. Ignored otherwise."
            ),
        )
        run_in_background: bool = Field(
            default=False,
            description=(
                "Run the command in the background and return a task id immediately. "
                "Read incremental output later with bash_output. timeout_s is ignored "
                "for background tasks. Intended for long-running commands such as "
                "builds, tests and servers."
            ),
        )

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="bash",
        description="Run a shell command in the workspace (validated, sandboxed cwd).",
        input_schema=model_tool_schema(Input),
        required_mode=PermissionMode.DANGER_FULL_ACCESS,
        parallel_safe=False,
    )

    async def run(self, inp: BaseModel, ctx: ToolContext) -> ToolResult:
        assert isinstance(inp, Bash.Input)
        mode = ctx.base_mode or PermissionMode(ctx.config.permission.default_mode)
        explicit = inp.model_fields_set
        isolation = isolation_for(
            ctx.config.sandbox,
            filesystem_mode=(
                FilesystemMode(inp.filesystem_mode) if "filesystem_mode" in explicit else None
            ),
            isolate_network=inp.isolate_network if "isolate_network" in explicit else None,
            allowed_mounts=inp.allowed_mounts if "allowed_mounts" in explicit else None,
        )
        validate_bash(inp.command, ctx.jail, mode=mode, workspace=ctx.workspace_root)
        mounts = mount_paths(isolation, ctx.workspace_root)
        base = ctx.env if ctx.env is not None else os.environ
        env = bash_environment(base, isolation=isolation, mounts=mounts)
        if ctx.auto_allowed:
            env = with_git_config(env, AUTO_ALLOWED_GIT_CONFIG)
        if inp.run_in_background:
            if ctx.background is None:
                return ToolResult(
                    output="background execution is unavailable in this context",
                    is_error=True,
                )
            task = await ctx.background.spawn(inp.command, cwd=ctx.workspace_root, env=env)
            return ToolResult(
                output=(
                    f"started background task {task.task_id} (pid {task.proc.pid}): "
                    f"{task.command_preview}\n"
                    f'Read incremental output with bash_output(task_id="{task.task_id}").'
                )
            )
        proc = await asyncio.create_subprocess_exec(
            "/bin/sh",
            "-c",
            inp.command,
            cwd=ctx.workspace_root,
            env=env,
            stdin=DEVNULL,
            stdout=PIPE,
            stderr=STDOUT,
        )
        timeout_s = min(inp.timeout_s, MAX_TIMEOUT_S)
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except TimeoutError:
            await _kill_and_reap(proc)
            capped = (
                f" timeout_s is capped at {MAX_TIMEOUT_S:g}s; run longer commands with "
                "run_in_background and read them with bash_output."
                if inp.timeout_s > MAX_TIMEOUT_S
                else ""
            )
            if _runs_tests(inp.command):
                return ToolResult(
                    output=(
                        f"timeout after {timeout_s}s. This was a test run, so a test may be "
                        "stuck waiting (on a server, on input, on a lock) or looping. Find the "
                        "test that does not finish instead of only raising timeout_s." + capped
                    ),
                    is_error=True,
                )
            plain = f"timeout after {timeout_s}s"
            return ToolResult(output=f"{plain}.{capped}" if capped else plain, is_error=True)
        except asyncio.CancelledError:
            await _kill_and_reap(proc)
            raise
        text = _clip_output(out.decode(errors="replace"))
        if proc.returncode != 0:
            suffix = f"(exit code {proc.returncode})"
            text = f"{text}\n{suffix}" if text and not text.endswith("\n") else f"{text}{suffix}"
        failed = proc.returncode != 0 and not benign_exit(inp.command, proc.returncode)
        return ToolResult(output=text, is_error=failed)

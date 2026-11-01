"""パスに NUL 文字を含むツール呼び出しでターンごと落ちない。"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

import satsumaestro.engine.runtime as runtime_module
from satsumaestro.engine.events import MessageStop, TextDelta, ToolUse
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.security import state_write_guard
from satsumaestro.security.deletion_guard import (
    PROHIBITED_DELETION_PREFIX,
    bash_targets,
    check_bash,
    check_path,
    path_targets,
)
from satsumaestro.security.jail import WorkspaceJail
from satsumaestro.session.models import ToolResultBlock
from satsumaestro.util.errors import PathEscapeError
from tests.test_p184_write_chunks import _runtime

NUL_NAME = "a\x00b.txt"
NUL_DIR = "d\x00ir/a.txt"


@pytest.mark.parametrize("raw", [NUL_NAME, NUL_DIR, "/tmp/a\x00b"])
def test_jail_treats_unresolvable_path_as_escape(tmp_path: Path, raw: str) -> None:
    jail = WorkspaceJail(tmp_path)
    with pytest.raises(PathEscapeError, match="cannot be resolved"):
        jail.resolve(raw)
    assert jail.is_inside(raw) is False
    assert jail.hides(raw) is False


def test_jail_still_resolves_ordinary_paths(tmp_path: Path) -> None:
    jail = WorkspaceJail(tmp_path)
    assert jail.resolve("sub/ok.txt") == (tmp_path / "sub" / "ok.txt").resolve()
    with pytest.raises(PathEscapeError):
        jail.resolve("../outside.txt")


@pytest.mark.parametrize("path", [NUL_NAME, NUL_DIR, "d\x00ir/."])
def test_delete_path_with_nul_is_not_prohibited_and_not_tracked(tmp_path: Path, path: str) -> None:
    assert check_path(path, tmp_path) is None
    assert path_targets(path, tmp_path) == []


@pytest.mark.parametrize("command", ["rm 'd\x00ir/a'", "rm 'd\x00ir/*'", "rm 'a\x00b.txt'"])
def test_bash_deletion_with_nul_is_not_prohibited_and_not_tracked(
    tmp_path: Path, command: str
) -> None:
    assert check_bash(command, tmp_path) is None
    assert bash_targets(command, tmp_path) == []


def test_bash_cd_into_nul_dir_stays_put(tmp_path: Path) -> None:
    command = "cd 'd\x00ir' && rm x"
    assert check_bash(command, tmp_path) is None
    assert bash_targets(command, tmp_path) == [tmp_path.resolve() / "x"]


def test_ordinary_prohibitions_are_unchanged(tmp_path: Path) -> None:
    for path in ("..", "/usr"):
        reason = check_path(path, tmp_path)
        assert reason is not None and reason.startswith(PROHIBITED_DELETION_PREFIX), path
    assert check_bash("rm -rf .satsuma", tmp_path) is not None


def test_state_write_guard_survives_nul(tmp_path: Path) -> None:
    assert state_write_guard.check_tool("write_file", NUL_DIR, tmp_path) is None
    assert state_write_guard.check_bash("echo x > 'd\x00ir/.satsuma/policy.md'", tmp_path) is None
    assert state_write_guard.check_tool("write_file", ".satsuma/policy.md", tmp_path) is not None


_CASES: dict[str, ToolUse] = {
    "write_file name": ToolUse(
        id="t", name="write_file", input={"path": NUL_NAME, "content": "x\n"}
    ),
    "write_file dir": ToolUse(id="t", name="write_file", input={"path": NUL_DIR, "content": "x\n"}),
    "edit_file dir": ToolUse(
        id="t", name="edit_file", input={"path": NUL_DIR, "old": "a", "new": "b"}
    ),
    "append_file dir": ToolUse(
        id="t", name="append_file", input={"path": NUL_DIR, "content": "x\n"}
    ),
    "delete_path name": ToolUse(id="t", name="delete_path", input={"path": NUL_NAME}),
    "delete_path dir": ToolUse(id="t", name="delete_path", input={"path": "d\x00ir/a"}),
    "bash rm dir": ToolUse(id="t", name="bash", input={"command": "rm 'd\x00ir/a'", "timeout": 5}),
    "bash rm glob": ToolUse(id="t", name="bash", input={"command": "rm 'd\x00ir/*'", "timeout": 5}),
    "bash cd": ToolUse(id="t", name="bash", input={"command": "cd 'd\x00ir' && ls", "timeout": 5}),
    "read_file dir": ToolUse(id="t", name="read_file", input={"path": NUL_DIR}),
}


def _results(rt: ConversationRuntime) -> list[ToolResultBlock]:
    return [b for m in rt.session.messages for b in m.blocks if isinstance(b, ToolResultBlock)]


def _said_ok(rt: ConversationRuntime) -> bool:
    return any(getattr(b, "text", None) == "ok" for m in rt.session.messages for b in m.blocks)


def _created_files(tmp_path: Path) -> list[Path]:
    return [p for p in tmp_path.rglob("*") if p.is_file() and ".satsuma" not in p.parts]


@pytest.mark.parametrize("label", list(_CASES))
async def test_nul_path_call_fails_as_tool_result_not_as_turn(tmp_path: Path, label: str) -> None:
    rt = _runtime([[_CASES[label], MessageStop()], [TextDelta(text="ok"), MessageStop()]], tmp_path)
    await rt.run_turn("やって")
    results = _results(rt)
    assert len(results) == 1 and results[0].is_error, results
    assert _said_ok(rt), "ツール失敗の後に次の応答が続く"
    assert _created_files(tmp_path) == []


async def test_guard_exception_becomes_invalid_input_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def boom(path: str, cwd: Path) -> str | None:
        raise RuntimeError("guard exploded")

    monkeypatch.setattr(runtime_module, "check_path", boom)
    call = ToolUse(id="t", name="delete_path", input={"path": "x.txt"})
    rt = _runtime([[call, MessageStop()], [TextDelta(text="ok"), MessageStop()]], tmp_path)
    with caplog.at_level(logging.WARNING, logger="satsumaestro.engine.runtime"):
        await rt.run_turn("消して")
    results = _results(rt)
    assert len(results) == 1 and results[0].is_error
    first = results[0].output.splitlines()[0]
    assert first.startswith("invalid input:") and "RuntimeError: guard exploded" in first
    assert "deletion guard could not read the tool input" in caplog.text
    assert _said_ok(rt)


async def test_deletion_target_exception_becomes_invalid_input_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(path: str, cwd: Path) -> list[Path]:
        raise RuntimeError("targets exploded")

    monkeypatch.setattr(runtime_module, "path_targets", boom)
    call = ToolUse(id="t", name="delete_path", input={"path": "x.txt"})
    rt = _runtime([[call, MessageStop()], [TextDelta(text="ok"), MessageStop()]], tmp_path)
    await rt.run_turn("消して")
    results = _results(rt)
    assert len(results) == 1 and results[0].is_error
    first = results[0].output.splitlines()[0]
    assert first.startswith("invalid input:") and "targets exploded" in first
    assert _said_ok(rt)

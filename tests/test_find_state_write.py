"""find の -exec の書き込みが状態ディレクトリに届くかは式で決め、名前の型はパスとみなさない。"""

from __future__ import annotations

from pathlib import Path

import pytest

from satsumaestro.engine.events import MessageStop, TextDelta, ToolUse
from satsumaestro.security import deletion_guard, state_write_guard
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.state_write_guard import PROHIBITED_STATE_WRITE_PREFIX
from tests.test_p145_plan_exit_nudge import _runtime
from tests.test_p232_state_write_guard import _Capture, _outputs


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "work"
    (ws / ".satsuma" / "evals" / "s1" / "__pycache__").mkdir(parents=True)
    (ws / ".satsuma" / "policy.md").write_text("rules\n")
    (ws / ".satsuma" / "settings.local.json").write_text("{}\n")
    (ws / ".satsuma" / "evals" / "s1" / "__pycache__" / "run.pyc").write_bytes(b"\0")
    (ws / "src" / "__pycache__").mkdir(parents=True)
    (ws / "src" / "a.py").write_text("x = 1\n")
    (ws / "src" / "__pycache__" / "a.pyc").write_bytes(b"\0")
    (ws / "notes.md").write_text("notes\n")
    return ws


def _refused(command: str, cwd: Path) -> bool:
    reason = state_write_guard.check_bash(command, cwd)
    return reason is not None and reason.startswith(PROHIBITED_STATE_WRITE_PREFIX)


@pytest.mark.parametrize(
    "command",
    [
        "find . -name '*.md' -exec sed -i '' s/a/b/ {} +",
        "find . -name '*.json' -exec chmod 600 {} +",
        "find . -type f -execdir sed -i s/a/b/ {} \\;",
        "find . -name '*.md' -ok sed -i s/a/b/ {} \\;",
        "find . -path './.satsuma/*' -exec sed -i s/a/b/ {} +",
        "find . -name '*.md' -exec sh -c 'cat \"$1\" > /tmp/x' _ {} \\;",
        "timeout 5 find . -name '*.md' -exec sed -i s/a/b/ {} +",
        "cd src && find .. -name '*.md' -exec sed -i s/a/b/ {} +",
        "find . -name '*.md' -exec sed -i s/a/b/ {} + -o -name '*.py' -exec touch {} +",
    ],
)
def test_find_writing_into_the_state_dir_is_prohibited(workspace: Path, command: str) -> None:
    assert _refused(command, workspace)


@pytest.mark.parametrize(
    "command",
    [
        "find . -name __pycache__ -not -path './.satsuma/*' -exec rm -rf {} +",
        "find . ! -path './.satsuma*' -name '*.pyc' -delete",
        "find . -path ./.satsuma -prune -o -name __pycache__ -exec rm -rf {} +",
        "find . -not -path './.satsuma/*' -name '*.md' -exec sed -i s/a/b/ {} +",
        "find . -name '*.py' -exec sed -i s/a/b/ {} +",
        "find src -name '*.py' -exec sed -i s/a/b/ {} +",
        "find . -name '*.md' -exec grep -l x {} +",
        "find . -path './.satsuma/*' -print",
    ],
)
def test_find_that_leaves_the_state_dir_alone_passes_both_guards(
    workspace: Path, command: str
) -> None:
    assert state_write_guard.check_bash(command, workspace) is None
    assert deletion_guard.check_bash(command, workspace) is None


@pytest.mark.parametrize(
    "command",
    [
        "find .satsuma -name x -exec touch {} +",
        "find . -newer .satsuma/policy.md -exec touch {} +",
        "find . -name x -fprint .satsuma/out.txt",
        "find . -name x -exec cp {} .satsuma/ \\;",
    ],
)
def test_paths_other_than_patterns_still_count(workspace: Path, command: str) -> None:
    assert _refused(command, workspace)


def test_deleting_with_find_is_left_to_the_deletion_guard(workspace: Path) -> None:
    command = "find . -name '*.pyc' -delete"
    assert state_write_guard.check_bash(command, workspace) is None
    reason = deletion_guard.check_bash(command, workspace)
    assert reason is not None and reason.startswith(deletion_guard.PROHIBITED_DELETION_PREFIX)


@pytest.mark.parametrize(
    "command", ["touch x && find .", "find . && touch x", "find . -maxdepth 1 && touch x"]
)
def test_a_find_without_an_action_does_not_count_as_a_write(workspace: Path, command: str) -> None:
    assert state_write_guard.check_bash(command, workspace) is None


def test_a_root_named_dot_counts_for_writes(workspace: Path) -> None:
    """rm は . を消さないが、書き込みのコマンドには . がそのまま渡る。"""
    assert _refused("find . -maxdepth 0 -exec chmod -R 700 {} +", workspace)


def test_patterns_still_count_for_the_agent_config_files(workspace: Path) -> None:
    (workspace / ".env").write_text("A=1\n")
    target = state_write_guard.config_file_bash("find . -name .env -delete", workspace)
    assert target == workspace / ".env"


async def _run_bash(ws: Path, command: str) -> str:
    """カードの出ない danger-full-access で、モデルが bash を 1 回呼ぶターンを回す。"""
    rt, script = _runtime(ws)
    rt.set_mode(main="code", sub=PermissionMode.DANGER_FULL_ACCESS)
    script[:] = [
        [ToolUse(id="b1", name="bash", input={"command": command}), MessageStop()],
        [TextDelta(text="ok"), MessageStop()],
    ]
    await rt.run_turn("run it", _Capture())
    return _outputs(rt)["b1"].output


async def test_a_find_that_leaves_the_state_dir_out_runs_through_a_turn(workspace: Path) -> None:
    output = await _run_bash(
        workspace, "find . -name __pycache__ -not -path './.satsuma/*' -exec rm -rf {} +"
    )
    assert not output.startswith(("deletion prohibited", PROHIBITED_STATE_WRITE_PREFIX))
    assert not (workspace / "src" / "__pycache__").exists()
    assert (workspace / ".satsuma" / "evals" / "s1" / "__pycache__" / "run.pyc").exists()


async def test_a_find_that_rewrites_the_policy_is_refused_in_a_turn(workspace: Path) -> None:
    output = await _run_bash(workspace, "find . -name '*.md' -exec sed -i.bak s/rules/x/ {} +")
    assert output.startswith(PROHIBITED_STATE_WRITE_PREFIX)
    assert (workspace / ".satsuma" / "policy.md").read_text() == "rules\n"
    assert (workspace / "notes.md").read_text() == "notes\n"

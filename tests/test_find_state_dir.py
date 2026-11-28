"""find の式が状態ディレクトリ（.satsuma）の中身まで消し得るなら、削除の絶対禁止で止める。"""

from __future__ import annotations

from pathlib import Path

import pytest

from satsumaestro.security import find_expr
from satsumaestro.security.deletion_guard import PROHIBITED_DELETION_PREFIX, check_bash


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "work"
    (ws / ".satsuma" / "sessions").mkdir(parents=True)
    (ws / ".satsuma" / "policy.md").write_text("rules\n")
    (ws / ".satsuma" / "settings.local.json").write_text("{}\n")
    (ws / ".satsuma" / "sessions" / "s1.jsonl").write_text("{}\n")
    (ws / "src" / "__pycache__").mkdir(parents=True)
    (ws / "src" / "a.py").write_text("x = 1\n")
    (ws / "src" / "__pycache__" / "a.pyc").write_bytes(b"\0")
    (ws / "notes.md").write_text("notes\n")
    (ws / "build").mkdir()
    return ws


def _refused(command: str, cwd: Path) -> bool:
    reason = check_bash(command, cwd)
    return reason is not None and reason.startswith(PROHIBITED_DELETION_PREFIX)


@pytest.mark.parametrize(
    "command",
    [
        "find . -delete",
        "find . -name '*.md' -delete",
        "find . -type f -delete",
        "find . ! -name '*.py' -delete",
        "find . -not -name '*.py' -delete",
        "find . -name '*.json' -exec rm {} +",
        "find . -name '*.json' -exec rm -f {} \\;",
        "find . -name '*.json' -exec env rm {} +",
        "find . -name '*.json' -execdir /bin/rm {} +",
        "yes | find . -name '*.md' -ok rm {} \\;",
        "find . -name '*.md' -okdir rm -f {} \\;",
        "find . -path ./.satsuma -prune -o -name '*.md' -delete",
        "find . -type d -name '.*' -exec rm -rf {} +",
        "find . -maxdepth 2 -name '*.md' -delete",
        "find . -mtime +7 -delete",
        "find . -regex '.*md' -delete",
        "find ./ -name '*.md' -delete",
        "find * .satsuma/.. -name '*.md' -delete",
        "cd src && find .. -name '*.md' -delete",
        "find . -name '*.md' -print -o -delete",
    ],
)
def test_find_reaching_the_state_dir_is_prohibited(workspace: Path, command: str) -> None:
    reason = check_bash(command, workspace)
    assert reason is not None and reason.startswith(PROHIBITED_DELETION_PREFIX)
    assert "state directory" in reason or "working directory" in reason


@pytest.mark.parametrize(
    "command",
    [
        "find . -name '*.pyc' -delete",
        "find . -name __pycache__ -exec rm -rf {} +",
        "find . -name '*.pyc' -exec rm {} \\;",
        "find . -maxdepth 1 -name '*.md' -delete",
        "find . -path ./.satsuma -prune -o -name '*.md' -exec rm {} +",
        "find . ! -path './.satsuma*' -mtime +7 -delete",
        "find . -mindepth 3 -name '*.md' -delete",
        "find . -name '*.md' -print -o -name '*.pyc' -delete",
        "find . -type d -empty -delete",
        "find build -delete",
        "find src -type f -delete",
        "find . -name '*.md' -exec grep -l x {} +",
    ],
)
def test_find_that_stays_out_of_the_state_dir_is_allowed(workspace: Path, command: str) -> None:
    assert check_bash(command, workspace) is None


def test_an_empty_directory_inside_the_state_dir_counts(workspace: Path) -> None:
    (workspace / ".satsuma" / "empty").mkdir()
    assert _refused("find . -type d -empty -delete", workspace)


def test_without_a_state_dir_find_is_not_limited(tmp_path: Path) -> None:
    ws = tmp_path / "plain"
    ws.mkdir()
    (ws / "a.md").write_text("a\n")
    assert check_bash("find . -name '*.md' -delete", ws) is None


@pytest.fixture
def cached(workspace: Path) -> Path:
    """状態ディレクトリの中にも __pycache__ がある作業場所。"""
    (workspace / ".satsuma" / "evals" / "s1" / "__pycache__").mkdir(parents=True)
    (workspace / ".satsuma" / "evals" / "s1" / "__pycache__" / "run.pyc").write_bytes(b"\0")
    return workspace


@pytest.mark.parametrize(
    "command",
    [
        "find . -name __pycache__ -not -path './.satsuma/*' -exec rm -rf {} +",
        "find . ! -path './.satsuma*' -name '*.pyc' -delete",
        "find . ! -ipath './.SATSUMA*' -name '*.pyc' -delete",
        "find . -not -path '*/.satsuma/*' -name '*.pyc' -delete",
        "find . -path ./.satsuma -prune -o -name __pycache__ -exec rm -rf {} +",
    ],
)
def test_leaving_the_state_dir_out_by_path_is_allowed(cached: Path, command: str) -> None:
    assert _refused("find . -name __pycache__ -exec rm -rf {} +", cached)
    assert check_bash(command, cached) is None


def test_a_path_test_skips_the_state_dir_without_walking_it(
    cached: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """中を全部たどると上限を超える状態ディレクトリでも、パスで除いた式は通る。"""
    monkeypatch.setattr(find_expr, "_MAX_ENTRIES", 3)
    assert check_bash("find . ! -path './.satsuma*' -name '*.orig' -delete", cached) is None
    assert check_bash("find . -not -path './.satsuma/*' -name '*.orig' -delete", cached) is None
    assert _refused("find . -name '*.orig' -delete", cached)


def test_a_linked_state_dir_inside_the_workspace_is_guarded_at_its_target(
    tmp_path: Path,
) -> None:
    ws = tmp_path / "work"
    (ws / "meta" / "state").mkdir(parents=True)
    (ws / "meta" / "state" / "policy.md").write_text("rules\n")
    (ws / ".satsuma").symlink_to("meta/state")
    assert _refused("find . -name '*.md' -delete", ws)
    assert _refused("find . -name meta -exec rm -rf {} +", ws)
    assert _refused("find . -name .satsuma -delete", ws)
    assert check_bash("find . -name '*.pyc' -delete", ws) is None


def test_a_link_to_a_state_dir_outside_is_followed_only_with_follow_options(
    tmp_path: Path,
) -> None:
    ws = tmp_path / "work"
    ws.mkdir()
    outside = tmp_path / "state"
    outside.mkdir()
    (outside / "policy.md").write_text("rules\n")
    (ws / ".satsuma").symlink_to(outside)
    assert check_bash("find . -name '*.md' -delete", ws) is None
    assert _refused("find -L . -name '*.md' -delete", ws)
    assert _refused("find . -follow -name '*.md' -delete", ws)
    assert check_bash("find -L -P . -name '*.md' -delete", ws) is None


def test_the_user_config_home_is_guarded_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    (home / ".satsuma").mkdir(parents=True)
    (home / ".satsuma" / "settings.json").write_text("{}\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("SATSUMA_CONFIG_HOME", str(home / ".satsuma"))
    ws = tmp_path / "work"
    ws.mkdir()
    assert _refused(f"find {home} -name '*.json' -delete", ws)
    assert _refused("find ~ -name '*.json' -delete", ws)
    assert _refused("find $HOME -type f -delete", ws)
    assert check_bash("find ~ -name '*.pyc' -delete", ws) is None


@pytest.mark.parametrize(
    "command",
    [
        "find -L / -delete",
        "find -H / -name x -delete",
        "find -P -L / -delete",
        "find -O3 / -delete",
        "find -D tree / -delete",
        "find -E / -regex x -delete",
        "find -x / -delete",
        "find -f / -delete",
        "find -f/ -delete",
        "find -xf / -delete",
        "find -L .. -delete",
        "find -d .. -name x -exec rm {} +",
    ],
)
def test_options_before_the_starting_points_do_not_hide_them(workspace: Path, command: str) -> None:
    assert _refused(command, workspace)


@pytest.mark.parametrize("action", ["-exec", "-execdir", "-ok", "-okdir"])
def test_every_exec_action_that_runs_rm_is_checked(workspace: Path, action: str) -> None:
    assert _refused(f"find / {action} rm -rf {{}} \;", workspace)
    assert _refused(f"find .. -name x {action} rm {{}} +", workspace)

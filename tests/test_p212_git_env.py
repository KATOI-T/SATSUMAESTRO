"""コミットのフックの中でテストを流しても、外のリポジトリを書き換えない。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_TARGET = "tests/test_p33_worker_ledger.py::test_preflight_warns_about_untracked_paths"


def _git(*args: str, cwd: Path) -> str:
    done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
    return done.stdout.strip()


def test_tests_in_a_worktree_hook_leave_the_outer_repository_alone(
    tmp_path: Path, repo_root: Path
) -> None:
    main = tmp_path / "main"
    main.mkdir()
    _git("init", "-q", cwd=main)
    identity = ("-c", "user.name=t", "-c", "user.email=t@example.com")
    _git(*identity, "commit", "-q", "--allow-empty", "-m", "init", cwd=main)
    worktree = tmp_path / "wt"
    _git("worktree", "add", "-q", str(worktree), cwd=main)
    gitdir = main / ".git" / "worktrees" / "wt"
    hook_env = {
        **os.environ,
        "GIT_DIR": str(gitdir),
        "GIT_INDEX_FILE": str(gitdir / "index"),
        "GIT_PREFIX": "",
    }

    inner = [f"--basetemp={tmp_path / 'inner'}", "-p", "no:cacheprovider", "-q", _TARGET]
    done = subprocess.run(
        [sys.executable, "-m", "pytest", *inner],
        cwd=repo_root,
        env=hook_env,
        capture_output=True,
        text=True,
    )

    assert done.returncode == 0, done.stdout + done.stderr
    assert _git("config", "--bool", "core.bare", cwd=main) == "false"
    assert _git("ls-files", cwd=worktree) == ""
    assert _git("status", "--short", cwd=main) == ""

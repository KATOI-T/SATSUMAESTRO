"""プロジェクトのファイルを作業ディレクトリから git の根まで探すテスト。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from satsumaestro.cli import app as cli_app
from satsumaestro.config.loader import ConfigLoader
from satsumaestro.security import PermissionMode, load_policies
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.skill.resolver import skill_dirs
from satsumaestro.util.paths import (
    ensure_state_gitignore,
    find_project_file,
    project_dirs,
)

RO = PermissionMode.READ_ONLY
WW = PermissionMode.WORKSPACE_WRITE
DFA = PermissionMode.DANGER_FULL_ACCESS


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """git の根（``.git`` だけ）と、その下の ``pkg/sub``。"""
    monkeypatch.setenv("SATSUMA_CONFIG_HOME", str(tmp_path / "home"))
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    (root / "pkg" / "sub").mkdir(parents=True)
    return root


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_project_dirs_stop_at_git_root(repo: Path) -> None:
    sub = repo / "pkg" / "sub"
    assert project_dirs(sub) == [sub.resolve(), (repo / "pkg").resolve(), repo.resolve()]


def test_project_dirs_outside_git_is_cwd_only(tmp_path: Path) -> None:
    plain = tmp_path / "plain" / "dir"
    plain.mkdir(parents=True)
    _write(tmp_path / "plain" / ".env", "X=1\n")
    assert project_dirs(plain) == [plain.resolve()]
    assert find_project_file(plain, ".env") is None


def test_find_project_file_prefers_nearest(repo: Path) -> None:
    _write(repo / ".env", "A=root\n")
    _write(repo / "pkg" / ".env", "A=pkg\n")
    assert find_project_file(repo / "pkg" / "sub", ".env") == (repo / "pkg" / ".env").resolve()


def test_root_settings_survive_a_sessions_only_state_dir(repo: Path) -> None:
    _write(repo / ".satsuma" / "settings.json", json.dumps({"provider": {"default_model": "m"}}))
    sub = repo / "pkg" / "sub"
    (sub / ".satsuma" / "sessions").mkdir(parents=True)
    cfg = ConfigLoader(sub, config_home=repo.parent / "home", trusted=True).load()
    assert cfg.provider.default_model == "m"


def test_dotenv_is_read_from_the_root(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    _write(repo / ".env", "OLLAMA_MODEL=from-root\n")
    cfg = ConfigLoader(repo / "pkg" / "sub", config_home=repo.parent / "home", trusted=True).load()
    assert cfg.provider.default_model == "from-root"


def test_policies_stack_from_the_root(repo: Path) -> None:
    _write(
        repo / ".satsuma" / "policy.md",
        "---\nmode: read-only\n---\n## Deny\n- bash when command glob: rm *\n",
    )
    _write(
        repo / "pkg" / ".satsuma" / "policy.md",
        "## Allow\n- bash when command glob: rm *\n- bash when command glob: ls*\n",
    )
    files = cli_app._policy_files(repo / "pkg" / "sub", ".satsuma/policy.md")
    parsed = load_policies(files, WW)
    p = parsed.policy
    assert p.mode is RO
    assert p.authorize("bash", {"command": "rm -rf build"}, DFA).decision == "deny"
    assert p.authorize("bash", {"command": "ls -la"}, DFA).decision != "deny"


def test_nearest_mode_wins(repo: Path) -> None:
    _write(repo / ".satsuma" / "policy.md", "---\nmode: read-only\n---\n")
    _write(repo / "pkg" / ".satsuma" / "policy.md", "---\nmode: workspace-write\n---\n")
    files = cli_app._policy_files(repo / "pkg", ".satsuma/policy.md")
    assert load_policies(files, DFA).policy.mode is WW


def test_policy_warnings_name_the_file_when_stacked(repo: Path) -> None:
    _write(repo / ".satsuma" / "policy.md", "## Deny\n- bash when nope glob: x\n")
    _write(repo / "pkg" / ".satsuma" / "policy.md", "## Allow\n- read_file\n")
    files = cli_app._policy_files(repo / "pkg", ".satsuma/policy.md")
    warnings = load_policies(files, WW).warnings
    assert warnings and all(w.startswith(str(repo / ".satsuma" / "policy.md")) for w in warnings)


def test_single_policy_is_unchanged(repo: Path) -> None:
    _write(repo / ".satsuma" / "policy.md", "---\nmode: read-only\n---\n## Denied tools\n- bash\n")
    files = cli_app._policy_files(repo, ".satsuma/policy.md")
    parsed = load_policies(files, WW)
    assert parsed.policy.mode is RO
    assert parsed.policy.blocked_tools == ["bash"]


def test_absolute_policy_file_is_read_once(repo: Path) -> None:
    path = repo / "custom-policy.md"
    _write(path, "## Deny\n- bash\n")
    assert cli_app._policy_files(repo / "pkg" / "sub", str(path)) == [path]


def test_memory_uses_the_nearest_file_or_cwd(repo: Path) -> None:
    sub = repo / "pkg" / "sub"
    layers = cli_app._memory_layers(sub, ".satsuma/memory.md", project=True)
    assert layers[0].path == sub / ".satsuma" / "memory.md"
    _write(repo / ".satsuma" / "memory.md", "")
    layers = cli_app._memory_layers(sub, ".satsuma/memory.md", project=True)
    assert layers[0].path == (repo / ".satsuma" / "memory.md").resolve()


def test_skills_are_found_from_the_root(repo: Path) -> None:
    (repo / ".satsuma" / "skills").mkdir(parents=True)
    assert skill_dirs(repo / "pkg" / "sub") == [(repo / ".satsuma" / "skills").resolve()]


def test_state_gitignore_is_written_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SATSUMA_CONFIG_HOME", str(tmp_path / "home"))
    ensure_state_gitignore(tmp_path)
    path = tmp_path / ".satsuma" / ".gitignore"
    text = path.read_text(encoding="utf-8")
    assert "sessions/" in text and "rag.db" in text
    assert "policy.md" not in text and "memory.md" not in text
    path.write_text("custom\n", encoding="utf-8")
    ensure_state_gitignore(tmp_path)
    assert path.read_text(encoding="utf-8") == "custom\n"


def test_state_gitignore_skips_the_user_config_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SATSUMA_CONFIG_HOME", str(tmp_path / ".satsuma"))
    ensure_state_gitignore(tmp_path)
    assert not (tmp_path / ".satsuma" / ".gitignore").exists()


def test_saving_a_session_writes_the_state_gitignore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SATSUMA_CONFIG_HOME", str(tmp_path / "home"))
    JSONLSessionStore().save(new_session(tmp_path, "m"))
    assert (tmp_path / ".satsuma" / ".gitignore").is_file()

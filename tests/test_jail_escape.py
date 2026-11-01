"""ワークスペース jail 逸脱テスト。"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote

import pytest

from satsumaestro.security.jail import WorkspaceJail
from satsumaestro.util.errors import PathEscapeError


def test_resolve_inside_ok(tmp_path: Path) -> None:
    jail = WorkspaceJail(tmp_path)
    assert jail.resolve("sub/file.txt") == (tmp_path / "sub" / "file.txt").resolve()
    assert jail.resolve(".") == tmp_path.resolve()


def test_resolve_nonexistent_inside_ok(tmp_path: Path) -> None:
    jail = WorkspaceJail(tmp_path)
    assert jail.resolve("not/created/yet.txt").is_relative_to(tmp_path.resolve())


def test_escape_dotdot(tmp_path: Path) -> None:
    jail = WorkspaceJail(tmp_path)
    with pytest.raises(PathEscapeError):
        jail.resolve("../outside.txt")


def test_escape_absolute(tmp_path: Path) -> None:
    jail = WorkspaceJail(tmp_path)
    with pytest.raises(PathEscapeError):
        jail.resolve("/etc/passwd")


def test_escape_symlink(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside_dir"
    outside.mkdir()
    root = tmp_path / "ws"
    root.mkdir()
    (root / "link").symlink_to(outside)
    jail = WorkspaceJail(root)
    with pytest.raises(PathEscapeError):
        jail.resolve("link/secret.txt")


def test_escape_url_encoded(tmp_path: Path) -> None:
    jail = WorkspaceJail(tmp_path)
    encoded = "%2E%2E%2Foutside.txt"
    with pytest.raises(PathEscapeError):
        jail.resolve(unquote(encoded))


def test_is_inside(tmp_path: Path) -> None:
    jail = WorkspaceJail(tmp_path)
    assert jail.is_inside("sub/ok.txt") is True
    assert jail.is_inside("../nope.txt") is False
    assert jail.is_inside("/etc/passwd") is False

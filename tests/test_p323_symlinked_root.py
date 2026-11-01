"""作業ディレクトリのパスがシンボリックリンクを通っても、走査ツールは中身を返す。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from satsumaestro.config.models import RuntimeConfig
from satsumaestro.security.jail import WorkspaceJail
from satsumaestro.tools.base import ToolContext
from satsumaestro.tools.builtins import list_directory as ld
from satsumaestro.tools.builtins.glob_search import GlobSearch
from satsumaestro.tools.builtins.grep_search import GrepSearch
from satsumaestro.tools.builtins.list_directory import ListDirectory


@pytest.fixture
def linked_root(tmp_path: Path) -> Path:
    """実体 ``real/`` を指すリンク ``link`` を作業ディレクトリとして返す（未解決のパス）。"""
    real = tmp_path / "real"
    (real / "src").mkdir(parents=True)
    (real / "src" / "app.py").write_text("def main():\n    pass\n", encoding="utf-8")
    (real / "README.md").write_text("# demo\n", encoding="utf-8")
    (real / ".gitignore").write_text("node_modules/\n", encoding="utf-8")
    (real / "node_modules" / "x").mkdir(parents=True)
    (real / "node_modules" / "x" / "index.js").write_text("x\n", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    assert link.resolve() == real.resolve() and link != link.resolve()
    return link


def _ctx(root: Path) -> ToolContext:
    return ToolContext(workspace_root=root, jail=WorkspaceJail(root), config=RuntimeConfig())


async def test_list_directory_lists_the_root_through_the_link(linked_root: Path) -> None:
    res = await ListDirectory().run(ListDirectory.Input(path="."), _ctx(linked_root))
    lines = res.output.splitlines()
    assert not res.is_error
    assert lines[:3] == ["src/", ".gitignore  14 B", "README.md  7 B"]
    assert "node_modules/" not in lines
    assert "(1 entries skipped by ignore rules)" in lines
    assert "(empty directory)" not in lines


async def test_list_directory_lists_a_subdirectory_through_the_link(linked_root: Path) -> None:
    res = await ListDirectory().run(ListDirectory.Input(path="src"), _ctx(linked_root))
    assert not res.is_error
    assert res.output.splitlines() == ["app.py  21 B"]


async def test_glob_and_grep_return_workspace_relative_paths_through_the_link(
    linked_root: Path,
) -> None:
    ctx = _ctx(linked_root)
    glob = await GlobSearch().run(GlobSearch.Input(pattern="**/*.py"), ctx)
    assert not glob.is_error
    assert "src/app.py" in glob.output and str(linked_root.resolve()) not in glob.output
    grep = await GrepSearch().run(GrepSearch.Input(pattern="def main"), ctx)
    assert not grep.is_error
    assert "src/app.py" in grep.output and str(linked_root.resolve()) not in grep.output


async def test_unreadable_entries_are_counted_instead_of_vanishing(
    linked_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """エントリ単位の失敗（壊れたリンク・権限）は飛ばすが、黙らずに件数を出す。"""
    real_scandir = os.scandir

    class _Broken:
        name = "broken"
        path = str(linked_root.resolve() / "broken")

        def is_dir(self) -> bool:
            raise OSError("stat failed")

    class _Iter:
        def __init__(self, inner: object) -> None:
            self._inner = inner

        def __enter__(self) -> _Iter:
            self._inner.__enter__()  # type: ignore[attr-defined]
            return self

        def __exit__(self, *a: object) -> None:
            self._inner.__exit__(*a)  # type: ignore[attr-defined]

        def __iter__(self):  # type: ignore[no-untyped-def]
            yield from self._inner  # type: ignore[misc]
            yield _Broken()

    monkeypatch.setattr(ld.os, "scandir", lambda p: _Iter(real_scandir(p)))
    res = await ListDirectory().run(ListDirectory.Input(path="."), _ctx(linked_root))
    lines = res.output.splitlines()
    assert not res.is_error
    assert lines[0] == "src/" and "broken" not in res.output
    assert "(1 entries could not be read)" in lines

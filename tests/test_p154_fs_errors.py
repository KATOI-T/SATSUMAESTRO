"""ファイル系ツールの失敗を「読める・安全」にする。"""

from __future__ import annotations

import errno
from pathlib import Path

import pytest

from satsumaestro.config.models import RuntimeConfig
from satsumaestro.engine.events import MessageStop, TextDelta, ToolUse
from satsumaestro.security.jail import WorkspaceJail
from satsumaestro.session.models import ToolResultBlock
from satsumaestro.tools import fs_errors
from satsumaestro.tools.base import ToolContext
from satsumaestro.tools.builtins.append_file import AppendFile
from satsumaestro.tools.builtins.edit_file import EditFile
from satsumaestro.tools.builtins.notebook_edit import NotebookEdit
from satsumaestro.tools.builtins.read_file import ReadFile
from satsumaestro.tools.builtins.write_file import WriteFile
from satsumaestro.util import jsonl
from satsumaestro.util.errors import PathEscapeError, ToolError
from satsumaestro.util.jsonl import write_atomic, write_text_atomic
from tests.test_p184_write_chunks import _runtime


def _ctx(root: Path) -> ToolContext:
    jail = WorkspaceJail(root)
    return ToolContext(workspace_root=jail.root, jail=jail, config=RuntimeConfig())


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "sub").mkdir()
    (root / "a.txt").write_text("hello\n", encoding="utf-8")
    return root


def _tmp_files(ws: Path) -> list[Path]:
    """作業ディレクトリの親から下にある ``*.tmp``（隠し名も含む）。"""
    return sorted(ws.parent.rglob("*.tmp"))


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (
            FileNotFoundError(errno.ENOENT, "No such file or directory", "/abs/x.md"),
            "file not found: x.md",
        ),
        (
            IsADirectoryError(errno.EISDIR, "Is a directory", "/abs"),
            f"path is a directory: x.md — {fs_errors.DIRECTORY_HINT}",
        ),
        (
            NotADirectoryError(errno.ENOTDIR, "Not a directory", "/abs"),
            "not a directory in the path: x.md",
        ),
        (PermissionError(errno.EACCES, "Permission denied", "/abs"), "permission denied: x.md"),
        (
            UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
            "not a UTF-8 text file: x.md (binary or another encoding)",
        ),
        (OSError(errno.EIO, "Input/output error", "/abs"), "Input/output error: x.md"),
    ],
)
def test_describe_uses_only_the_given_path(
    exc: OSError | UnicodeDecodeError, expected: str
) -> None:
    text = fs_errors.describe(exc, "x.md")
    assert text == expected
    assert "/abs" not in text and "Errno" not in text and "Error:" not in text


async def test_read_file_missing_is_short_and_relative(ws: Path) -> None:
    res = await ReadFile().run(ReadFile.Input(path="missing.md"), _ctx(ws))
    assert res.is_error and res.output == "file not found: missing.md"


@pytest.mark.parametrize("path", [".", "sub", "sub/"])
async def test_read_file_directory_says_so(ws: Path, path: str) -> None:
    res = await ReadFile().run(ReadFile.Input(path=path), _ctx(ws))
    assert res.is_error and res.output.startswith(f"path is a directory: {path}")
    assert "list_directory" in res.output and str(ws) not in res.output


async def test_read_file_range_read_fails_the_same_way(ws: Path) -> None:
    res = await ReadFile().run(ReadFile.Input(path="missing.md", offset=2, limit=3), _ctx(ws))
    assert res.is_error and res.output == "file not found: missing.md"
    res = await ReadFile().run(ReadFile.Input(path="sub", offset=2, limit=3), _ctx(ws))
    assert res.is_error and res.output.startswith("path is a directory: sub")


async def test_read_file_binary_is_reported(ws: Path) -> None:
    (ws / "bin.dat").write_bytes(b"\xff\xfe\x00\x01" * 8)
    res = await ReadFile().run(ReadFile.Input(path="bin.dat"), _ctx(ws))
    assert res.is_error and res.output.startswith("not a UTF-8 text file: bin.dat")


async def test_read_file_escape_is_still_a_path_escape(ws: Path) -> None:
    with pytest.raises(PathEscapeError):
        await ReadFile().run(ReadFile.Input(path="../outside.md"), _ctx(ws))


async def test_read_file_success_is_unchanged(ws: Path) -> None:
    res = await ReadFile().run(ReadFile.Input(path="a.txt"), _ctx(ws))
    assert not res.is_error and res.output == "hello\n"


async def test_edit_file_missing_and_directory(ws: Path) -> None:
    ctx = _ctx(ws)
    res = await EditFile().run(EditFile.Input(path="missing.py", old="a", new="b"), ctx)
    assert res.is_error and res.output == "file not found: missing.py"
    res = await EditFile().run(EditFile.Input(path="sub", old="a", new="b"), ctx)
    assert res.is_error and res.output.startswith("path is a directory: sub")
    assert _tmp_files(ws) == []


@pytest.mark.parametrize("path", [".", "sub"])
async def test_write_and_append_to_a_directory_leave_no_tmp(ws: Path, path: str) -> None:
    ctx = _ctx(ws)
    res = await WriteFile().run(WriteFile.Input(path=path, content="x\n"), ctx)
    assert res.is_error and res.output.startswith(f"path is a directory: {path}")
    assert str(ws) not in res.output
    assert not (ws.parent / f"{ws.name}.tmp").exists()
    assert _tmp_files(ws) == []
    res = await AppendFile().run(AppendFile.Input(path=path, content="x\n"), ctx)
    assert res.is_error and res.output.startswith(f"path is a directory: {path}")
    assert _tmp_files(ws) == []
    assert sorted(p.name for p in ws.iterdir()) == ["a.txt", "sub"]


async def test_write_file_success_is_unchanged(ws: Path) -> None:
    res = await WriteFile().run(WriteFile.Input(path="sub/new.txt", content="x\n"), _ctx(ws))
    assert not res.is_error and (ws / "sub" / "new.txt").read_text(encoding="utf-8") == "x\n"
    assert _tmp_files(ws) == []


def test_write_text_atomic_leaves_no_tmp_and_creates_parents(tmp_path: Path) -> None:
    target = tmp_path / "out" / "a.txt"
    write_text_atomic(target, "one\n")
    write_text_atomic(target, "two\n")
    assert target.read_text(encoding="utf-8") == "two\n"
    assert list(tmp_path.rglob("*.tmp")) == []


def test_write_text_atomic_refuses_a_directory_before_creating_tmp(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    with pytest.raises(IsADirectoryError):
        write_text_atomic(root, "x")
    assert not (tmp_path / "ws.tmp").exists()
    assert list(tmp_path.rglob("*.tmp")) == []


def test_write_text_atomic_removes_tmp_when_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "a.txt"

    def boom(src: object, dst: object) -> None:
        raise OSError(errno.EIO, "Input/output error")

    monkeypatch.setattr(jsonl.os, "replace", boom)
    with pytest.raises(OSError):
        write_text_atomic(target, "x")
    assert list(tmp_path.rglob("*.tmp")) == [] and not target.exists()


def test_write_atomic_jsonl_shares_the_same_guarantees(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    write_atomic(path, [{"n": 1}])
    write_atomic(path, [{"n": 2}])
    assert jsonl.read(path) == [{"n": 2}]
    assert list(tmp_path.rglob("*.tmp")) == []


async def test_runtime_returns_the_short_message_to_the_model(tmp_path: Path) -> None:
    rt = _runtime(
        [
            [ToolUse(id="r1", name="read_file", input={"path": "missing.md"}), MessageStop()],
            [TextDelta(text="ok"), MessageStop()],
        ],
        tmp_path,
    )
    await rt.run_turn("読んで")
    results = [b for m in rt.session.messages for b in m.blocks if isinstance(b, ToolResultBlock)]
    assert len(results) == 1 and results[0].is_error
    assert results[0].output.startswith("file not found: missing.md")
    assert "FileNotFoundError" not in results[0].output and str(tmp_path) not in results[0].output


async def test_notebook_edit_directory_is_a_tool_error(ws: Path) -> None:
    (ws / "nb.ipynb").mkdir()
    inp = NotebookEdit.Input(notebook_path="nb.ipynb", cell_id="c1", new_source="x")
    with pytest.raises(ToolError, match=r"path is a directory: nb\.ipynb"):
        await NotebookEdit().run(inp, _ctx(ws))

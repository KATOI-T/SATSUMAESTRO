"""読み取り専用の ``list_directory`` ツール。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from satsumaestro.cli.tool_display import describe, result_note
from satsumaestro.config.models import RuntimeConfig
from satsumaestro.engine.events import MessageStop, TextDelta, ToolUse
from satsumaestro.engine.subagent import EXPLORE_TOOLS
from satsumaestro.security.enforcer import PermissionMode, classify
from satsumaestro.security.jail import WorkspaceJail
from satsumaestro.security.policy import default_policy
from satsumaestro.session.models import ModeState, ToolResultBlock
from satsumaestro.tools import default_registry, tool_names
from satsumaestro.tools.base import ToolContext
from satsumaestro.tools.builtins.list_directory import ListDirectory, format_size
from satsumaestro.util.errors import PathEscapeError
from tests.fixtures.mock_provider import SequencedMockProvider
from tests.test_p12_planning import _make_runtime, new_session
from tests.test_p261_unknown_tool import _call, _result, _runtime

REG = default_registry()
CORE = set(REG.core_names())


def _ctx(root: Path) -> ToolContext:
    return ToolContext(workspace_root=root, jail=WorkspaceJail(root), config=RuntimeConfig())


async def _list(root: Path, path: str = ".", limit: int = 200) -> tuple[bool, list[str]]:
    res = await ListDirectory().run(ListDirectory.Input(path=path, limit=limit), _ctx(root))
    return res.is_error, res.output.splitlines()


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "src").mkdir()
    (root / "src" / "app.py").write_text("print('a')\n", encoding="utf-8")
    (root / "tests").mkdir()
    (root / "README.md").write_text("# demo\n", encoding="utf-8")
    (root / "big.bin").write_bytes(b"\0" * 2048)
    return root


async def test_lists_directories_then_files_with_sizes(ws: Path) -> None:
    err, lines = await _list(ws)
    assert not err
    assert lines == ["src/", "tests/", "README.md  7 B", "big.bin  2.0 KB"]


def test_format_size() -> None:
    assert format_size(0) == "0 B"
    assert format_size(1023) == "1023 B"
    assert format_size(1024) == "1.0 KB"
    assert format_size(1_536) == "1.5 KB"
    assert format_size(3 * 1024 * 1024) == "3.0 MB"
    assert format_size(2 * 1024**3) == "2.0 GB"


async def test_ignore_rules_skip_entries_and_count_them(ws: Path) -> None:
    (ws / ".gitignore").write_text("node_modules/\n*.log\n", encoding="utf-8")
    (ws / "node_modules" / "left-pad").mkdir(parents=True)
    (ws / "node_modules" / "left-pad" / "index.js").write_text("x\n", encoding="utf-8")
    (ws / "debug.log").write_text("noise\n", encoding="utf-8")
    (ws / ".git").mkdir()
    (ws / "__pycache__").mkdir()
    (ws / ".github").mkdir()
    err, lines = await _list(ws)
    assert not err
    assert lines[:3] == [".github/", "src/", "tests/"]
    assert ".gitignore  20 B" in lines
    assert not any(
        ln.startswith(("node_modules", "debug.log", ".git/", "__pycache__")) for ln in lines
    )
    assert lines[-1] == "(4 entries skipped by ignore rules)"


async def test_limit_truncates_with_the_glob_search_note(ws: Path) -> None:
    for name in ("c.txt", "d.txt", "e.txt"):
        (ws / name).write_text("x", encoding="utf-8")
    err, lines = await _list(ws, limit=3)
    assert not err
    assert lines == ["src/", "tests/", "README.md  7 B", "... (truncated at 3 entries)"]


async def test_empty_directory(ws: Path) -> None:
    err, lines = await _list(ws, "tests")
    assert not err and lines == ["(empty directory)"]


async def test_subdirectory_is_listed_relative_to_itself(ws: Path) -> None:
    err, lines = await _list(ws, "src/")
    assert not err and lines == ["app.py  11 B"]


async def test_file_path_says_so(ws: Path) -> None:
    err, lines = await _list(ws, "README.md")
    assert err and lines == ["path is a file: README.md — use read_file to read it"]


async def test_missing_directory(ws: Path) -> None:
    err, lines = await _list(ws, "nope/deeper")
    assert err and lines == ["directory not found: nope/deeper"]
    assert str(ws) not in lines[0]


async def test_path_escape_raises(ws: Path) -> None:
    with pytest.raises(PathEscapeError):
        await _list(ws, "../")


async def test_state_dir_and_escaping_symlink_are_hidden(ws: Path, tmp_path: Path) -> None:
    (ws / ".satsuma" / "sessions").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(outside, ws / "escape")
    err, lines = await _list(ws)
    assert not err
    assert not any(ln.startswith((".satsuma", "escape")) for ln in lines)
    assert lines[-1] == "(1 entries skipped by ignore rules)"


def test_is_a_core_read_only_tool() -> None:
    spec = ListDirectory.spec
    assert (
        spec.required_mode is PermissionMode.READ_ONLY and spec.parallel_safe and not spec.deferred
    )
    assert "list_directory" in {s["name"] for s in REG.schemas_for(set())}
    assert (
        classify("list_directory", {"path": "."}, PermissionMode.WORKSPACE_WRITE)
        is PermissionMode.READ_ONLY
    )
    labels = {r.label for r in default_policy(PermissionMode.READ_ONLY).allow}
    assert "default allow: list_directory" in labels
    assert "list_directory" in EXPLORE_TOOLS


async def test_runs_in_plan_mode_where_bash_ls_is_blocked(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    session = new_session(tmp_path, "mock-model")
    session.mode = ModeState(main="plan", sub=PermissionMode.WORKSPACE_WRITE)
    provider = SequencedMockProvider(
        [
            [
                ToolUse(id="l1", name="list_directory", input={"path": "."}),
                ToolUse(id="b1", name="bash", input={"command": "ls"}),
                MessageStop(),
            ],
            [TextDelta(text="done"), MessageStop()],
        ]
    )
    rt = _make_runtime(provider, tmp_path, session=session)
    await rt.run_turn("何がある？")
    by_id = {
        b.tool_use_id: b
        for m in rt.session.messages
        for b in m.blocks
        if isinstance(b, ToolResultBlock)
    }
    assert not by_id["l1"].is_error and by_id["l1"].output.startswith("a.txt  1 B")
    assert by_id["b1"].is_error and "plan mode" in by_id["b1"].output


@pytest.mark.parametrize("written", ["ls", "listdir", "list_dir", "list_files", "ListDirectory"])
async def test_listing_variants_run_the_real_tool(tmp_path: Path, written: str) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    rt = _runtime([_call(written, path="."), [TextDelta(text="ok"), MessageStop()]], tmp_path)
    summary = await rt.run_turn("一覧")
    block = _result(rt, "c1")
    assert not block.is_error and block.tool_name == "list_directory"
    assert block.output.startswith("a.txt  1 B")
    assert f"note: `{written}` is not a tool name; ran list_directory." in block.output
    assert (summary.tool_name_fixes, summary.unknown_tool_calls) == (1, 0)


def test_listing_names_put_list_directory_first() -> None:
    def names(name: str, inp: dict[str, object]) -> list[str]:
        return [c.name for c in tool_names.suggest(name, inp, REG, CORE).candidates]

    assert names("print_tree", {}) == ["list_directory", "glob_search"]
    assert names("list_workspace", {}) == ["list_directory", "glob_search"]
    assert names("readdir", {"directory": "."}) == ["list_directory", "glob_search"]
    assert tool_names.resolve_variant("list_workspace", REG) is None
    assert tool_names.resolve_variant("list-directory", REG) == "list_directory"


def test_display_label_and_count_note() -> None:
    call = ToolUse(id="1", name="list_directory", input={"path": "src"})
    shown = describe(call)
    assert shown.label == "一覧" and shown.items == ("src",) and not shown.at_start

    def note(output: str) -> str | None:
        return result_note(
            call, ToolResultBlock(tool_use_id="1", tool_name="list_directory", output=output)
        )

    assert note("src/\nREADME.md  7 B\n(2 entries skipped by ignore rules)") == "2 件"
    assert note("a.py  1 B\nb.py  1 B\n... (truncated at 2 entries)") == "2 件以上"
    assert note("(empty directory)") == "空のディレクトリ"

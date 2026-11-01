"""絶対パスで始まる入力がスラッシュコマンドとして食われ、会話に届かない。"""

from __future__ import annotations

from pathlib import Path

import pytest
from rich.console import Console

from satsumaestro.cli import repl as repl_mod
from satsumaestro.cli.commands import SlashDispatcher, _reads_as_path, parse_slash
from tests.test_repl_turn_failure import _FakeRender, _FakeRuntime, _FakeSession, _summary

ISSUE_LINE = "/Users/alice/project/dashboardを削除して"


@pytest.mark.parametrize(
    "line",
    [
        ISSUE_LINE,
        "/Users/alice/project/dashboard を削除して",
        "/tmpを削除して",
        "/help/foo を読んで",
        "/private/var/folders/x を消して",
    ],
)
def test_path_like_lines_pass_through_to_conversation(line: str) -> None:
    assert parse_slash(line) is None


@pytest.mark.skipif(not Path("/tmp").exists(), reason="/tmp が無い環境")
def test_existing_top_level_path_passes_through() -> None:
    assert parse_slash("/tmp を削除して") is None
    assert parse_slash("/tmp") is None


def test_typos_are_still_unknown_commands() -> None:
    cmd = parse_slash("/hlep")
    assert cmd is not None and (cmd.name, cmd.args, cmd.raw) == ("hlep", [], "hlep")
    cmd2 = parse_slash("/no_such_dir_xyz を消して")
    assert cmd2 is not None and cmd2.name == "no_such_dir_xyz"


def test_known_commands_are_unchanged() -> None:
    for line, name, args in (
        ("/help", "help", []),
        ("/mode plan", "mode", ["plan"]),
        ("/effort high", "effort", ["high"]),
        ("/Q", "q", []),
        ("/HELP", "help", []),
        ("/sub-mode", "sub-mode", []),
    ):
        cmd = parse_slash(line)
        assert cmd is not None and (cmd.name, cmd.args) == (name, args), line
    bare = parse_slash("/")
    assert bare is not None and bare.name == "help"
    assert parse_slash("hello /Users/x") is None


def test_raw_keeps_the_original_spelling() -> None:
    cmd = parse_slash("/HELP")
    assert cmd is not None and cmd.name == "help" and cmd.raw == "HELP"


def test_reads_as_path_never_raises() -> None:
    assert _reads_as_path("a\x00b") is False
    assert _reads_as_path("hlep") is False
    assert _reads_as_path("Users/x") is True
    assert _reads_as_path("tmpを消して") is True


async def test_unknown_command_message_shows_original_spelling() -> None:
    console = Console(record=True, width=120)
    cmd = parse_slash("/SATSUMAESTRO")
    assert cmd is not None
    await SlashDispatcher(_FakeRuntime([]), console).dispatch(cmd)  # type: ignore[arg-type]
    out = console.export_text()
    assert "未知のコマンドです: /SATSUMAESTRO" in out
    assert "/satsumaestro" not in out


async def test_repl_sends_path_lines_to_run_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = ["/Users/x/y を削除して", "/hlep", "/tmpを消して"]
    monkeypatch.setattr(repl_mod, "PromptSession", lambda **_: _FakeSession(lines))
    monkeypatch.setattr(repl_mod, "mode_prompt_fragments", lambda _: "> ")
    rt = _FakeRuntime([_summary("消しました"), _summary("消しました")])
    console = Console(record=True, width=120)
    render = _FakeRender()
    await repl_mod.run_repl(rt, None, render, console)  # type: ignore[arg-type]
    assert rt.calls == ["/Users/x/y を削除して", "/tmpを消して"]
    assert "未知のコマンドです: /hlep" in console.export_text()
    assert render.turns == 2

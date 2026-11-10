"""コマンドなしの起動は REPL にする。"""

from __future__ import annotations

from pathlib import Path

import pytest

from satsumaestro.cli import app as cli_app
from satsumaestro.cli.app import default_to_repl


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ([], ["repl"]),
        (["-m", "qwen3:8b"], ["repl", "-m", "qwen3:8b"]),
        (["--trust"], ["repl", "--trust"]),
        (["--cwd", "../app"], ["repl", "--cwd", "../app"]),
        (["--debug"], ["--debug", "repl"]),
        (["--debug", "-m", "qwen3:8b"], ["--debug", "repl", "-m", "qwen3:8b"]),
    ],
)
def test_no_command_starts_the_repl(args: list[str], expected: list[str]) -> None:
    assert default_to_repl(args, interactive=True) == expected


@pytest.mark.parametrize(
    "args",
    [
        ["repl"],
        ["repl", "-m", "qwen3:8b"],
        ["doctor"],
        ["prompt", "hi"],
        ["--help"],
        ["--version"],
        ["-V"],
        ["--debug", "doctor"],
        ["--debug", "--help"],
        ["promt", "hi"],
    ],
)
def test_commands_and_root_options_are_kept(args: list[str]) -> None:
    assert default_to_repl(args, interactive=True) == args


@pytest.mark.parametrize("args", [[], ["-m", "qwen3:8b"], ["--debug"]])
def test_without_a_terminal_nothing_is_added(args: list[str]) -> None:
    assert default_to_repl(args, interactive=False) == args


def _run_main(monkeypatch: pytest.MonkeyPatch, argv: list[str], *, terminal: bool) -> int:
    monkeypatch.setattr("sys.argv", argv)
    monkeypatch.setattr(cli_app, "_on_terminal", lambda: terminal)
    with pytest.raises(SystemExit) as done:
        cli_app.main()
    code = done.value.code
    return code if isinstance(code, int) else 0


def test_main_without_a_command_runs_the_repl(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, object] = {}

    async def fake_repl_session(model, workspace, allowed_tools, mode, *, trusted, lang):
        seen.update(model=model, workspace=workspace, trusted=trusted)

    monkeypatch.setattr(cli_app, "_repl_session", fake_repl_session)
    monkeypatch.setattr(cli_app, "_choose_screen_language", lambda cwd, *, trust: None)
    monkeypatch.setattr(cli_app, "_decide_trust", lambda cwd, *, trust, interactive: trust)

    code = _run_main(
        monkeypatch, ["stro", "-m", "qwen3:8b", "--cwd", str(tmp_path), "--trust"], terminal=True
    )

    assert code == 0
    assert seen == {"model": "qwen3:8b", "workspace": tmp_path, "trusted": True}


def test_main_without_a_terminal_prints_the_help(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def must_not_run(*args, **kwargs):
        raise AssertionError("the REPL started without a terminal")

    monkeypatch.setattr(cli_app, "_repl_session", must_not_run)

    _run_main(monkeypatch, ["satsumaestro"], terminal=False)

    out = capsys.readouterr().out
    assert "repl" in out and "doctor" in out

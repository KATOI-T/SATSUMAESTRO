"""bash 検証テスト。"""

from __future__ import annotations

import pytest

from satsumaestro.security.jail import WorkspaceJail
from satsumaestro.tools.base import ToolContext
from satsumaestro.tools.bash_validation import validate_bash
from satsumaestro.tools.builtins.bash import Bash
from satsumaestro.util.errors import PathEscapeError, ToolError


def test_redirect_escape_rejected(tmp_path) -> None:  # type: ignore[no-untyped-def]
    jail = WorkspaceJail(tmp_path)
    with pytest.raises(PathEscapeError):
        validate_bash("echo x > /etc/passwd", jail)


def test_redirect_escape_attached_rejected(tmp_path) -> None:  # type: ignore[no-untyped-def]
    jail = WorkspaceJail(tmp_path)
    with pytest.raises(PathEscapeError):
        validate_bash("echo x >/etc/passwd", jail)
    with pytest.raises(PathEscapeError):
        validate_bash("echo x 2>/etc/passwd", jail)


def test_redirect_inside_jail_ok(tmp_path) -> None:  # type: ignore[no-untyped-def]
    jail = WorkspaceJail(tmp_path)
    validate_bash("echo x > out.txt", jail)
    validate_bash("echo x >> logs/out.txt", jail)


def test_command_substitution_rejected(tmp_path) -> None:  # type: ignore[no-untyped-def]
    jail = WorkspaceJail(tmp_path)
    with pytest.raises(ToolError):
        validate_bash("echo `whoami`", jail)
    with pytest.raises(ToolError):
        validate_bash("echo $(whoami)", jail)


def test_pipe_and_operators_allowed(tmp_path) -> None:  # type: ignore[no-untyped-def]
    jail = WorkspaceJail(tmp_path)
    validate_bash("cat a.txt | grep x", jail)
    validate_bash("ls; echo done", jail)
    validate_bash("test -f a && echo ok || echo ng", jail)
    validate_bash("cmd 2>&1", jail)


@pytest.mark.parametrize(
    "command",
    [
        "grep -c '</body>' index.html",
        'grep -o "</item>" feed.xml',
        'for t in html body; do grep -c "<$t" f; grep -c "</$t" f; done',
        "python3 - <<'EOF'\n# it's fine\nprint(r'</div>')\nEOF",
        "grep -n 'if x > 3' app.py",
        "echo x # > /etc/passwd",
    ],
)
def test_quoted_operators_are_not_redirects(tmp_path, command: str) -> None:  # type: ignore[no-untyped-def]
    validate_bash(command, WorkspaceJail(tmp_path))


@pytest.mark.parametrize(
    "command",
    [
        "echo x > '/etc/x'",
        'echo x >"/etc/x y"',
        "echo x &> /etc/x",
        "echo x >&/etc/passwd",
        "cat <<EOF > /etc/out\nx\nEOF",
        "echo '<<X'\necho pwned > /etc/passwd",
        "cat <(echo y > /etc/z)",
        "cat < ../outside.txt",
    ],
)
def test_real_redirects_still_rejected(tmp_path, command: str) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(PathEscapeError):
        validate_bash(command, WorkspaceJail(tmp_path))


def test_device_targets_allowed(tmp_path) -> None:  # type: ignore[no-untyped-def]
    jail = WorkspaceJail(tmp_path)
    validate_bash("ls 2>/dev/null", jail)
    validate_bash("cmd > /dev/null 2>&1", jail)
    validate_bash("echo oops > /dev/stderr", jail)


def test_rejection_names_the_redirect(tmp_path) -> None:  # type: ignore[no-untyped-def]
    expected = r"redirect '>' target is outside the workspace: /etc/passwd"
    with pytest.raises(PathEscapeError, match=expected):
        validate_bash("echo x > /etc/passwd", WorkspaceJail(tmp_path))


def test_unbalanced_quote_is_parse_error(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ToolError, match="could not parse command"):
        validate_bash('echo "abc', WorkspaceJail(tmp_path))


async def test_bash_runs_inside_workspace(tool_ctx: ToolContext) -> None:
    result = await Bash().run(Bash.Input(command="echo hello > note.txt"), tool_ctx)
    assert result.is_error is False
    assert (tool_ctx.jail.root / "note.txt").read_text(encoding="utf-8").strip() == "hello"


async def test_bash_redirect_escape_blocked(tool_ctx: ToolContext) -> None:
    with pytest.raises(PathEscapeError):
        await Bash().run(Bash.Input(command="echo x > /etc/evil"), tool_ctx)

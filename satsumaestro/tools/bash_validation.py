"""bash コマンドの構文安全化。"""

from __future__ import annotations

from pathlib import Path

from satsumaestro.security.bash_lex import DEVICE_TARGETS, iter_redirects
from satsumaestro.security.bash_pipeline import Verdict, review_command
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.jail import WorkspaceJail
from satsumaestro.util.errors import PathEscapeError, ToolError

_ALLOW = Verdict(kind="allow")


def validate_bash(
    command: str,
    jail: WorkspaceJail,
    mode: PermissionMode | None = None,
    workspace: Path | None = None,
) -> Verdict:
    """``command`` を検証し、危険ならば例外を送出する。"""
    if "`" in command or "$(" in command:
        raise ToolError("command substitution (backtick / $()) is not allowed")

    try:
        redirects = list(iter_redirects(command))
    except ValueError as e:
        raise ToolError(f"could not parse command: {e}") from e

    for op, target in redirects:
        if target in DEVICE_TARGETS:
            continue
        if target.startswith("~") or "$" in target:
            raise PathEscapeError(
                f"redirect '{op}' target '{target}' is expanded by the shell, so it cannot be "
                "checked against the workspace; write to a plain path inside the workspace"
            )
        if not jail.is_inside(target):
            raise PathEscapeError(f"redirect '{op}' target is outside the workspace: {target}")

    if mode is None:
        return _ALLOW
    result = review_command(command, mode, workspace or jail.root)
    if result.kind == "block":
        raise ToolError(result.message)
    return result

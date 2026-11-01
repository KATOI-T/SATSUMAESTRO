"""権限モード enum と ``classify``。"""

from __future__ import annotations

from enum import StrEnum
from pathlib import PurePath
from typing import Any


class PermissionMode(StrEnum):
    """権限段階。"""

    READ_ONLY = "read-only"
    WORKSPACE_WRITE = "workspace-write"
    DANGER_FULL_ACCESS = "danger-full-access"
    PROMPT = "prompt"
    ALLOW = "allow"


_MODE_RANK: dict[PermissionMode, int] = {
    PermissionMode.READ_ONLY: 0,
    PermissionMode.WORKSPACE_WRITE: 1,
    PermissionMode.DANGER_FULL_ACCESS: 2,
}


def is_required_mode(mode: PermissionMode) -> bool:
    """``required_modes`` に書ける（順序を持つ）モードか。"""

    return mode in _MODE_RANK


def mode_le(required: PermissionMode, active: PermissionMode) -> bool:
    """``required <= active`` を順序ランクで判定する。"""

    if required not in _MODE_RANK or active not in _MODE_RANK:
        return False
    return _MODE_RANK[required] <= _MODE_RANK[active]


_WRITE_TOOLS = frozenset({"write_file", "edit_file", "append_file"})
_READ_TOOLS = frozenset(
    {
        "read_file",
        "glob_search",
        "list_directory",
        "grep_search",
        "web_fetch",
        "web_search",
        "rag_search",
        "todo_write",
        "enter_plan_mode",
        "exit_plan_mode",
    }
)


def classify(
    tool_name: str,
    inp: Any,
    spec_default: PermissionMode,
) -> PermissionMode:
    """具体呼び出しから必要権限モードを導出する。"""

    if tool_name == "bash":
        return PermissionMode.DANGER_FULL_ACCESS
    if tool_name == "lsp":
        action = str(getattr(inp, "action", ""))
        return PermissionMode.WORKSPACE_WRITE if action == "format" else PermissionMode.READ_ONLY
    if tool_name in _WRITE_TOOLS or tool_name == "notebook_edit":
        if _targets_git_dir(inp):
            return PermissionMode.DANGER_FULL_ACCESS
        if tool_name in _WRITE_TOOLS:
            return PermissionMode.WORKSPACE_WRITE
        return spec_default
    if tool_name in _READ_TOOLS:
        return PermissionMode.READ_ONLY
    return spec_default


def _targets_git_dir(inp: Any) -> bool:
    """書込ツール入力の対象パスが ``.git`` ディレクトリ配下（または ``.git`` 自体）か。"""

    for key in ("path", "notebook_path"):
        value = getattr(inp, key, None)
        if value is None and isinstance(inp, dict):
            value = inp.get(key)
        if isinstance(value, str) and value:
            if ".git" in PurePath(value).parts:
                return True
    return False

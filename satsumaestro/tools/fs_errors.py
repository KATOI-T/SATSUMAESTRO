"""ファイル系ツールの失敗文。"""

from __future__ import annotations

from satsumaestro.tools.base import ToolResult

DIRECTORY_HINT = "pass a file path, or use list_directory to list its contents"


def describe(e: OSError | UnicodeDecodeError, path: str) -> str:
    """OS の例外を、モデルが指定した ``path`` で短く言い直す。"""
    if isinstance(e, FileNotFoundError):
        return f"file not found: {path}"
    if isinstance(e, IsADirectoryError):
        return f"path is a directory: {path} — {DIRECTORY_HINT}"
    if isinstance(e, NotADirectoryError):
        return f"not a directory in the path: {path}"
    if isinstance(e, PermissionError):
        return f"permission denied: {path}"
    if isinstance(e, UnicodeDecodeError):
        return f"not a UTF-8 text file: {path} (binary or another encoding)"
    reason = e.strerror or type(e).__name__
    return f"{reason}: {path}"


def error_result(e: OSError | UnicodeDecodeError, path: str) -> ToolResult:
    """:func:`describe` の文面を ``is_error`` のツール結果にする。"""
    return ToolResult(output=describe(e, path), is_error=True)

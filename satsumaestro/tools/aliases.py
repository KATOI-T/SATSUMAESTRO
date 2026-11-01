"""ツール名の書き方の揺れを吸収する。"""

from __future__ import annotations

import re

ALIAS_TABLE: dict[str, str] = {
    "Read": "read_file",
    "Write": "write_file",
    "Edit": "edit_file",
    "Append": "append_file",
    "Glob": "glob_search",
    "Grep": "grep_search",
    "ls": "list_directory",
    "listdir": "list_directory",
    "list_files": "list_directory",
    "run_command": "bash",
    "run_command_output": "bash",
    "run_shell_command": "bash",
    "execute_command": "bash",
    "shell": "bash",
    "ToolSearch": "tool_search",
    "Skill": "skill",
    "LSP": "lsp",
}

_WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+")


def name_words(value: str) -> list[str]:
    """名前を小文字の語に分ける（``WebFetch`` → web, fetch・``read-file`` → read, file）。"""
    return [word.lower() for word in _WORD.findall(value)]


def name_key(value: str) -> str:
    """比べるためのキー。"""
    return "".join(name_words(value))


def alias_target(value: str) -> str | None:
    """呼び名がこのハーネスのどのツールを指すか。"""
    key = name_key(value)
    return next((tool for alias, tool in ALIAS_TABLE.items() if name_key(alias) == key), None)

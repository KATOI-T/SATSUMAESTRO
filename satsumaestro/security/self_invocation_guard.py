"""エージェント自身の入れ子起動を絶対禁止にする。"""

from __future__ import annotations

import os

from satsumaestro.security.deletion_guard import _segments, _tokenize, _unwrap

PROHIBITED_SELF_INVOCATION_PREFIX = "self-invocation prohibited:"

PROHIBITED_SELF_INVOCATION_GUIDANCE = (
    "You are SATSUMAESTRO: this session is the agent itself, so starting another copy of it is "
    "prohibited and cannot be done by any tool, regardless of permissions or user approval. "
    "A nested run would hold the GPU, hang on interactive input, and write over this session's "
    "own state. Do not retry it with another command, flag, or path spelling. Do the work with "
    "the tools you already have in this session; if a run of the CLI is really needed, ask the "
    "user to do it themselves in another terminal."
)

PROHIBITED_SELF_INVOCATION_REPEAT = (
    "This is the second blocked attempt in this turn: the check you are trying cannot be done "
    "from here in any form. If you need to know something, use the read-only tools "
    "(read_file, grep_search, list_directory) instead."
)

_SELF_COMMANDS = frozenset({"satsumaestro", "satsuma", "stro"})
_RUNNERS = frozenset({"uv", "uvx", "poetry", "pdm", "hatch", "pipx", "rye"})
_RUNNER_VALUE_OPTIONS = frozenset(
    {"--project", "--directory", "--with", "--python", "-p", "--from", "--index", "--extra"}
)
_PYTHON_NAMES = frozenset(
    {"python", "python3", "python3.12", "python3.13", "python3.14", "python3.15", "py"}
)


def check_bash(command: str) -> str | None:
    """bash コマンドが自身の起動なら理由（1 行目の文言）を返す。"""
    tokens = _tokenize(command)
    if tokens is None:
        return None
    for words in _segments(tokens):
        _privileged, name, args = _unwrap(words)
        found = _self_invocation(name, args)
        if found is not None:
            return (
                f"{PROHIBITED_SELF_INVOCATION_PREFIX} {found} cannot be started from inside itself"
            )
    return None


def _self_invocation(name: str, args: list[str]) -> str | None:
    """単純コマンドが自身の起動なら、画面に出す表記を返す。"""
    if _is_self(name):
        return _describe(name, args)
    if name in _PYTHON_NAMES:
        for i, tok in enumerate(args):
            if tok == "-m" and i + 1 < len(args) and _module_is_self(args[i + 1]):
                return f"python -m {args[i + 1]}"
            if not tok.startswith("-"):
                break
        return None
    if name in _RUNNERS:
        inner = _runner_target(args)
        return None if inner is None else _describe(inner[0], inner[1])
    return None


def _is_self(name: str) -> bool:
    """コマンド名（パス起動も basename で見る）が自身か。"""
    return os.path.basename(name) in _SELF_COMMANDS


def _module_is_self(module: str) -> bool:
    return module.split(".")[0] in _SELF_COMMANDS


def _runner_target(args: list[str]) -> tuple[str, list[str]] | None:
    """``uv run [オプション] <コマンド> …`` の ``<コマンド>`` 以降を返す。"""
    i = 0
    while i < len(args):
        tok = args[i]
        if tok in _RUNNER_VALUE_OPTIONS:
            i += 2
            continue
        if tok.startswith("-"):
            i += 1
            continue
        if tok in ("run", "tool", "exec", "shell"):
            i += 1
            continue
        rest = args[i + 1 :]
        if _is_self(tok):
            return tok, rest
        if os.path.basename(tok) in _PYTHON_NAMES:
            for j, inner in enumerate(rest):
                if inner == "-m" and j + 1 < len(rest) and _module_is_self(rest[j + 1]):
                    return tok, rest
                if not inner.startswith("-"):
                    break
        return None
    return None


def _describe(name: str, args: list[str]) -> str:
    """拒否の 1 行目に出す表記（``satsumaestro eval`` のようにサブコマンドまで）。"""
    base = os.path.basename(name)
    if base in _PYTHON_NAMES:
        for i, tok in enumerate(args):
            if tok == "-m" and i + 1 < len(args):
                return f"{base} -m {args[i + 1]}"
        return base
    sub = next((a for a in args if not a.startswith("-")), None)
    return f"{base} {sub}" if sub else base

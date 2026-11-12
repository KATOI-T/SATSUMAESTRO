"""子プロセスへ渡す環境変数から、ハーネスが使う秘密を外す純関数。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from fnmatch import fnmatchcase

PROVIDER_KEY_VARS: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "XAI_API_KEY",
    "SATSUMA_EMBED_API_KEY",
    "LM_API_TOKEN",
    "LLAMA_API_KEY",
)

DEFAULT_STRIP_GLOBS: tuple[str, ...] = ("*_API_KEY",)

DEFAULT_ALLOW_NAMES: tuple[str, ...] = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "TERM",
    "COLORTERM",
    "LANG",
    "LANGUAGE",
    "LC_*",
    "TZ",
    "TMPDIR",
    "TMP",
    "TEMP",
    "XDG_*",
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "COMSPEC",
    "PATHEXT",
    "WINDIR",
    "USERPROFILE",
    "APPDATA",
    "LOCALAPPDATA",
)


def _matches(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatchcase(name, pattern) for pattern in patterns)


def strip_secret_env(
    env: Mapping[str, str],
    *,
    extra_strip: Iterable[str] = (),
    keep: Iterable[str] = (),
    allow_list: bool = False,
) -> dict[str, str]:
    """``env`` から秘密の名前を外した写し（``allow_list`` では既定の名前と ``keep`` だけ）。"""
    strip_patterns = (*PROVIDER_KEY_VARS, *DEFAULT_STRIP_GLOBS, *extra_strip)
    keep_patterns = tuple(keep)
    out: dict[str, str] = {}
    for name, value in env.items():
        if not _matches(name, keep_patterns) and (
            _matches(name, strip_patterns)
            or (allow_list and not _matches(name, DEFAULT_ALLOW_NAMES))
        ):
            continue
        out[name] = value
    return out

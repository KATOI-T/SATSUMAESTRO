"""言語→サーバ既定表・拡張子→言語解決。"""

from __future__ import annotations

import shutil
from pathlib import Path

EXTENSION_LANGUAGES: dict[str, str] = {
    **dict.fromkeys((".py", ".pyi"), "python"),
    **dict.fromkeys((".ts", ".tsx", ".mts", ".cts"), "typescript"),
    **dict.fromkeys((".js", ".jsx", ".mjs", ".cjs"), "javascript"),
    ".go": "go",
}

DEFAULT_SERVERS: dict[str, tuple[tuple[str, ...], ...]] = {
    "python": (("pyright-langserver", "--stdio"), ("pylsp",)),
    "typescript": (("typescript-language-server", "--stdio"),),
    "javascript": (("typescript-language-server", "--stdio"),),
    "go": (("gopls",),),
}


def language_for_path(path: Path) -> str | None:
    """拡張子から言語を解決する。"""
    return EXTENSION_LANGUAGES.get(path.suffix.lower())


def resolve_server_command(
    language: str, overrides: dict[str, list[str]], *, path: str | None = None
) -> list[str] | None:
    """言語のサーバ起動コマンドを解決する（``path`` は探す PATH・既定は ``os.environ``）。"""
    override = overrides.get(language)
    if override:
        return override if shutil.which(override[0], path=path) else None
    for candidate in DEFAULT_SERVERS.get(language, ()):
        if shutil.which(candidate[0], path=path):
            return list(candidate)
    return None

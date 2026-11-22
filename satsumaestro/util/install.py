"""SATSUMAESTRO の入れ方（brew・uv tool・pipx・pip・editable）と、更新のコマンドを判定する。"""

from __future__ import annotations

import json
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Literal

PACKAGE = "satsumaestro"

InstallKind = Literal["brew", "uv", "pipx", "pip", "editable"]

_COMMANDS: dict[InstallKind, tuple[str, ...]] = {
    "brew": ("brew", "upgrade", PACKAGE),
    "uv": ("uv", "tool", "upgrade", PACKAGE),
    "pipx": ("pipx", "upgrade", PACKAGE),
    "pip": ("pip", "install", "-U", PACKAGE),
    "editable": (),
}
_MARKERS: tuple[tuple[InstallKind, str], ...] = (
    ("uv", "uv-receipt.toml"),
    ("pipx", "pipx_metadata.json"),
)


@dataclass(frozen=True)
class Install:
    """入れ方と更新のコマンド（``executable`` が無ければコマンドは出すだけ）。"""

    kind: InstallKind
    command: tuple[str, ...]
    executable: str | None = None

    @property
    def runnable(self) -> bool:
        return self.executable is not None

    @property
    def argv(self) -> list[str]:
        """回すときの引数（先頭は見つけた実行ファイル）。"""
        if self.executable is None:
            return []
        return [self.executable, *self.command[1:]]

    @property
    def shown(self) -> str:
        """利用者に見せるコマンド。"""
        return " ".join(self.command)


def detect_install(
    prefix: Path | None = None,
    *,
    editable: bool | None = None,
    windows: bool | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> Install:
    """動いている SATSUMAESTRO の入れ方を判定する（引数は試験のために外から渡せる）。"""
    if editable is None:
        editable = installed_editable()
    if editable:
        return Install("editable", _COMMANDS["editable"])
    root = Path(sys.prefix if prefix is None else prefix)
    try:
        root = root.resolve()
    except OSError:
        pass
    if windows is None:
        windows = sys.platform == "win32"
    brew_prefix = _homebrew_prefix(root)
    if brew_prefix is not None:
        brew = brew_prefix / "bin" / "brew"
        found = str(brew) if brew.is_file() else which("brew")
        return Install("brew", _COMMANDS["brew"], found)
    for kind, marker in _MARKERS:
        if (root / marker).is_file():
            # 走っている .exe は Windows では置き換えられない
            found = None if windows else which(_COMMANDS[kind][0])
            return Install(kind, _COMMANDS[kind], found)
    return Install("pip", _COMMANDS["pip"])


def _homebrew_prefix(root: Path) -> Path | None:
    parts = root.parts
    for i in range(len(parts) - 1):
        if parts[i] == "Cellar" and parts[i + 1] == PACKAGE:
            return Path(*parts[:i])
    return None


def installed_editable(name: str = PACKAGE) -> bool:
    """editable で入れたか（配布物の記録が無いときも、ソースから動いている扱い）。"""
    try:
        text = metadata.distribution(name).read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return True
    return is_editable(text)


def is_editable(direct_url: str | None) -> bool:
    """``direct_url.json`` の中身が editable の印を持つか。"""
    if not direct_url:
        return False
    try:
        data = json.loads(direct_url)
    except ValueError:
        return False
    dir_info = data.get("dir_info") if isinstance(data, dict) else None
    return isinstance(dir_info, dict) and dir_info.get("editable") is True


__all__ = [
    "PACKAGE",
    "Install",
    "InstallKind",
    "detect_install",
    "installed_editable",
    "is_editable",
]

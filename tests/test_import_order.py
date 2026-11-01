"""どのパッケージを最初に import しても、循環 import で落ちない。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import satsumaestro

_ROOT = Path(satsumaestro.__file__).resolve().parent


def _packages() -> list[str]:
    return sorted(
        ".".join(init.parent.relative_to(_ROOT.parent).parts) for init in _ROOT.rglob("__init__.py")
    )


def test_every_package_imports_first() -> None:
    packages = _packages()
    assert "satsumaestro.tools" in packages and "satsumaestro.hooks" in packages
    procs = {
        name: subprocess.Popen(
            [sys.executable, "-c", f"import {name}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        for name in packages
    }
    failed: dict[str, str] = {}
    for name, proc in procs.items():
        _, err = proc.communicate(timeout=120)
        if proc.returncode != 0:
            failed[name] = (err.strip().splitlines() or [""])[-1]
    assert not failed, failed

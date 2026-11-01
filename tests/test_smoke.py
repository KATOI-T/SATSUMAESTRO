"""スモークテスト。"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

SUBPACKAGES = [
    "satsumaestro",
    "satsumaestro.cli",
    "satsumaestro.cli.app",
    "satsumaestro.engine",
    "satsumaestro.providers",
    "satsumaestro.tools",
    "satsumaestro.tools.builtins",
    "satsumaestro.security",
    "satsumaestro.session",
    "satsumaestro.config",
    "satsumaestro.prompt",
    "satsumaestro.mcp",
    "satsumaestro.rag",
    "satsumaestro.hooks",
    "satsumaestro.util",
]

REQUIRED_ENV_KEYS = [
    "OLLAMA_HOST",
    "OLLAMA_MODEL",
    "SATSUMA_EMBED_MODEL",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
]


@pytest.mark.parametrize("module", SUBPACKAGES)
def test_smoke_import(module: str) -> None:
    """各サブパッケージが例外なく import できる。"""
    assert importlib.import_module(module) is not None


def test_smoke_version() -> None:
    """``satsumaestro --version`` がサブプロセスで 0 終了しバージョンを出力する。"""
    from satsumaestro import __version__

    result = subprocess.run(
        ["satsumaestro", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert __version__ in result.stdout


def test_smoke_version_via_module() -> None:
    """`python -m satsumaestro.cli.app --version` でも疎通する（install 非依存）。"""
    from satsumaestro import __version__

    result = subprocess.run(
        [sys.executable, "-m", "satsumaestro.cli.app", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert __version__ in result.stdout


def test_env_example_keys(repo_root: Path) -> None:
    """``.env.example`` に必須キーが揃っている。"""
    env_example = repo_root / ".env.example"
    assert env_example.exists(), ".env.example が存在しない"
    text = env_example.read_text(encoding="utf-8")
    for key in REQUIRED_ENV_KEYS:
        assert f"{key}=" in text, f"{key} が .env.example に無い"

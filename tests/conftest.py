"""pytest 共有フィクスチャ。"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from satsumaestro.providers.detect import reset_used_models

_SESSION_CONFIG_HOME = "_satsuma_session_config_home"


def pytest_configure(config: pytest.Config) -> None:
    """テストのモジュールを読み込む前に、ユーザー設定の置き場所を空の場所にする。"""
    home = tempfile.mkdtemp(prefix="satsuma-config-home-")
    setattr(config, _SESSION_CONFIG_HOME, home)
    os.environ["SATSUMA_CONFIG_HOME"] = home


def pytest_unconfigure(config: pytest.Config) -> None:
    home = getattr(config, _SESSION_CONFIG_HOME, None)
    if home is not None:
        shutil.rmtree(home, ignore_errors=True)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """リポジトリルート（tests/ の親）を返す。"""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def tool_ctx(tmp_path: Path):  # type: ignore[no-untyped-def]
    """ツール実行用の ``ToolContext``（jail=tmp_path・既定設定）を返す。"""
    from satsumaestro.config.models import RuntimeConfig
    from satsumaestro.security.jail import WorkspaceJail
    from satsumaestro.tools.base import ToolContext

    jail = WorkspaceJail(tmp_path)
    return ToolContext(workspace_root=tmp_path, jail=jail, config=RuntimeConfig())


@pytest.fixture(autouse=True)
def _hermetic_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """テストを開発環境の LLM エンドポイントから切り離す。"""
    for key in ("OLLAMA_HOST", "OLLAMA_MODEL", "OPENAI_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SATSUMA_LOG_FILE", "off")


_GIT_LOCAL_ENV_VARS = (
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_CONFIG",
    "GIT_CONFIG_PARAMETERS",
    "GIT_CONFIG_COUNT",
    "GIT_OBJECT_DIRECTORY",
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_IMPLICIT_WORK_TREE",
    "GIT_GRAFT_FILE",
    "GIT_INDEX_FILE",
    "GIT_NO_REPLACE_OBJECTS",
    "GIT_REPLACE_REF_BASE",
    "GIT_PREFIX",
    "GIT_SHALLOW_FILE",
    "GIT_COMMON_DIR",
)


@pytest.fixture(autouse=True)
def _hermetic_git_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """テストの中の git を、テストが作ったリポジトリだけに向ける。"""
    for key in _GIT_LOCAL_ENV_VARS:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def _no_used_ollama_models() -> Iterator[None]:
    """このプロセスが Ollama で使ったモデルの記録を、テストごとに空にする。"""
    reset_used_models()
    yield
    reset_used_models()


@pytest.fixture(autouse=True)
def _hermetic_config_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """ユーザー設定の置き場所（信頼の記録 trusted.json を含む）を、テストごとの空の場所にする。"""
    monkeypatch.setenv("SATSUMA_CONFIG_HOME", str(tmp_path_factory.mktemp("config-home")))


@pytest.fixture(autouse=True)
def _japanese_screen() -> Iterator[None]:
    """画面の言語を日本語に固定する（既存のテストは日本語の画面の文で確かめている）。"""
    from satsumaestro.i18n.text import pinned_ui_language, use_language, use_screen_language

    use_language(None)
    use_screen_language(None)
    with pinned_ui_language("ja"):
        yield
    use_language(None)
    use_screen_language(None)

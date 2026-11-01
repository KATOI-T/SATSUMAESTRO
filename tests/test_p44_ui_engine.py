"""engine ほかの、画面に出る通知と例外の英語の画面。"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import AsyncExitStack
from pathlib import Path

import pytest

from satsumaestro.config.loader import ConfigLoader
from satsumaestro.config.models import McpServerConfig
from satsumaestro.engine.context_budget import ResultCut
from satsumaestro.i18n.text import pinned_ui_language
from satsumaestro.mcp.client import open_session
from satsumaestro.providers.notices import ProviderNotice, describe
from satsumaestro.tools import default_registry
from satsumaestro.util.errors import (
    ConfigError,
    ContextOverflowError,
    ContextTruncatedError,
    PlanExitLoopError,
    ProviderTimeoutError,
    RepetitionLoopError,
    RetriesExhaustedError,
    ToolCallingUnsupportedError,
    ToolError,
    UnknownToolLoopError,
)

_JAPANESE = re.compile(r"[　-ヿ一-鿿＀-￯]")


@pytest.fixture(autouse=True)
def _english_screen() -> Iterator[None]:
    with pinned_ui_language("en"):
        yield


def _english(text: str) -> str:
    assert not _JAPANESE.search(text), text
    return text


def _settings(tmp_path: Path, text: str) -> ConfigLoader:
    (tmp_path / ".satsuma").mkdir()
    (tmp_path / ".satsuma" / "settings.json").write_text(text, encoding="utf-8")
    return ConfigLoader(tmp_path, tmp_path / "home", trusted=True)


def test_broken_settings_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as caught:
        _settings(tmp_path, "{oops").load()
    assert _english(str(caught.value)).startswith("Invalid JSON in the settings file: ")


def test_settings_that_are_not_an_object(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="The settings file must be a JSON object"):
        _settings(tmp_path, "[1, 2]").load()


def test_settings_that_fail_validation(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as caught:
        _settings(tmp_path, '{"language": {"fixed": "zz"}}').load()
    message = str(caught.value)
    assert message.startswith("Invalid settings: ")
    assert "Unknown language: 'zz' (available: ja, en)" in message


def test_invalid_environment_values(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SATSUMA_WEB_SEARCH_PROVIDER", "bing")
    with pytest.raises(ConfigError) as caught:
        ConfigLoader(tmp_path, tmp_path / "home", trusted=True).load()
    assert _english(str(caught.value)).startswith(
        "Invalid value for the environment variable SATSUMA_WEB_SEARCH_PROVIDER: 'bing' (use "
    )


def test_result_cut_lines() -> None:
    read = ResultCut("read_file", 2_000, 90_000, 6_000, lines=(1, 400), total_lines=20_000)
    assert read.describe() == (
        "The read_file result was large, so only lines 1–400 of 20,000 (~2,000 tokens) were "
        "passed on (the limit for the results of one reply is ~6,000 tokens)"
    )
    bash = ResultCut("bash", 2_000, 90_000, 6_000)
    assert bash.describe() == (
        "The bash result was large, so only the first ~2,000 tokens (of ~90,000) were passed "
        "on (the limit for the results of one reply is ~6,000 tokens)"
    )


def test_provider_notices() -> None:
    assert describe(ProviderNotice(kind="fallback", from_model="a", to_model="b")) == (
        "'a' is not responding, so switching to 'b'"
    )
    assert describe(ProviderNotice(kind="rewrite")) == (
        "The model's output kept breaking, so it was asked to rewrite it"
    )
    assert describe(ProviderNotice(kind="retry", attempt=1, limit=2, reason="tool_parse")) == (
        "The model's output broke midway, so generating it again (1/2)"
    )
    assert describe(ProviderNotice(kind="retry", attempt=2, limit=9, delay_s=3.2)) == (
        "The model's reply failed, so trying again in 3 s (2/9)"
    )


def test_exceptions_shown_on_the_screen() -> None:
    assert str(RetriesExhaustedError(2, 25.4, last_error="boom")) == (
        "Gave up after the retry limit (2 attempts / 25.4s): boom"
    )
    assert (
        str(ProviderTimeoutError(3, 600.0))
        == "Ran out of the API time budget (3 attempts / 600.0s)"
    )
    assert str(ToolCallingUnsupportedError("tiny", "read_file")) == (
        "Model 'tiny' does not support native tool calling (it wrote a read_file call as text). "
        "Use a model that supports tools (e.g. llama3.1:8b / llama3.3:70b / mistral-large)."
    )
    with pinned_ui_language("ja"):
        assert str(RetriesExhaustedError(2, 25.4)) == "リトライ上限に到達（2 回試行 / 25.4s）"


def test_exceptions_kept_as_english_records() -> None:
    with pinned_ui_language("ja"):
        records = [
            str(RepetitionLoopError(7, "loop", recovered_before=True)),
            str(PlanExitLoopError(3)),
            str(UnknownToolLoopError("run_turn", 3)),
            str(ContextOverflowError(140_000, 131_072, dropped_request=True)),
            str(ContextTruncatedError(150_000, 3_000)),
        ]
    for record in records:
        _english(record)
    assert (
        records[2] == "stopped the turn: the unknown tool name run_turn was called 3 times in a row"
    )


def test_allowed_tools_errors() -> None:
    registry = default_registry()
    with pytest.raises(ToolError) as caught:
        registry.resolve_allowed_tools([" , "])
    assert _english(str(caught.value)).startswith("--allowed-tools has no tool names")
    with pytest.raises(ToolError) as caught:
        registry.resolve_allowed_tools(["read_file,nope"])
    text = _english(str(caught.value))
    assert text.startswith("--allowed-tools has unknown tool names: nope\nTool names: ")
    assert "\nOther names: " in text


async def test_mcp_settings_errors() -> None:
    async with AsyncExitStack() as stack:
        with pytest.raises(ConfigError, match="A stdio MCP server needs a command"):
            await open_session(stack, McpServerConfig(transport="stdio"))
        with pytest.raises(ConfigError, match="A streamable-http MCP server needs a url"):
            await open_session(stack, McpServerConfig(transport="streamable-http"))

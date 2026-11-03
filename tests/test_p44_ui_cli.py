"""CLI のヘルプ・doctor・コマンドの出力と、起動時の案内の英語の画面。"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from rich.console import Console
from rich.table import Table
from typer.testing import CliRunner

from satsumaestro.cli import app as cli_app
from satsumaestro.config.models import ProviderConfig, RuntimeConfig
from satsumaestro.engine.context_fit import (
    ContextFitResult,
    ModelCheck,
    check_reachable,
    missing_model_notice,
)
from satsumaestro.i18n.text import pinned_ui_language
from satsumaestro.providers.base import ProviderKind
from satsumaestro.providers.detect import ModelContextInfo

REPO = Path(__file__).resolve().parents[1]
_JAPANESE = re.compile(r"[　-ヿ一-鿿＀-￯]")
_ROOT = "http://10.0.0.5:11434"

runner = CliRunner()


def _english(text: str) -> str:
    assert not _JAPANESE.search(text), text
    return text


_DUMP_HELP = """
import json
from typer.main import get_command
from typer.testing import CliRunner
from satsumaestro.cli.app import app

runner = CliRunner()
out = {}
for name in ["", *sorted(get_command(app).commands)]:
    args = [name, "--help"] if name else ["--help"]
    out[name or "(root)"] = runner.invoke(app, args, terminal_width=200).output
print(json.dumps(out))
"""


def _help_in(lang: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in ("LC_ALL", "LC_MESSAGES")}
    env.update(LANG=lang, COLUMNS="200", SATSUMA_LOG_FILE="off")
    done = subprocess.run(
        [sys.executable, "-c", _DUMP_HELP],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    helps: dict[str, str] = json.loads(done.stdout)
    return helps


def test_help_follows_the_environment_language() -> None:
    english = _help_in("en_US.UTF-8")
    for name, text in english.items():
        assert not _JAPANESE.search(text), name
    flat = {name: " ".join(text.split()) for name, text in english.items()}
    assert "SATSUMAESTRO — an agentic coding CLI" in flat["(root)"]
    assert "Interactive REPL." in flat["(root)"]
    assert "Show the version and exit." in flat["(root)"]
    assert "Trust this directory's settings for this run only (not recorded)" in flat["repl"]
    assert "Fix the reply language for this run" in flat["prompt"]
    assert "Check the health of the settings, keys, provider and policy" in flat["doctor"]
    assert "The git root is recorded" in flat["trust"]
    assert "Agents to evaluate (repeatable)" in flat["eval"]
    japanese = _help_in("ja_JP.UTF-8")
    assert "SATSUMAESTRO — エージェント型コーディング CLI" in japanese["(root)"]
    assert "対話 REPL。" in japanese["repl"]


@pytest.fixture
def _english_screen() -> Iterator[None]:
    with pinned_ui_language("en"):
        yield


@pytest.mark.usefixtures("_english_screen")
def test_doctor_in_english(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for k in ("ANTHROPIC_API_KEY", "XAI_API_KEY", "OLLAMA_HOST", "OPENAI_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("COLUMNS", "300")
    (tmp_path / ".satsuma").mkdir()
    (tmp_path / ".satsuma" / "settings.json").write_text(
        json.dumps(
            {
                "provider": {"default_model": "m", "base_url": f"{_ROOT}/v1"},
                "web_fetch": {"enabled": True},
                "web_search": {"provider": "duckduckgo"},
                "mcp_servers": {"demo": {"transport": "stdio", "command": "echo"}},
                "permission": {"bash_auto_allow": "off"},
            }
        ),
        encoding="utf-8",
    )
    result = runner.invoke(cli_app.app, ["doctor", "--cwd", str(tmp_path), "--trust"])
    assert result.exit_code == 0, result.output
    out = _english(result.output)
    assert "--trust (this run only, not recorded)" in out
    assert "1 server: demo(stdio) (status with `mcp list`)" in out
    assert "not set (provider defaults)" in out
    assert "off (read-only bash is also confirmed every time)" in out
    assert "the unofficial HTML version sometimes goes silent on bot detection" in out
    assert "confirmed only the first time in a session; internal destinations" in out
    assert "(--probe to query the context length and residency)" in out
    assert "no index (run `rag ingest`)" in out


@pytest.mark.usefixtures("_english_screen")
async def test_web_search_probe_in_english(monkeypatch: pytest.MonkeyPatch) -> None:
    from satsumaestro.tools.builtins import web_search as web_search_mod

    def answer(response: httpx.Response) -> None:
        monkeypatch.setattr(
            web_search_mod,
            "_make_client",
            lambda timeout_s: httpx.AsyncClient(
                transport=httpx.MockTransport(lambda req: response)
            ),
        )

    answer(httpx.Response(200, json={"results": [{"title": "t", "url": "https://e.com"}]}))
    assert await cli_app._probe_web_search("searxng", 1.0, "http://s/search") == (
        "ok",
        "reachable (1 result)",
    )
    answer(httpx.Response(200, text="<html></html>"))
    status, detail = await cli_app._probe_web_search("searxng", 1.0, "http://s/search")
    assert status == "warn" and detail.startswith("it does not answer in JSON")
    answer(httpx.Response(202))
    status, detail = await cli_app._probe_web_search("duckduckgo", 1.0, "http://d/html")
    assert detail.startswith("bot detection in progress (HTTP 202)")


@pytest.mark.usefixtures("_english_screen")
def test_context_fit_lines_in_english() -> None:
    def fit(info: ModelContextInfo, effective: int) -> str:
        return ContextFitResult("m", info, 120_000, effective).describe()

    assert fit(ModelContextInfo(native=131_072, source="show.model_info"), 104_857) == (
        "m: context_length=131072 [the model's native window (effective num_ctx unknown)] "
        "→ compact_token_threshold 120000 → 104857"
    )
    assert fit(ModelContextInfo(effective=262_144, source="ps", loaded=True), 120_000) == (
        "m: context_length=262144 [effective num_ctx (loaded)] / "
        "compact_token_threshold=120000 (unchanged)"
    )
    assert fit(ModelContextInfo(), 120_000) == (
        "m: could not detect the context length (keeping the threshold 120000)"
    )


@pytest.mark.usefixtures("_english_screen")
def test_context_fit_brackets_are_not_read_as_markup(monkeypatch: pytest.MonkeyPatch) -> None:
    info = ModelContextInfo(native=131_072, source="show.model_info")
    fit = ContextFitResult("m", info, 120_000, 104_857)
    console = Console(record=True, width=300)

    async def probe(cfg: RuntimeConfig, model: str) -> ContextFitResult:
        return fit

    monkeypatch.setattr(cli_app, "_probe_ollama", probe)
    monkeypatch.setenv("OLLAMA_HOST", _ROOT)
    cfg = RuntimeConfig(provider=ProviderConfig(base_url=f"{_ROOT}/v1"))
    table = Table()
    for _ in range(3):
        table.add_column()
    for row in cli_app._ollama_rows(cfg, "m", ProviderKind.OPENAI, probe=True):
        table.add_row(*row)
    console.print(table)
    assert "[the model's native window" in console.export_text()


@pytest.mark.usefixtures("_english_screen")
def test_missing_model_in_english() -> None:
    many = ["gpt-oss:120b"] + [f"m{i}:1b" for i in range(12)]
    notice = missing_model_notice("gpt-oss:12b", ModelCheck(_ROOT, missing=True, available=many))
    assert notice.text().splitlines() == [
        f"Model 'gpt-oss:12b' was not found on the Ollama server ({_ROOT}).",
        "Similar names: gpt-oss:120b",
        "Available: gpt-oss:120b, " + ", ".join(f"m{i}:1b" for i in range(9)) + " and 3 more",
        "Run `ollama pull gpt-oss:12b` to download it, or check --model / OLLAMA_MODEL in .env.",
    ]
    text = cli_app.missing_model_text(notice, prefix="error: ")

    def styles(part: str) -> set[str]:
        start = text.plain.index(part)
        return {str(s.style) for s in text.spans if s.start <= start < s.end}

    assert text.plain.splitlines()[-1] == (
        "Run ollama pull gpt-oss:12b to download it, or check --model / OLLAMA_MODEL in .env."
    )
    assert styles("error: Model") == {"red"} and styles("Similar names:") == {"yellow"}
    assert styles("ollama pull gpt-oss:12b") == {"yellow"} and styles("Run ") == set()
    repl = missing_model_notice("x", ModelCheck(_ROOT, missing=True, available=[]), startup=False)
    assert repl.text().splitlines()[-1] == (
        "Run `ollama pull x` to download it, or check the model name."
    )


@pytest.mark.usefixtures("_english_screen")
async def test_unreachable_ollama_in_english() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    cfg = RuntimeConfig(provider=ProviderConfig(base_url=f"{_ROOT}/v1"))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        problem = await check_reachable(cfg, "gpt-oss:120b", {}, client=client)
    assert problem == (
        f"Cannot reach Ollama: {_ROOT} (ConnectError: Connection refused). Check that the "
        "server is running and check OLLAMA_HOST in .env (OLLAMA_* in .env take precedence "
        "over the shell's environment variables)"
    )


@pytest.mark.usefixtures("_english_screen")
def test_command_messages_in_english(tmp_path: Path) -> None:
    def run(*args: str) -> str:
        return _english(runner.invoke(cli_app.app, [*args, "--cwd", str(tmp_path)]).output)

    assert "resume needs a session_id" in run("session", "resume")
    assert "Session not found: nope" in run("session", "resume", "nope")
    assert "No sessions." in run("session", "list")
    assert "No MCP servers are configured (mcp_servers)." in run("mcp", "list", "--trust")
    assert "search needs a search query" in run("rag", "search")
    assert "There is no index yet. Create it with `satsumaestro rag ingest`." in run(
        "rag", "stats", "--trust"
    )
    (tmp_path / ".satsuma").mkdir()
    (tmp_path / ".satsuma" / "settings.json").write_text(
        '{"rag": {"enabled": false}}', encoding="utf-8"
    )
    assert "RAG is disabled in the settings (rag.enabled = false)." in run(
        "rag", "stats", "--trust"
    )
    bad = runner.invoke(cli_app.app, ["prompt", "hi", "--lang", "zz"])
    assert bad.exit_code == 2 and "Unknown language: 'zz'" in " ".join(bad.output.split())

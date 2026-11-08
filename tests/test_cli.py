"""CLI サブコマンドのテスト。"""

from __future__ import annotations

import io
import json
from pathlib import Path

import httpx
from rich.console import Console
from typer.testing import CliRunner

from satsumaestro.cli import app as cli_app
from satsumaestro.cli.render import RichRenderer
from satsumaestro.cli.repl import InteractivePrompter
from satsumaestro.engine.events import MessageStop, TextDelta, Thinking, ToolUse
from satsumaestro.engine.turn import DenyPrompter
from satsumaestro.providers import local as local_servers
from satsumaestro.providers.local import _probe as _real_probe
from satsumaestro.providers.registry import expand_model_name
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.session.models import ToolResultBlock
from tests.fixtures.mock_provider import MockProvider

runner = CliRunner()


def _patch_provider(monkeypatch, script, captured=None):
    def fake_build_provider(config, model, env):
        if captured is not None:
            captured["model"] = model
        return MockProvider(script)

    monkeypatch.setattr(cli_app, "build_provider", fake_build_provider)


def test_cli_prompt_json(monkeypatch, tmp_path: Path) -> None:
    _patch_provider(monkeypatch, [TextDelta(text="hi there"), MessageStop()])
    result = runner.invoke(
        cli_app.app,
        ["prompt", "hello", "--output-format", "json", "--cwd", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["final_text"] == "hi there"
    assert payload["stop_reason"] == "end_turn"
    assert payload["iterations"] == 1
    assert "usage" in payload


def test_cli_prompt_text(monkeypatch, tmp_path: Path) -> None:
    _patch_provider(monkeypatch, [TextDelta(text="plain answer"), MessageStop()])
    result = runner.invoke(cli_app.app, ["prompt", "hello", "--cwd", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "plain answer" in result.output


def test_cli_prompt_invalid_output_format_errors(tmp_path: Path) -> None:
    result = runner.invoke(
        cli_app.app, ["prompt", "hi", "--output-format", "jsno", "--cwd", str(tmp_path)]
    )
    assert result.exit_code != 0


def test_cli_flags_model_alias(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, str] = {}
    _patch_provider(monkeypatch, [TextDelta(text="ok"), MessageStop()], captured=captured)
    result = runner.invoke(
        cli_app.app,
        ["prompt", "hi", "--model", "opus", "--output-format", "json", "--cwd", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert captured["model"] == expand_model_name("opus")


def test_cli_prompt_text_tool_call_rejected(monkeypatch, tmp_path: Path) -> None:
    leaked = '{"name": "write_file", "arguments": {"path": "a", "content": "x"}}'
    _patch_provider(monkeypatch, [TextDelta(text=leaked), MessageStop()])
    result = runner.invoke(cli_app.app, ["prompt", "make a file", "--cwd", str(tmp_path)])
    assert result.exit_code != 0
    assert "tool calling" in result.output


def test_key_status_openai_compat_endpoints_ok() -> None:
    from satsumaestro.cli.app import _key_status
    from satsumaestro.config.models import ProviderConfig, RuntimeConfig

    cfg, model = RuntimeConfig(), "gpt-oss:20b"
    with_url = RuntimeConfig(provider=ProviderConfig(base_url="http://h/v1"))
    assert _key_status(cfg, model, {"OLLAMA_HOST": "h"}) == (
        "ok",
        "OLLAMA_HOST (http://h:11434/v1)",
    )
    assert _key_status(cfg, model, {"OPENAI_BASE_URL": "http://h/v1"})[0] == "ok"
    assert _key_status(with_url, model, {}) == ("ok", "provider.base_url (http://h/v1)")
    assert _key_status(cfg, model, {"OPENAI_API_KEY": "sk"})[0] == "ok"
    assert _key_status(cfg, model, {})[0] == "warn"


def test_key_status_shows_the_endpoint_in_use() -> None:
    """設定の base_url が OLLAMA_HOST より先に効くなら、base_url を出す。"""
    from satsumaestro.cli.app import _key_status
    from satsumaestro.config.models import ProviderConfig, RuntimeConfig

    cfg = RuntimeConfig(provider=ProviderConfig(base_url="http://127.0.0.1:9931/v1"))
    status = _key_status(cfg, "gpt-oss-20b", {"OLLAMA_HOST": "192.168.1.5"})
    assert status == ("ok", "provider.base_url (http://127.0.0.1:9931/v1)")


def test_key_status_follows_the_chosen_provider() -> None:
    from satsumaestro.cli.app import _key_status
    from satsumaestro.config.models import ProviderConfig, RuntimeConfig

    def status(kind: str, env: dict[str, str]) -> tuple[str, str]:
        cfg = RuntimeConfig(provider=ProviderConfig(kind=kind, base_url="http://h:11434/v1"))
        return _key_status(cfg, "gpt-oss:20b", env)

    assert status("anthropic", {})[0] == "warn"
    assert status("openai", {"OPENAI_API_KEY": "k"}) == (
        "ok",
        "OPENAI_API_KEY set (https://api.openai.com/v1)",
    )
    assert status("openai", {})[0] == "warn"
    assert status("llamacpp", {}) == ("ok", "provider.base_url (http://h:11434/v1)")
    no_url = RuntimeConfig(provider=ProviderConfig(kind="lmstudio"))
    assert _key_status(no_url, "m", {})[0] == "warn"
    assert _key_status(no_url, "m", {"LMSTUDIO_HOST": "10.0.0.2"}) == (
        "ok",
        "LMSTUDIO_HOST (http://10.0.0.2:1234/v1)",
    )


def test_key_status_anthropic_needs_key() -> None:
    from satsumaestro.cli.app import _key_status
    from satsumaestro.config.models import RuntimeConfig

    model = "claude-opus-5-5"
    assert _key_status(RuntimeConfig(), model, {})[0] == "warn"
    assert _key_status(RuntimeConfig(), model, {"ANTHROPIC_API_KEY": "x"})[0] == "ok"


def test_doctor_offline(monkeypatch, tmp_path: Path) -> None:
    keys = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY", "OLLAMA_HOST", "OPENAI_BASE_URL")
    for k in keys:
        monkeypatch.delenv(k, raising=False)
    result = runner.invoke(cli_app.app, ["doctor", "--cwd", str(tmp_path), "--trust"])
    assert result.exit_code == 0, result.output
    assert "provider" in result.output
    assert "policy" in result.output


def test_models_and_version() -> None:
    assert runner.invoke(cli_app.app, ["models"]).exit_code == 0
    v = runner.invoke(cli_app.app, ["--version"])
    assert v.exit_code == 0


def test_mcp_list_and_rag_commands(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SATSUMA_CONFIG_HOME", str(tmp_path / "home"))
    r1 = runner.invoke(cli_app.app, ["mcp", "--cwd", str(tmp_path)])
    assert r1.exit_code == 0 and "mcp_servers" in r1.output
    r_bad = runner.invoke(cli_app.app, ["mcp", "bogus", "--cwd", str(tmp_path)])
    assert r_bad.exit_code == 1
    r_bad_rag = runner.invoke(cli_app.app, ["rag", "bogus", "--cwd", str(tmp_path)])
    assert r_bad_rag.exit_code == 1
    r2 = runner.invoke(cli_app.app, ["rag", "stats", "--cwd", str(tmp_path)])
    assert r2.exit_code == 1 and "rag ingest" in r2.output
    assert not (tmp_path / ".satsuma" / "rag.db").exists()
    r3 = runner.invoke(cli_app.app, ["rag", "search", "--cwd", str(tmp_path)])
    assert r3.exit_code == 1


def test_interactive_prompter_yes_no(monkeypatch) -> None:
    p = InteractivePrompter(Console())
    call = ToolUse(id="1", name="bash", input={"command": "ls"})
    monkeypatch.setattr(p.console, "input", lambda *a, **k: "y")
    assert p.decide(call, PermissionMode.DANGER_FULL_ACCESS).decision == "allow"
    monkeypatch.setattr(p.console, "input", lambda *a, **k: "n")
    assert p.decide(call, PermissionMode.DANGER_FULL_ACCESS).decision == "deny"
    monkeypatch.setattr(p.console, "input", lambda *a, **k: "")
    assert p.decide(call, PermissionMode.DANGER_FULL_ACCESS).decision == "deny"


def test_deny_prompter_always_deny() -> None:
    out = DenyPrompter().decide(ToolUse(id="1", name="bash", input={}), PermissionMode.READ_ONLY)
    assert out.decision == "deny"


def test_render_nontty_plain() -> None:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=80)
    r = RichRenderer(console=console)
    r.start_turn()
    r.on_assistant_event(Thinking(text="考え中"))
    r.on_assistant_event(TextDelta(text="# Title"))
    r.on_tool_start(ToolUse(id="1", name="read_file", input={"path": "x"}))
    r.on_tool_result(
        ToolResultBlock(tool_use_id="1", tool_name="read_file", output="body", is_error=False)
    )
    r.on_assistant_event(TextDelta(text="answer"))
    r.end_turn()
    out = buf.getvalue()
    assert "Title" in out and "・読み込み" in out and "- x" in out and "answer" in out


def test_doctor_ollama_rows_offline(monkeypatch, tmp_path: Path) -> None:
    """Ollama 判定時は --probe 無しでもオフラインで設定行を出す（ネットワーク不使用）。"""
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    monkeypatch.setenv("SATSUMA_CONFIG_HOME", str(tmp_path / "home"))
    (tmp_path / ".satsuma").mkdir()
    (tmp_path / ".satsuma" / "settings.json").write_text(
        '{"provider": {"default_model": "m", "sampling": {"temperature": 0.3}}}', encoding="utf-8"
    )
    result = runner.invoke(cli_app.app, ["doctor", "--cwd", str(tmp_path), "--trust"])
    assert result.exit_code == 0, result.output
    assert "keep_alive=30m" in result.output
    assert "unload_on_exit=on" in result.output
    assert "temperature=0.3" in result.output
    assert "--probe" in result.output


def test_doctor_probe_non_ollama_is_noop(monkeypatch, tmp_path: Path) -> None:
    """--probe でも Ollama・LM Studio と判定されなければ照会しない（汎用の OpenAI 互換先）。"""
    for k in ("ANTHROPIC_API_KEY", "XAI_API_KEY", "OLLAMA_HOST"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8080/v1")
    monkeypatch.setenv("SATSUMA_CONFIG_HOME", str(tmp_path / "home"))
    _mock_web_search_http(monkeypatch, lambda req: httpx.Response(200, text="<html></html>"))
    result = runner.invoke(cli_app.app, ["doctor", "--cwd", str(tmp_path), "--probe", "--trust"])
    assert result.exit_code == 0, result.output
    assert "local server" in result.output
    assert "provider.local.server で固定可" in result.output
    assert "--probe で確かめる" not in result.output


def test_doctor_probe_llamacpp_rows(monkeypatch, tmp_path: Path) -> None:
    """llama.cpp は --probe でサーバの形・1 スロットの窓・窓を広げる助言を出す。"""
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY", "OLLAMA_HOST"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:9931/v1")
    monkeypatch.setenv("SATSUMA_CONFIG_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("COLUMNS", "400")
    (tmp_path / ".satsuma").mkdir()
    (tmp_path / ".satsuma" / "settings.json").write_text(
        '{"provider": {"default_model": "gpt-oss-20b", "local": {"server": "llamacpp"}}}',
        encoding="utf-8",
    )
    props = {
        "default_generation_settings": {"params": {}, "n_ctx": 4096},
        "total_slots": 4,
        "model_alias": "gpt-oss-20b",
        "is_sleeping": False,
        "chat_template_caps": {"supports_tools": False},
    }

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/props":
            return httpx.Response(200, json=props)
        return httpx.Response(404, json={})

    real = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda *a, **k: real(transport=httpx.MockTransport(handler))
    )
    result = runner.invoke(cli_app.app, ["doctor", "--cwd", str(tmp_path), "--probe", "--trust"])
    assert result.exit_code == 0, result.output
    assert "1 つのモデル: gpt-oss-20b / スロット 4" in result.output
    assert "1 スロットの窓（llama.cpp）" in result.output
    assert "llama-server を `-c 16384` 以上で起動し直します" in result.output
    slots = "`--parallel 4` で `--kv-unified` を付けていないなら `-c 65536` 以上です"
    assert slots in result.output
    assert "llamacpp tools" in result.output
    assert "チャットテンプレートはツールを受け付けない" in result.output


def test_doctor_probe_says_a_local_server_is_down(monkeypatch, tmp_path: Path) -> None:
    """ローカルに見える接続先につながらなければ、--probe でそう伝える。"""
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY", "OLLAMA_HOST"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9931/v1")
    monkeypatch.setenv("SATSUMA_CONFIG_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("COLUMNS", "400")
    monkeypatch.setattr(local_servers, "_probe", _real_probe)

    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=req)

    real = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda *a, **k: real(transport=httpx.MockTransport(handler))
    )
    result = runner.invoke(cli_app.app, ["doctor", "--cwd", str(tmp_path), "--probe"])
    assert result.exit_code == 0, result.output
    assert "ローカルのサーバに接続できません: http://127.0.0.1:9931" in result.output


async def test_startup_warns_when_the_llamacpp_template_takes_no_tools(monkeypatch) -> None:
    """llama-server のテンプレートがツールを受け付けなければ、起動時に知らせる。"""
    from satsumaestro.config.models import LocalConfig, ProviderConfig, RuntimeConfig
    from satsumaestro.engine.provider_switch import tools_note

    caps = {"supports_tools": False}

    def handler(req: httpx.Request) -> httpx.Response:
        body = {"default_generation_settings": {"n_ctx": 8192}, "total_slots": 1}
        return httpx.Response(200, json={**body, "chat_template_caps": caps})

    real = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda *a, **k: real(transport=httpx.MockTransport(handler))
    )
    cfg = RuntimeConfig(
        provider=ProviderConfig(
            base_url="http://127.0.0.1:9931/v1", local=LocalConfig(server="llamacpp")
        )
    )
    note = await tools_note(cfg, "gpt-oss-20b", {})
    assert note is not None and "--chat-template-file" in note
    caps["supports_tools"] = True
    assert await tools_note(cfg, "gpt-oss-20b", {}) is None


def _mock_web_search_http(monkeypatch, handler) -> None:
    from satsumaestro.tools.builtins import web_search as web_search_mod

    monkeypatch.setattr(
        web_search_mod,
        "_make_client",
        lambda timeout_s: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def test_doctor_web_search_offline_default(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SATSUMA_CONFIG_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("SATSUMA_WEB_FETCH_ENABLED", raising=False)
    monkeypatch.setenv("COLUMNS", "200")
    result = runner.invoke(cli_app.app, ["doctor", "--cwd", str(tmp_path), "--trust"])
    assert result.exit_code == 0, result.output
    assert "無効（web_search.provider 未設定）" in result.output
    assert "無効（web_fetch.enabled / SATSUMA_WEB_FETCH_ENABLED 未設定）" in result.output


def test_doctor_web_fetch_enabled(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SATSUMA_CONFIG_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("COLUMNS", "200")
    (tmp_path / ".satsuma").mkdir()
    (tmp_path / ".satsuma" / "settings.json").write_text(
        '{"web_fetch": {"enabled": true}}', encoding="utf-8"
    )
    result = runner.invoke(cli_app.app, ["doctor", "--cwd", str(tmp_path), "--trust"])
    assert result.exit_code == 0, result.output
    assert "セッションの最初の 1 回だけ確認" in result.output and "は拒否" in result.output


def test_doctor_web_search_duckduckgo_offline(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SATSUMA_CONFIG_HOME", str(tmp_path / "home"))
    (tmp_path / ".satsuma").mkdir()
    (tmp_path / ".satsuma" / "settings.json").write_text(
        '{"web_search": {"provider": "duckduckgo"}}', encoding="utf-8"
    )
    result = runner.invoke(cli_app.app, ["doctor", "--cwd", str(tmp_path), "--trust"])
    assert result.exit_code == 0, result.output
    assert "duckduckgo" in result.output
    assert "SearXNG" in result.output


def test_doctor_web_search_searxng_missing_base_url(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SATSUMA_CONFIG_HOME", str(tmp_path / "home"))
    (tmp_path / ".satsuma").mkdir()
    (tmp_path / ".satsuma" / "settings.json").write_text(
        '{"web_search": {"provider": "searxng"}}', encoding="utf-8"
    )
    result = runner.invoke(cli_app.app, ["doctor", "--cwd", str(tmp_path), "--trust"])
    assert result.exit_code == 0, result.output
    assert "searxng: base_url" in result.output
    assert "未稼働" in result.output


def test_doctor_probe_web_search_searxng(monkeypatch, tmp_path: Path) -> None:
    for k in ("ANTHROPIC_API_KEY", "XAI_API_KEY", "OLLAMA_HOST"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:9999/v1")
    monkeypatch.setenv("SATSUMA_CONFIG_HOME", str(tmp_path / "home"))
    (tmp_path / ".satsuma").mkdir()
    (tmp_path / ".satsuma" / "settings.json").write_text(
        '{"web_search": {"provider": "searxng", "base_url": "http://localhost:8080"}}',
        encoding="utf-8",
    )
    ok = httpx.Response(200, json={"results": [{"title": "t", "url": "https://e.com"}]})
    _mock_web_search_http(monkeypatch, lambda req: ok)
    result = runner.invoke(cli_app.app, ["doctor", "--cwd", str(tmp_path), "--probe", "--trust"])
    assert result.exit_code == 0, result.output
    assert "疎通 OK" in result.output

    _mock_web_search_http(monkeypatch, lambda req: httpx.Response(200, text="<html></html>"))
    result = runner.invoke(cli_app.app, ["doctor", "--cwd", str(tmp_path), "--probe", "--trust"])
    assert result.exit_code == 0, result.output
    assert "json" in result.output

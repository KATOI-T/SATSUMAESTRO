"""子プロセスへ渡す環境からプロバイダの鍵を外す。"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from satsumaestro.cli import app as cli_app
from satsumaestro.config.models import HookConfig, LspConfig, RuntimeConfig, SandboxConfig
from satsumaestro.engine.events import MessageStop, TextDelta, ToolUse
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.engine.subagent import RuntimeFactory, resolve_subagent_type
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.jail import WorkspaceJail
from satsumaestro.security.policy import PermissionOutcome, default_policy
from satsumaestro.security.sandbox import (
    FS_MODE_VAR,
    Isolation,
    bash_environment,
    child_environment,
    isolation_for,
)
from satsumaestro.security.secret_env import PROVIDER_KEY_VARS, strip_secret_env
from satsumaestro.session.redact import secret_values
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import ToolContext, default_registry
from satsumaestro.tools.base import ToolResult
from satsumaestro.tools.builtins.bash import Bash
from satsumaestro.tools.lsp.client import LspClient
from satsumaestro.tools.lsp.registry import LanguageServers
from satsumaestro.util.errors import ToolError
from tests.fixtures.mock_provider import SequencedMockProvider

_BASE = {
    "PATH": "/usr/bin",
    "HOME": "/home/u",
    "ANTHROPIC_API_KEY": "sk-ant-x",
    "OPENAI_API_KEY": "sk-o",
    "XAI_API_KEY": "xai-1",
    "SATSUMA_EMBED_API_KEY": "e",
    "LM_API_TOKEN": "t",
    "LLAMA_API_KEY": "l",
    "FOO_API_KEY": "f",
    "GH_TOKEN": "g",
    "api_key": "lower",
}


def test_strip_defaults() -> None:
    out = strip_secret_env(_BASE)
    for name in PROVIDER_KEY_VARS:
        assert name not in out
    assert "FOO_API_KEY" not in out
    assert out["PATH"] == "/usr/bin" and out["HOME"] == "/home/u"
    assert out["GH_TOKEN"] == "g"
    assert out["api_key"] == "lower"
    assert _BASE["ANTHROPIC_API_KEY"] == "sk-ant-x"


def test_strip_extra_and_keep() -> None:
    out = strip_secret_env(_BASE, extra_strip=["GH_TOKEN", "*_SECRET"], keep=["OPENAI_API_KEY"])
    assert "GH_TOKEN" not in out
    assert out["OPENAI_API_KEY"] == "sk-o"
    assert "ANTHROPIC_API_KEY" not in out


def test_keep_glob_restores_all() -> None:
    out = strip_secret_env(_BASE, keep=["*_API_KEY", "LM_API_TOKEN"])
    assert all(name in out for name in PROVIDER_KEY_VARS)
    assert out["FOO_API_KEY"] == "f"


def test_isolation_carries_config_only() -> None:
    cfg = SandboxConfig(env_strip=["GH_TOKEN"], env_keep=["OPENAI_API_KEY"])
    isolation = isolation_for(cfg)
    assert isolation.env_strip == ("GH_TOKEN",) and isolation.env_keep == ("OPENAI_API_KEY",)
    env = bash_environment(_BASE, isolation=isolation)
    assert "GH_TOKEN" not in env and "ANTHROPIC_API_KEY" not in env
    assert env["OPENAI_API_KEY"] == "sk-o"
    plain = bash_environment(_BASE, isolation=Isolation())
    assert "ANTHROPIC_API_KEY" not in plain and "FOO_API_KEY" not in plain
    assert plain["PATH"] == "/usr/bin"
    assert plain[FS_MODE_VAR] == "workspace-only"


_WIDE = {
    **_BASE,
    "LANG": "ja_JP.UTF-8",
    "LC_ALL": "C",
    "XDG_CONFIG_HOME": "/home/u/.config",
    "TMPDIR": "/tmp/x",
    "SATSUMA_KEEP_ME": "1",
    "DATABASE_URL": "postgres://u:pw@db/x",
}


def test_the_strip_mode_is_the_default() -> None:
    assert SandboxConfig().env_mode == "strip"
    assert isolation_for(SandboxConfig()).env_mode == "strip"
    assert child_environment(SandboxConfig(), _WIDE)["DATABASE_URL"] == "postgres://u:pw@db/x"


def test_the_allow_list_passes_only_the_listed_names() -> None:
    out = strip_secret_env(_WIDE, allow_list=True)
    assert out == {
        "PATH": "/usr/bin",
        "HOME": "/home/u",
        "LANG": "ja_JP.UTF-8",
        "LC_ALL": "C",
        "XDG_CONFIG_HOME": "/home/u/.config",
        "TMPDIR": "/tmp/x",
    }


def test_the_allow_list_takes_keep_and_strip() -> None:
    out = strip_secret_env(
        _WIDE, allow_list=True, keep=["GH_TOKEN", "OPENAI_API_KEY"], extra_strip=["TMPDIR"]
    )
    assert out["GH_TOKEN"] == "g" and out["OPENAI_API_KEY"] == "sk-o"
    assert "TMPDIR" not in out and "ANTHROPIC_API_KEY" not in out
    assert "DATABASE_URL" not in out


def test_the_allow_list_reaches_bash_and_other_children() -> None:
    cfg = SandboxConfig(env_mode="allow-list", env_keep=["SATSUMA_KEEP_ME"])
    env = bash_environment(_WIDE, isolation=isolation_for(cfg))
    assert "DATABASE_URL" not in env and "GH_TOKEN" not in env
    assert env["SATSUMA_KEEP_ME"] == "1" and env[FS_MODE_VAR] == "workspace-only"
    child = child_environment(cfg, _WIDE)
    assert "DATABASE_URL" not in child and child["SATSUMA_KEEP_ME"] == "1"


async def test_bash_takes_the_base_environment(tmp_path: Path) -> None:
    cfg = RuntimeConfig(sandbox=SandboxConfig(env_mode="allow-list", env_keep=["SATSUMA_KEEP_ME"]))
    base = {**_WIDE, "PATH": "/usr/bin:/bin"}
    ctx = ToolContext(workspace_root=tmp_path, jail=WorkspaceJail(tmp_path), config=cfg, env=base)
    res = await Bash().run(
        Bash.Input(command='printf "%s|%s|%s" "$DATABASE_URL" "$SATSUMA_KEEP_ME" "$LANG"'), ctx
    )
    assert res.is_error is False
    assert res.output == "|1|ja_JP.UTF-8"


class _Allow:
    def decide(
        self, call: ToolUse, required: PermissionMode, warning: str | None = None
    ) -> PermissionOutcome:
        return PermissionOutcome(decision="allow", reason="test")


def _runtime(
    tmp_path: Path, provider: object, cfg: RuntimeConfig, **kwargs: Any
) -> ConversationRuntime:
    return ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=provider,  # type: ignore[arg-type]
        tools=default_registry(),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=cfg,
        **kwargs,
    )


async def test_the_runtime_gives_bash_the_base_environment(tmp_path: Path) -> None:
    cfg = RuntimeConfig(sandbox=SandboxConfig(env_mode="allow-list", env_keep=["SATSUMA_KEEP_ME"]))
    command = 'printf "%s|%s" "$DATABASE_URL" "$SATSUMA_KEEP_ME"'
    provider = SequencedMockProvider(
        [
            [ToolUse(id="b1", name="bash", input={"command": command}), MessageStop()],
            [TextDelta(text="done"), MessageStop()],
        ]
    )
    rt = _runtime(tmp_path, provider, cfg, env={**_WIDE, "PATH": "/usr/bin:/bin"})
    await rt.run_turn("show it", prompter=_Allow())
    block = rt.session.messages[2].blocks[0]
    assert block.is_error is False
    assert block.output == "|1"


def test_a_child_runtime_takes_the_factory_environment(tmp_path: Path) -> None:
    cfg = RuntimeConfig()
    factory = RuntimeFactory(
        provider=SequencedMockProvider([[MessageStop()]]),  # type: ignore[arg-type]
        model="mock-model",
        config=cfg,
        tools=default_registry(),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        hooks=HookRunner(HookConfig(), tmp_path),
        cwd=tmp_path,
        provider_builder=lambda m: SequencedMockProvider([[MessageStop()]]),  # type: ignore[arg-type,return-value]
        env=_WIDE,
    )
    child = factory.build_child(preset=resolve_subagent_type("general"), depth=1)
    assert child._ctx.env == _WIDE
    assert _runtime(tmp_path, factory.provider, cfg, runtime_factory=factory)._ctx.env == _WIDE
    assert _runtime(tmp_path, factory.provider, cfg)._ctx.env is None


def test_child_environment_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("SATSUMA_KEEP_ME", "1")
    env = child_environment()
    assert "ANTHROPIC_API_KEY" not in env and env["SATSUMA_KEEP_ME"] == "1"
    kept = child_environment(SandboxConfig(env_keep=["ANTHROPIC_API_KEY"]))
    assert kept["ANTHROPIC_API_KEY"] == "sk-ant-test"


def test_the_lmstudio_token_is_kept_from_children_and_records() -> None:
    env = {"LM_API_TOKEN": "lm-token-value", "PATH": "/usr/bin"}
    assert strip_secret_env(env) == {"PATH": "/usr/bin"}
    assert secret_values(env) == ("lm-token-value",)


def test_the_llamacpp_key_is_kept_from_children_and_records() -> None:
    env = {"LLAMA_API_KEY": "llama-key-value", "PATH": "/usr/bin"}
    assert strip_secret_env(env) == {"PATH": "/usr/bin"}
    assert secret_values(env) == ("llama-key-value",)


def test_provider_key_vars_cover_source() -> None:
    """ソースが読む `*_API_KEY`・`*_API_TOKEN` の名前が一覧から漏れていない。"""
    root = Path(__file__).resolve().parents[1] / "satsumaestro"
    pattern = re.compile(r"\b([A-Z][A-Z0-9_]*_API_(?:KEY|TOKEN))\b")
    found: set[str] = set()
    for path in root.rglob("*.py"):
        found.update(pattern.findall(path.read_text(encoding="utf-8")))
    assert found == set(PROVIDER_KEY_VARS)


async def test_bash_child_does_not_see_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("FOO_API_KEY", "x")
    monkeypatch.setenv("SATSUMA_KEEP_ME", "1")
    ctx = ToolContext(workspace_root=tmp_path, jail=WorkspaceJail(tmp_path), config=RuntimeConfig())
    res = await Bash().run(
        Bash.Input(
            command='printf "%s|%s|%s" "$ANTHROPIC_API_KEY" "$FOO_API_KEY" "$SATSUMA_KEEP_ME"'
        ),
        ctx,
    )
    assert res.is_error is False
    assert res.output == "||1"


async def test_bash_env_keep(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("FOO_API_KEY", "x")
    cfg = RuntimeConfig(sandbox=SandboxConfig(env_keep=["FOO_API_KEY"]))
    ctx = ToolContext(workspace_root=tmp_path, jail=WorkspaceJail(tmp_path), config=cfg)
    res = await Bash().run(
        Bash.Input(command='printf "%s|%s" "$ANTHROPIC_API_KEY" "$FOO_API_KEY"'), ctx
    )
    assert res.output == "|x"


def test_bash_input_cannot_restore_keys() -> None:
    """ツールの入力に env_keep は無い（モデルが鍵を取り戻せない）。"""
    assert "env_keep" not in Bash.Input.model_fields
    assert "env_strip" not in Bash.Input.model_fields


async def test_hook_child_does_not_see_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("SATSUMA_KEEP_ME", "1")
    dump = tmp_path / "env.json"
    script = tmp_path / "hook.py"
    script.write_text(
        "import json, os, sys\njson.load(sys.stdin)\n"
        f"open({str(dump)!r}, 'w').write(json.dumps(dict(os.environ)))\nprint('{{}}')\n",
        encoding="utf-8",
    )
    runner = HookRunner(HookConfig(post_tool_use=[f"{sys.executable} {script}"]), tmp_path)
    call = ToolUse(id="1", name="bash", input={"command": "true"})
    await runner.post_tool_use(call, ToolResult(output="ok"))
    seen = json.loads(dump.read_text(encoding="utf-8"))
    assert "ANTHROPIC_API_KEY" not in seen
    assert seen["SATSUMA_KEEP_ME"] == "1"


async def test_hook_env_keep(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    dump = tmp_path / "env.json"
    script = tmp_path / "hook.py"
    script.write_text(
        "import json, os, sys\njson.load(sys.stdin)\n"
        f"open({str(dump)!r}, 'w').write(json.dumps(dict(os.environ)))\nprint('{{}}')\n",
        encoding="utf-8",
    )
    runner = HookRunner(
        HookConfig(post_tool_use=[f"{sys.executable} {script}"]),
        tmp_path,
        sandbox=SandboxConfig(env_keep=["ANTHROPIC_API_KEY"]),
    )
    call = ToolUse(id="1", name="bash", input={"command": "true"})
    await runner.post_tool_use(call, ToolResult(output="ok"))
    seen = json.loads(dump.read_text(encoding="utf-8"))
    assert seen["ANTHROPIC_API_KEY"] == "sk-ant-test"


async def test_hook_takes_the_base_environment(tmp_path: Path) -> None:
    dump = tmp_path / "env.json"
    script = tmp_path / "hook.py"
    script.write_text(
        "import json, os, sys\njson.load(sys.stdin)\n"
        f"open({str(dump)!r}, 'w').write(json.dumps(dict(os.environ)))\nprint('{{}}')\n",
        encoding="utf-8",
    )
    runner = HookRunner(
        HookConfig(post_tool_use=[f"{sys.executable} {script}"]),
        tmp_path,
        sandbox=SandboxConfig(env_mode="allow-list"),
        env={**_WIDE, "PATH": "/usr/bin:/bin"},
    )
    call = ToolUse(id="1", name="bash", input={"command": "true"})
    await runner.post_tool_use(call, ToolResult(output="ok"))
    seen = json.loads(dump.read_text(encoding="utf-8"))
    assert "DATABASE_URL" not in seen and "SATSUMA_KEEP_ME" not in seen
    assert seen["HOME"] == "/home/u"


async def test_lsp_client_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("SATSUMA_KEEP_ME", "1")
    captured: dict[str, object] = {}

    async def fake_exec(*args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        raise OSError("spawn stopped by test")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    client = LspClient(["no-such-server"], tmp_path)
    with pytest.raises(OSError):
        await client.start()
    env = captured["env"]
    assert isinstance(env, dict)
    assert "ANTHROPIC_API_KEY" not in env and env["SATSUMA_KEEP_ME"] == "1"


async def test_lsp_registry_passes_sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    captured: dict[str, object] = {}

    async def fake_exec(*args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        raise OSError("spawn stopped by test")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    registry = LanguageServers(
        tmp_path,
        LspConfig(servers={"python": [sys.executable, "-c", "pass"]}),
        sandbox=SandboxConfig(env_keep=["ANTHROPIC_API_KEY"]),
    )
    with pytest.raises(ToolError):
        await registry._client_for(tmp_path / "a.py")
    env = captured["env"]
    assert isinstance(env, dict) and env["ANTHROPIC_API_KEY"] == "sk-ant-test"
    await registry.aclose()


async def test_lsp_registry_takes_the_base_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    async def fake_exec(*args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        raise OSError("spawn stopped by test")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    registry = LanguageServers(
        tmp_path,
        LspConfig(servers={"python": [sys.executable, "-c", "pass"]}),
        sandbox=SandboxConfig(env_mode="allow-list"),
        env=_WIDE,
    )
    with pytest.raises(ToolError):
        await registry._client_for(tmp_path / "a.py")
    env = captured["env"]
    assert isinstance(env, dict) and env["HOME"] == "/home/u"
    assert "DATABASE_URL" not in env and "GH_TOKEN" not in env
    await registry.aclose()


def _child_only_server(tmp_path: Path) -> Path:
    bindir = tmp_path / "child-bin"
    bindir.mkdir()
    server = bindir / "child-only-ls"
    server.write_text("#!/bin/sh\n", encoding="utf-8")
    server.chmod(0o755)
    return bindir


async def test_lsp_registry_finds_servers_on_the_child_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[object] = []

    async def fake_exec(*args: object, **kwargs: object) -> object:
        started.extend(args)
        raise OSError("spawn stopped by test")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    bindir = _child_only_server(tmp_path)
    registry = LanguageServers(
        tmp_path,
        LspConfig(servers={"python": ["child-only-ls"]}),
        env={**_BASE, "PATH": str(bindir)},
    )
    assert registry.search_path == str(bindir)
    with pytest.raises(ToolError, match="failed to start"):
        await registry._client_for(tmp_path / "a.py")
    assert started[0] == "child-only-ls"
    await registry.aclose()


async def test_lsp_registry_skips_servers_only_on_the_host_path(tmp_path: Path) -> None:
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    registry = LanguageServers(
        tmp_path, LspConfig(servers={"python": ["sh"]}), env={**_BASE, "PATH": str(empty)}
    )
    with pytest.raises(ToolError, match="no language server available"):
        await registry._client_for(tmp_path / "a.py")
    await registry.aclose()


def test_config_show_masks_mcp_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SATSUMA_CONFIG_HOME", str(tmp_path / "home"))
    state = tmp_path / ".satsuma"
    state.mkdir()
    (state / "settings.json").write_text(
        json.dumps(
            {
                "mcp_servers": {
                    "srv": {
                        "transport": "stdio",
                        "command": "my-server",
                        "env": {"TOKEN": "secret-value", "MODE": "x"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    result = CliRunner().invoke(cli_app.app, ["config", "show", "--cwd", str(tmp_path), "--trust"])
    assert result.exit_code == 0, result.output
    assert "secret-value" not in result.output
    shown = json.loads(result.output)
    assert shown["mcp_servers"]["srv"]["env"] == {"TOKEN": "***", "MODE": "***"}
    assert shown["mcp_servers"]["srv"]["command"] == "my-server"

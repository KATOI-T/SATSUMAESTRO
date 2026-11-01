"""隔離設定と既存不変条件・権限結線のテスト。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from satsumaestro.config.models import HookConfig, RuntimeConfig, SandboxConfig
from satsumaestro.engine.events import MessageStop, TextDelta, ToolUse
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security.allowlist import PermissionResolver, SessionAllowlist, signature_for
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.jail import WorkspaceJail
from satsumaestro.security.policy import PermissionOutcome, PermissionPolicy
from satsumaestro.security.sandbox import (
    FS_MODE_VAR,
    FS_MOUNTS_VAR,
    FilesystemMode,
    Isolation,
    bash_environment,
    isolation_for,
    mount_paths,
)
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from satsumaestro.tools.base import ToolContext
from satsumaestro.tools.bash_validation import validate_bash
from satsumaestro.tools.builtins.bash import Bash
from satsumaestro.util.errors import PathEscapeError, ToolError
from tests.fixtures.mock_provider import SequencedMockProvider


def test_redirect_still_jailed(tmp_path: Path) -> None:
    """既存不変: リダイレクト先の jail 逸脱は PathEscapeError（mode 付きでも前段が先）。"""
    jail = WorkspaceJail(tmp_path)
    with pytest.raises(PathEscapeError):
        validate_bash("echo x > /etc/passwd", jail)
    with pytest.raises(PathEscapeError):
        validate_bash(
            "echo x > /etc/passwd", jail, mode=PermissionMode.DANGER_FULL_ACCESS, workspace=tmp_path
        )
    result = validate_bash(
        "echo x > out.txt", jail, mode=PermissionMode.DANGER_FULL_ACCESS, workspace=tmp_path
    )
    assert result.kind == "allow"


def test_quoted_closing_tag_passes_both_stages(tmp_path: Path) -> None:
    """引用符の中の ``</`` は jail 段でも read-only 段でも止めない。"""
    jail = WorkspaceJail(tmp_path)
    for mode in (PermissionMode.READ_ONLY, PermissionMode.WORKSPACE_WRITE):
        result = validate_bash("grep -c '</body>' index.html", jail, mode=mode, workspace=tmp_path)
        assert result.kind == "allow"


def test_command_substitution_still_blocked(tmp_path: Path) -> None:
    """既存不変: バッククォート / $( は ToolError（mode 付きでも同じ）。"""
    jail = WorkspaceJail(tmp_path)
    with pytest.raises(ToolError):
        validate_bash("echo `whoami`", jail)
    with pytest.raises(ToolError):
        validate_bash("echo $(whoami)", jail, mode=PermissionMode.READ_ONLY, workspace=tmp_path)
    with pytest.raises(ToolError):
        validate_bash("rm -rf x", jail, mode=PermissionMode.READ_ONLY, workspace=tmp_path)
    result = validate_bash(
        "rm -r -f build/", jail, mode=PermissionMode.DANGER_FULL_ACCESS, workspace=tmp_path
    )
    assert result.kind == "warn"


def test_allow_list_mount_resolution(tmp_path: Path) -> None:
    """allowed_mounts は jail と同じ規則で正規化され、workspace-only では無視される。"""
    isolation = isolation_for(
        SandboxConfig(),
        filesystem_mode=FilesystemMode.ALLOW_LIST,
        allowed_mounts=["logs", "/opt/data"],
    )
    assert mount_paths(isolation, tmp_path) == (
        (tmp_path / "logs").resolve(),
        Path("/opt/data").resolve(),
    )
    ws_only = Isolation(allowed_mounts=("logs",))
    assert mount_paths(ws_only, tmp_path) == ()
    degraded = isolation_for(SandboxConfig(), filesystem_mode=FilesystemMode.ALLOW_LIST)
    assert degraded.filesystem_mode is FilesystemMode.WORKSPACE_ONLY
    cfg = SandboxConfig(
        filesystem_mode="allow-list", allowed_mounts=["shared"], isolate_network=True
    )
    inherited = isolation_for(cfg)
    assert inherited.filesystem_mode is FilesystemMode.ALLOW_LIST
    assert inherited.allowed_mounts == ("shared",)
    assert inherited.isolate_network is True
    overridden = isolation_for(cfg, isolate_network=False, allowed_mounts=[])
    assert overridden.isolate_network is False
    assert overridden.filesystem_mode is FilesystemMode.WORKSPACE_ONLY


async def test_isolate_network_env(tmp_path: Path) -> None:
    """isolate_network=True で子 env にプロキシ無効化が入る（ベストエフォート）。"""
    isolation = Isolation(isolate_network=True)
    env = bash_environment(
        {"http_proxy": "http://corp:8080", "no_proxy": "localhost", "PATH": "/usr/bin"},
        isolation=isolation,
    )
    assert env["http_proxy"] == "http://127.0.0.1:9"
    assert env["HTTPS_PROXY"] == "http://127.0.0.1:9"
    assert "no_proxy" not in env
    assert env["PATH"] == "/usr/bin"
    assert env[FS_MODE_VAR] == "workspace-only"
    assert env[FS_MOUNTS_VAR] == ""
    listed = bash_environment(
        {},
        isolation=Isolation(filesystem_mode=FilesystemMode.ALLOW_LIST),
        mounts=[Path("/a"), Path("/b")],
    )
    assert listed[FS_MODE_VAR] == "allow-list"
    assert listed[FS_MOUNTS_VAR] == f"/a{os.pathsep}/b"
    untouched = bash_environment({"http_proxy": "http://corp:8080"}, isolation=Isolation())
    assert untouched["http_proxy"] == "http://corp:8080"
    ctx = ToolContext(workspace_root=tmp_path, jail=WorkspaceJail(tmp_path), config=RuntimeConfig())
    result = await Bash().run(
        Bash.Input(command='printf "%s" "$http_proxy"', isolate_network=True), ctx
    )
    assert result.is_error is False
    assert result.output == "http://127.0.0.1:9"


async def test_block_overrides_allow(tmp_path: Path) -> None:
    """権限 allow（session allowlist）でも検証段 Block なら実行に至らない。"""
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me", encoding="utf-8")
    command = f"truncate -s 0 {victim.name}"

    session_allow = SessionAllowlist()
    session_allow.add(signature_for("bash", {"command": command}))
    resolver = PermissionResolver(
        base=PermissionPolicy(mode=PermissionMode.READ_ONLY), session=session_allow
    )
    assert resolver.authorize("bash", {"command": command}, PermissionMode.DANGER_FULL_ACCESS) == (
        PermissionOutcome(decision="allow", reason="session allowlist", scoped=True)
    )

    prompter_calls: list[str] = []

    class AllowPrompter:
        def decide(
            self, call: ToolUse, required: PermissionMode, warning: str | None = None
        ) -> PermissionOutcome:
            prompter_calls.append(call.name)
            return PermissionOutcome(decision="allow", reason="approved")

    provider = SequencedMockProvider(
        [
            [ToolUse(id="b1", name="bash", input={"command": command}), MessageStop()],
            [TextDelta(text="done"), MessageStop()],
        ]
    )
    rt = ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=provider,  # type: ignore[arg-type]
        tools=default_registry(),
        policy=resolver,
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=RuntimeConfig(),
    )
    await rt.run_turn("delete it", prompter=AllowPrompter())

    tool_msg = rt.session.messages[2]
    assert tool_msg.role == "tool"
    assert tool_msg.blocks[0].is_error is True
    assert "blocked by bash validation" in tool_msg.blocks[0].output
    assert prompter_calls == []
    assert victim.read_text(encoding="utf-8") == "keep me"

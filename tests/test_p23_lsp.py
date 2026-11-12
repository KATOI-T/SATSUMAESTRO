"""LSP 統合のテスト。"""

from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from satsumaestro.config.models import LspConfig
from satsumaestro.security.enforcer import PermissionMode, classify
from satsumaestro.tools import ToolContext
from satsumaestro.tools.builtins.lsp import Lsp
from satsumaestro.tools.lsp.client import encode_message, read_message
from satsumaestro.tools.lsp.models import LspRequest, action_for
from satsumaestro.tools.lsp.registry import LanguageServers, _apply_edits
from satsumaestro.tools.lsp.servers import language_for_path, resolve_server_command
from satsumaestro.util.errors import ToolError

_STUB = str(Path(__file__).parent / "fixtures" / "lsp_stub.py")


def _stub_config(**kwargs: object) -> LspConfig:
    return LspConfig(servers={"python": [sys.executable, _STUB]}, **kwargs)  # type: ignore[arg-type]


@pytest.fixture
def sample(tmp_path: Path) -> Path:
    path = tmp_path / "sample.py"
    path.write_text("x=1\ndef f() -> int:\n    return x\n", encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("raw", "action"),
    [
        ("goToDefinition", LspRequest.DEFINITION),
        ("textDocument/definition", LspRequest.DEFINITION),
        ("find-references", LspRequest.REFERENCES),
        ("refs", LspRequest.REFERENCES),
        ("completions", LspRequest.COMPLETION),
        ("documentSymbol", LspRequest.SYMBOLS),
        ("workspace/symbol", LspRequest.SYMBOLS),
        ("formatting", LspRequest.FORMAT),
        (" Hover ", LspRequest.HOVER),
        ("publishDiagnostics", LspRequest.DIAGNOSTICS),
    ],
)
def test_action_spellings(raw: str, action: LspRequest) -> None:
    assert action_for(raw) is action
    assert Lsp.Input(action=raw).action == action.value


def test_unknown_action_is_an_input_error() -> None:
    assert action_for("rename") is None
    with pytest.raises(ValidationError, match="unknown lsp action 'rename'"):
        Lsp.Input(action="rename")


def test_classify_format_is_write() -> None:
    fmt = classify("lsp", Lsp.Input(action="format"), PermissionMode.READ_ONLY)
    assert fmt is PermissionMode.WORKSPACE_WRITE
    alias = classify("lsp", Lsp.Input(action="textDocument/formatting"), PermissionMode.READ_ONLY)
    assert alias is PermissionMode.WORKSPACE_WRITE
    for action in ("diagnostics", "hover", "definition", "references", "completion", "symbols"):
        assert classify("lsp", Lsp.Input(action=action), PermissionMode.READ_ONLY) is (
            PermissionMode.READ_ONLY
        )


async def test_dispatch_names_what_is_missing(tmp_path: Path) -> None:
    servers = LanguageServers(tmp_path, LspConfig())
    with pytest.raises(ToolError, match="lsp hover needs path"):
        await servers.dispatch(LspRequest.HOVER, path=None, line=0, character=0, query=None)
    target = tmp_path / "a.py"
    target.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(
        ToolError, match=r"lsp definition needs line and character \(both 0-based\)"
    ):
        await servers.dispatch(
            LspRequest.DEFINITION, path=target, line=None, character=3, query=None
        )
    with pytest.raises(ToolError, match="lsp symbols needs path"):
        await servers.dispatch(LspRequest.SYMBOLS, path=None, line=None, character=None, query=None)
    other = tmp_path / "notes.txt"
    other.write_text("hi\n", encoding="utf-8")
    with pytest.raises(ToolError) as err:
        await servers.dispatch(LspRequest.HOVER, path=other, line=0, character=0, query=None)
    assert str(err.value).startswith("lsp has no language server for .txt files")
    assert ".mjs" in str(err.value) and ".cts" in str(err.value)
    await servers.aclose()


def test_language_and_server_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    assert language_for_path(Path("a.py")) == "python"
    assert language_for_path(Path("a.tsx")) == "typescript"
    assert language_for_path(Path("a.txt")) is None
    monkeypatch.setattr(shutil, "which", lambda cmd, path=None: None)
    assert resolve_server_command("python", {"python": ["no-such-server"]}) is None
    assert resolve_server_command("python", {}) is None


async def test_client_framing() -> None:
    payload = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"a": "あ"}}
    encoded = encode_message(payload)
    reader = asyncio.StreamReader()
    reader.feed_data(encoded + encode_message({"jsonrpc": "2.0", "method": "n", "params": {}}))
    reader.feed_eof()
    assert await read_message(reader) == payload
    second = await read_message(reader)
    assert second is not None and second["method"] == "n"
    assert await read_message(reader) is None


async def test_server_missing_degrades(tool_ctx: ToolContext, sample: Path) -> None:
    res = await Lsp().run(
        Lsp.Input(action="hover", path="sample.py", line=0, character=0), tool_ctx
    )
    assert res.is_error and "disabled" in res.output
    registry = LanguageServers(
        tool_ctx.workspace_root, LspConfig(servers={"python": ["no-such-binary"]})
    )
    ctx = tool_ctx.model_copy(update={"lsp": registry})
    res2 = await Lsp().run(Lsp.Input(action="hover", path="sample.py", line=0, character=0), ctx)
    assert res2.is_error and "no language server" in res2.output
    await registry.aclose()


def test_every_spelling_of_format_is_classified_as_write() -> None:
    from satsumaestro.tools.lsp.models import _FORMS

    for raw, action in _FORMS.items():
        mode = classify("lsp", Lsp.Input(action=raw), PermissionMode.READ_ONLY)
        expected = (
            PermissionMode.WORKSPACE_WRITE
            if action is LspRequest.FORMAT
            else PermissionMode.READ_ONLY
        )
        assert mode is expected, raw


async def test_jail_enforced(tool_ctx: ToolContext) -> None:
    registry = LanguageServers(tool_ctx.workspace_root, _stub_config())
    ctx = tool_ctx.model_copy(update={"lsp": registry})
    from satsumaestro.util.errors import PathEscapeError

    with pytest.raises(PathEscapeError):
        await Lsp().run(Lsp.Input(action="hover", path="/etc/hosts", line=0, character=0), ctx)
    await registry.aclose()


async def test_position_required(tool_ctx: ToolContext, sample: Path) -> None:
    registry = LanguageServers(tool_ctx.workspace_root, _stub_config())
    ctx = tool_ctx.model_copy(update={"lsp": registry})
    res = await Lsp().run(Lsp.Input(action="hover", path="sample.py"), ctx)
    assert res.is_error and "line and character" in res.output
    await registry.aclose()


@pytest.fixture
async def stub_ctx(tool_ctx: ToolContext, sample: Path):  # type: ignore[no-untyped-def]
    registry = LanguageServers(tool_ctx.workspace_root, _stub_config())
    yield tool_ctx.model_copy(update={"lsp": registry}), registry
    await registry.aclose()


async def test_dispatch_diagnostics(stub_ctx) -> None:  # type: ignore[no-untyped-def]
    ctx, _registry = stub_ctx
    res = await Lsp().run(Lsp.Input(action="diagnostics", path="sample.py"), ctx)
    assert not res.is_error
    assert (
        "1 diagnostics" in res.output and "sample.py:1:4 error stub type error [stub]" in res.output
    )


async def test_dispatch_diagnostics_aggregate(stub_ctx) -> None:  # type: ignore[no-untyped-def]
    ctx, _registry = stub_ctx
    await Lsp().run(Lsp.Input(action="diagnostics", path="sample.py"), ctx)
    res = await Lsp().run(Lsp.Input(action="diagnostics"), ctx)
    assert not res.is_error and "cached" in res.output and "sample.py:1:4" in res.output


async def test_dispatch_hover_definition_references(stub_ctx) -> None:  # type: ignore[no-untyped-def]
    ctx, _registry = stub_ctx
    hover = await Lsp().run(Lsp.Input(action="hover", path="sample.py", line=1, character=4), ctx)
    assert "def f() -> int" in hover.output
    definition = await Lsp().run(
        Lsp.Input(action="go_to_definition", path="sample.py", line=2, character=11), ctx
    )
    assert "1 definition(s)" in definition.output and "sample.py:0:4" in definition.output
    refs = await Lsp().run(
        Lsp.Input(action="references", path="sample.py", line=0, character=0), ctx
    )
    assert "2 reference(s)" in refs.output and "sample.py:3:8" in refs.output


async def test_dispatch_completion_and_symbols(stub_ctx) -> None:  # type: ignore[no-untyped-def]
    ctx, _registry = stub_ctx
    comp = await Lsp().run(
        Lsp.Input(action="completion", path="sample.py", line=2, character=11), ctx
    )
    assert "frobnicate (function)" in comp.output
    symbols = await Lsp().run(Lsp.Input(action="symbols", path="sample.py"), ctx)
    assert "class Greeter" in symbols.output and "  method greet" in symbols.output
    ws = await Lsp().run(Lsp.Input(action="symbols", query="Greeter"), ctx)
    assert "Greeter" in ws.output


async def test_dispatch_format_applies_edits(stub_ctx, tool_ctx: ToolContext) -> None:  # type: ignore[no-untyped-def]
    ctx, _registry = stub_ctx
    res = await Lsp().run(Lsp.Input(action="format", path="sample.py"), ctx)
    assert "applied 1 formatting edits" in res.output
    assert (tool_ctx.workspace_root / "sample.py").read_text().startswith("x = 1\n")


async def test_registry_closes_all(tool_ctx: ToolContext, sample: Path) -> None:
    registry = LanguageServers(tool_ctx.workspace_root, _stub_config())
    ctx = tool_ctx.model_copy(update={"lsp": registry})
    await Lsp().run(Lsp.Input(action="hover", path="sample.py", line=0, character=0), ctx)
    client = registry._clients["python"]
    assert client is not None and client._proc is not None
    await registry.aclose()
    assert client._proc.returncode is not None
    assert registry._clients == {}


async def test_server_reused_across_calls(stub_ctx) -> None:  # type: ignore[no-untyped-def]
    ctx, registry = stub_ctx
    await Lsp().run(Lsp.Input(action="hover", path="sample.py", line=0, character=0), ctx)
    first = registry._clients["python"]
    await Lsp().run(Lsp.Input(action="hover", path="sample.py", line=1, character=0), ctx)
    assert registry._clients["python"] is first


async def test_hang_init_times_out(tool_ctx: ToolContext, sample: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("LSP_STUB_MODE", "hang_init")
    registry = LanguageServers(tool_ctx.workspace_root, _stub_config(request_timeout_s=0.5))
    ctx = tool_ctx.model_copy(update={"lsp": registry})
    res = await Lsp().run(Lsp.Input(action="hover", path="sample.py", line=0, character=0), ctx)
    assert res.is_error and "failed to start" in res.output
    res2 = await Lsp().run(Lsp.Input(action="hover", path="sample.py", line=0, character=0), ctx)
    assert res2.is_error and "no language server" in res2.output
    await registry.aclose()


def test_apply_edits_multiline() -> None:
    text = "a\nbb\nccc\n"
    edits = [
        {
            "range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 1}},
            "newText": "A",
        },
        {
            "range": {"start": {"line": 2, "character": 0}, "end": {"line": 2, "character": 3}},
            "newText": "C",
        },
    ]
    assert _apply_edits(text, edits) == "A\nbb\nC\n"


@pytest.mark.skipif(shutil.which("pyright-langserver") is None, reason="pyright 未導入")
async def test_dispatch_real_pyright(tool_ctx: ToolContext) -> None:
    bad = tool_ctx.workspace_root / "bad.py"
    bad.write_text('x: int = "not-an-int"\n', encoding="utf-8")
    registry = LanguageServers(tool_ctx.workspace_root, LspConfig())
    ctx = tool_ctx.model_copy(update={"lsp": registry})
    try:
        res = await Lsp().run(Lsp.Input(action="diagnostics", path="bad.py"), ctx)
        assert not res.is_error and "error" in res.output
    finally:
        await registry.aclose()


async def test_child_runtime_shares_lsp_registry(tmp_path: Path) -> None:
    from satsumaestro.config.models import HookConfig, RuntimeConfig
    from satsumaestro.engine.events import MessageStop
    from satsumaestro.engine.subagent import RuntimeFactory, resolve_subagent_type
    from satsumaestro.hooks.runner import HookRunner
    from satsumaestro.security.policy import default_policy
    from satsumaestro.tools import default_registry
    from tests.fixtures.mock_provider import SequencedMockProvider

    provider = SequencedMockProvider([[MessageStop()]])
    registry = LanguageServers(tmp_path, LspConfig())
    factory = RuntimeFactory(
        provider=provider,  # type: ignore[arg-type]
        model="mock-model",
        config=RuntimeConfig(),
        tools=default_registry(),
        policy=default_policy(PermissionMode.READ_ONLY),
        hooks=HookRunner(HookConfig(), tmp_path),
        cwd=tmp_path,
        provider_builder=lambda m: provider,  # type: ignore[arg-type, return-value]
        lsp=registry,
    )
    child = factory.build_child(preset=resolve_subagent_type("general"), depth=1)
    assert child.tool_context.lsp is registry
    await registry.aclose()

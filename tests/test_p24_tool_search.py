"""ツール遅延ロード・tool_search のテスト。"""

from __future__ import annotations

from pathlib import Path

import pytest

from satsumaestro.config.models import HookConfig, RuntimeConfig, ToolsConfig
from satsumaestro.engine.events import MessageStop, TextDelta, ToolUse
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder, format_tool_catalog
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.models import ToolResultBlock
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import ToolContext, default_registry
from satsumaestro.tools.aliases import alias_target, name_key, name_words
from satsumaestro.tools.builtins.tool_search import ToolSearch
from satsumaestro.util.errors import ToolError
from tests.fixtures.mock_provider import SequencedMockProvider


@pytest.mark.parametrize(
    ("raw", "words"),
    [
        ("web-fetch", ["web", "fetch"]),
        ("WebFetch", ["web", "fetch"]),
        ("ToolSearch", ["tool", "search"]),
        ("read_file", ["read", "file"]),
        ("_Bash_", ["bash"]),
        ("HTTPServer", ["http", "server"]),
        ("read_file?", ["read", "file"]),
    ],
)
def test_name_words(raw: str, words: list[str]) -> None:
    assert name_words(raw) == words


def test_name_key_and_aliases() -> None:
    assert name_key("Web-Fetch") == name_key("web_fetch") == "webfetch"
    assert alias_target("read") == alias_target("READ") == "read_file"
    assert alias_target("list_dir") == alias_target("listDir") == "list_directory"
    assert alias_target("tool-search") == "tool_search"
    assert alias_target("read_file") is None


def test_core_schemas_exclude_deferred() -> None:
    reg = default_registry()
    core = {e["name"] for e in reg.core_schemas()}
    catalog = {e["name"] for e in reg.deferred_catalog()}
    assert "read_file" in core and "bash" in core and "tool_search" in core
    assert "web_fetch" in catalog and "web_search" in catalog and "todo_write" in catalog
    assert not (core & catalog)
    entry = next(e for e in reg.deferred_catalog() if e["name"] == "web_fetch")
    assert set(entry) == {"name", "description", "required"}


def test_deferred_invisible_before_search() -> None:
    reg = default_registry()
    names = {e["name"] for e in reg.schemas_for(set())}
    assert "web_fetch" not in names and "tool_search" in names
    names2 = {e["name"] for e in reg.schemas_for({"web_fetch"})}
    assert "web_fetch" in names2


def test_core_override_forces_exposure() -> None:
    reg = default_registry(core_override=["web_fetch"])
    assert not reg.is_deferred("web_fetch")
    assert "web_fetch" in {e["name"] for e in reg.schemas_for(set())}
    child = reg.filtered(None)
    assert not child.is_deferred("web_fetch")


def test_json_schemas_backward_compatible() -> None:
    reg = default_registry()
    assert {e["name"] for e in reg.json_schemas()} >= {"web_fetch", "read_file", "tool_search"}


def test_search_select_exact() -> None:
    reg = default_registry()
    assert reg.search("select:web_fetch") == ["web_fetch"]
    assert reg.search("select:WebFetch") == ["web_fetch"]
    assert reg.search("select:web_fetch,web_search") == ["web_fetch", "web_search"]
    assert reg.search("select:no_such_tool") == []


def test_search_select_keeps_order_and_drops_duplicates() -> None:
    reg = default_registry()
    assert reg.search("select:web_search, WebFetch, web_search") == ["web_search", "web_fetch"]
    assert reg.search("select:todo, web_fetch, lsp", max_results=2) == ["web_fetch", "lsp"]


def test_search_keyword_scoring() -> None:
    reg = default_registry()
    assert reg.search("web")[:2] == ["web_fetch", "web_search"]
    assert reg.search("web", max_results=1) == ["web_fetch"]
    assert reg.search("web_fetch")[0] == "web_fetch"
    assert reg.search("WebSearch")[0] == "web_search"
    assert reg.search("+search web") == ["web_search"]
    assert reg.search("+web todo") == ["web_fetch", "web_search"]
    assert "lsp" in reg.search("definition")
    assert reg.search("   ", max_results=20) == sorted(reg.deferred_names())
    assert reg.search("zzz_nothing") == []


def test_search_excludes_core() -> None:
    reg = default_registry()
    assert reg.search("select:read_file") == []


def test_normalize_aliases_and_canonical() -> None:
    reg = default_registry()
    got = reg.resolve_allowed_tools(["Read, web-fetch bash", "LIST_DIR"])
    assert got == {"read_file", "web_fetch", "bash", "list_directory"}
    assert reg.resolve_allowed_tools([]) is None


def test_normalize_lists_every_unknown_name_at_once() -> None:
    reg = default_registry()
    with pytest.raises(ToolError) as ei:
        reg.resolve_allowed_tools(["no_such_tool, bash, frob"])
    lines = str(ei.value).splitlines()
    assert lines[0] == "--allowed-tools に知らないツール名があります: no_such_tool, frob"
    assert lines[1].startswith("使えるツール名: ") and "read_file" in lines[1]
    assert lines[2].startswith("別の呼び名: ") and "Read（read_file）" in lines[2]
    with pytest.raises(ToolError, match="ツール名がありません"):
        reg.resolve_allowed_tools([" , "])


async def test_tool_search_tool_outputs_matches(tool_ctx: ToolContext) -> None:
    reg = default_registry()
    ctx = tool_ctx.model_copy(update={"registry": reg})
    res = await ToolSearch().run(ToolSearch.Input(query="select:web_fetch"), ctx)
    assert not res.is_error
    assert res.output == "Loaded: web_fetch. Call them by name from your next step."
    miss = await ToolSearch().run(ToolSearch.Input(query="zzz"), ctx)
    assert not miss.is_error
    assert miss.output.startswith("Nothing matched `zzz`. Deferred tools: ")
    assert all(name in miss.output for name in reg.deferred_names())


async def test_tool_search_degrades_without_registry(tool_ctx: ToolContext) -> None:
    res = await ToolSearch().run(ToolSearch.Input(query="web"), tool_ctx)
    assert res.is_error


def test_toolsearch_is_core_always_exposed() -> None:
    reg = default_registry()
    assert "tool_search" in {e["name"] for e in reg.schemas_for(set())}
    assert ToolSearch.spec.required_mode is PermissionMode.READ_ONLY
    assert ToolSearch.spec.parallel_safe and not ToolSearch.spec.deferred


def test_format_tool_catalog() -> None:
    text = format_tool_catalog(
        [{"name": "web_fetch", "description": "Fetch a URL. Second sentence is dropped."}]
    )
    assert "- web_fetch: Fetch a URL" in text and "Second sentence" not in text
    assert "tool_search" in text
    assert format_tool_catalog([]) == ""


def test_catalog_in_static_half() -> None:
    reg = default_registry()
    builder = SystemPromptBuilder(tool_catalog=format_tool_catalog(reg.deferred_catalog()))
    static = builder._static_sections("m", RuntimeConfig())
    assert "# Deferred tools" in static and "web_fetch" in static


def _runtime(provider: object, tmp_path: Path, cfg: RuntimeConfig) -> ConversationRuntime:
    return ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=provider,  # type: ignore[arg-type]
        tools=default_registry(core_override=cfg.tools.core),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=cfg,
    )


async def test_deferred_visible_after_search(tmp_path: Path) -> None:
    provider = SequencedMockProvider(
        [
            [
                ToolUse(id="c1", name="tool_search", input={"query": "select:web_fetch"}),
                MessageStop(stop_reason="tool_use"),
            ],
            [TextDelta(text="done"), MessageStop()],
        ]
    )
    rt = _runtime(provider, tmp_path, RuntimeConfig())
    before = {s["name"] for s in rt._all_schemas()}
    assert "web_fetch" not in before
    summary = await rt.run_turn("hi")
    assert summary.final_text == "done"
    assert "web_fetch" in rt._expanded
    after = {s["name"] for s in rt._all_schemas()}
    assert "web_fetch" in after


async def test_unexpanded_call_loads_and_runs(tmp_path: Path) -> None:
    provider = SequencedMockProvider(
        [
            [
                ToolUse(id="c1", name="todo_write", input={"todos": [{"content": "A"}]}),
                MessageStop(stop_reason="tool_use"),
            ],
            [TextDelta(text="done"), MessageStop()],
        ]
    )
    rt = _runtime(provider, tmp_path, RuntimeConfig(tools=ToolsConfig(core=[])))
    summary = await rt.run_turn("hi")
    assert summary.final_text == "done"
    tool_msgs = [m for m in rt.session.messages if m.role == "tool"]
    block = tool_msgs[0].blocks[0]
    assert isinstance(block, ToolResultBlock)
    assert not block.is_error and "[ ] A" in block.output
    assert [t.content for t in rt.session.todos] == ["A"]
    assert "todo_write" in rt._expanded and summary.deferred_autoloads == 1
    assert "todo_write" in {str(s["name"]) for s in rt._all_schemas()}


async def test_deferred_loading_disabled_fallback(tmp_path: Path) -> None:
    cfg = RuntimeConfig(tools=ToolsConfig(deferred_tool_loading=False))
    provider = SequencedMockProvider([[TextDelta(text="done"), MessageStop()]])
    rt = _runtime(provider, tmp_path, cfg)
    names = {s["name"] for s in rt._all_schemas()}
    assert "web_fetch" in names and "todo_write" in names
    assert not rt._deferred_unexposed("web_fetch")


async def test_expansion_persists_across_turns(tmp_path: Path) -> None:
    provider = SequencedMockProvider(
        [
            [
                ToolUse(id="c1", name="tool_search", input={"query": "select:todo_write"}),
                MessageStop(stop_reason="tool_use"),
            ],
            [TextDelta(text="one"), MessageStop()],
            [TextDelta(text="two"), MessageStop()],
        ]
    )
    rt = _runtime(provider, tmp_path, RuntimeConfig(tools=ToolsConfig(core=[])))
    await rt.run_turn("first")
    assert "todo_write" in rt._expanded
    await rt.run_turn("second")
    assert "todo_write" in rt._expanded

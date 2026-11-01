"""プロンプト再構築と KV キャッシュ安定化のテスト。"""

from __future__ import annotations

from pathlib import Path

import pytest

from satsumaestro.config.models import HookConfig, RuntimeConfig, WebSearchConfig
from satsumaestro.engine.events import MessageStop, TextDelta, ToolUse
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.i18n.state import LanguageState
from satsumaestro.prompt.builder import SystemPrompt, SystemPromptBuilder, TurnContext
from satsumaestro.providers.base import ApiRequest, ProviderKind
from satsumaestro.providers.openai_provider import _to_openai_messages
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.models import ModeState, TodoItem
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from tests.fixtures.mock_provider import SequencedMockProvider


def _todo(content: str, status: str = "pending") -> TodoItem:
    return TodoItem(content=content, active_form=f"{content} 中", status=status)  # type: ignore[arg-type]


def _runtime(
    provider: object, tmp_path: Path, cfg: RuntimeConfig | None = None
) -> ConversationRuntime:
    return ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=provider,  # type: ignore[arg-type]
        tools=default_registry(),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=cfg if cfg is not None else RuntimeConfig(),
    )


async def test_system_is_byte_stable_across_iterations_after_a_write(tmp_path: Path) -> None:
    """ターン中にファイルを書いても、同じターンの system のバイト列は動かない。"""
    provider = SequencedMockProvider(
        [
            [
                ToolUse(id="t1", name="write_file", input={"path": "a.txt", "content": "x" * 50}),
                MessageStop(stop_reason="tool_use"),
            ],
            [
                ToolUse(id="t2", name="write_file", input={"path": "b.txt", "content": "y" * 50}),
                MessageStop(stop_reason="tool_use"),
            ],
            [TextDelta(text="終わりました"), MessageStop(stop_reason="end_turn")],
        ]
    )
    runtime = _runtime(provider, tmp_path)
    summary = await runtime.run_turn("ファイルを 2 つ作ってください")

    assert summary.iterations == 3
    systems = [(r.system, r.system_dynamic) for r in provider.requests]
    assert len(set(systems)) == 1, "ターン内で system が動いた = キャッシュ接頭辞が壊れている"
    assert summary.prompt_prefix_breaks == 0
    assert (tmp_path / "a.txt").is_file() and (tmp_path / "b.txt").is_file()


async def test_todo_update_does_not_move_the_system_prompt(tmp_path: Path) -> None:
    """TODO は system に置かない。"""
    provider = SequencedMockProvider(
        [
            [
                ToolUse(id="s1", name="tool_search", input={"query": "select:todo_write"}),
                MessageStop(stop_reason="tool_use"),
            ],
            [
                ToolUse(
                    id="t1",
                    name="todo_write",
                    input={"todos": [{"content": "A", "activeForm": "A 中", "status": "pending"}]},
                ),
                MessageStop(stop_reason="tool_use"),
            ],
            [TextDelta(text="計画しました"), MessageStop(stop_reason="end_turn")],
        ]
    )
    runtime = _runtime(provider, tmp_path)
    summary = await runtime.run_turn("計画を立ててください")

    assert summary.iterations == 3
    assert len({(r.system, r.system_dynamic) for r in provider.requests}) == 1
    assert summary.prompt_prefix_breaks == 1
    assert provider.requests[1].tools == provider.requests[2].tools
    assert "# Current TODOs" not in provider.requests[2].system_dynamic
    assert runtime.session.todos and runtime.session.todos[0].content == "A"


async def test_a_moving_system_prompt_is_still_counted(tmp_path: Path) -> None:
    """ターンの中で system が動いたら数える（指標が死んでいないこと）。"""

    class _Moving(SystemPromptBuilder):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def compose(
            self,
            model: str,
            cfg: RuntimeConfig,
            language: LanguageState | None,
            turn: TurnContext,
        ) -> SystemPrompt:
            self.calls += 1
            prompt = super().compose(model, cfg, language, turn)
            return SystemPrompt(static=prompt.static, dynamic=f"{prompt.dynamic}\n#{self.calls}")

    provider = SequencedMockProvider(
        [
            [
                ToolUse(id="r1", name="read_file", input={"path": "missing.txt"}),
                MessageStop(stop_reason="tool_use"),
            ],
            [TextDelta(text="done"), MessageStop(stop_reason="end_turn")],
        ]
    )
    runtime = _runtime(provider, tmp_path)
    runtime.prompt_builder = _Moving()
    summary = await runtime.run_turn("読んで")
    assert summary.iterations == 2
    assert summary.prompt_prefix_breaks == 1


async def test_snapshot_is_taken_once_per_turn(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """``snapshot`` はターンあたり 1 回（= git サブプロセスが反復ごとに起きない）。"""
    calls = 0
    original = SystemPromptBuilder.snapshot

    async def counting(self: SystemPromptBuilder, cwd: Path) -> TurnContext:
        nonlocal calls
        calls += 1
        return await original(self, cwd)

    monkeypatch.setattr(SystemPromptBuilder, "snapshot", counting)
    provider = SequencedMockProvider(
        [
            [
                ToolUse(id="t1", name="read_file", input={"path": "a.txt"}),
                MessageStop(stop_reason="tool_use"),
            ],
            [TextDelta(text="読みました"), MessageStop(stop_reason="end_turn")],
        ]
    )
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    runtime = _runtime(provider, tmp_path)
    summary = await runtime.run_turn("a.txt を読んで")

    assert summary.iterations == 2
    assert calls == 1


async def test_plan_mode_and_todos_are_not_in_the_system_prompt(tmp_path: Path) -> None:
    """plan mode と TODO は system に置かない（変わるたびに会話全体が読み直しになるため）。"""
    (tmp_path / "CLAUDE.md").write_text("RULE", encoding="utf-8")
    session = new_session(tmp_path, "m")
    session.mode = ModeState(main="plan")
    session.todos = [_todo("A")]
    prompt = await SystemPromptBuilder().build(tmp_path, "m", RuntimeConfig())

    dynamic = prompt.dynamic
    assert dynamic.index("# Instructions from") < dynamic.index("# Environment")
    assert "# Plan mode" not in prompt.joined()
    assert "# Current TODOs" not in prompt.joined()
    session.mode = ModeState(main="code")
    session.todos = []
    other = await SystemPromptBuilder().build(tmp_path, "m", RuntimeConfig())
    assert other.static == prompt.static
    assert other.dynamic == dynamic


async def test_static_half_is_memoized_and_byte_identical(tmp_path: Path) -> None:
    b = SystemPromptBuilder()
    cfg = RuntimeConfig()
    first = (await b.build(tmp_path, "m", cfg)).static
    second = (await b.build(tmp_path, "m", cfg)).static
    assert first == second
    assert first is second


async def test_static_memo_rebuilds_on_model_or_config_change(tmp_path: Path) -> None:
    """メモ化はキャッシュ汚染を起こさない（キーに入れた項目が変われば作り直す）。"""
    b = SystemPromptBuilder()
    generic = (await b.build(tmp_path, "gpt-oss:120b", RuntimeConfig())).static
    claude = (await b.build(tmp_path, "claude-opus-4-8", RuntimeConfig())).static
    assert generic != claude

    plain = (await b.build(tmp_path, "m", RuntimeConfig())).static
    planning = (await b.build(tmp_path, "m", RuntimeConfig(planning_guidance=True))).static
    assert plain != planning

    search = (
        await b.build(
            tmp_path, "m", RuntimeConfig(web_search=WebSearchConfig(provider="duckduckgo"))
        )
    ).static
    assert "# Web search usage" not in plain and "# Web search usage" in search
    assert (await b.build(tmp_path, "m", RuntimeConfig())).static == plain


async def test_git_calls_use_no_optional_locks(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """``--no-optional-locks`` 付きで呼ぶ。"""
    seen: list[tuple[str, ...]] = []

    async def fake_run_git(cwd: Path, *args: str) -> str | None:
        seen.append(args)
        return "" if args[-1] == "diff" else " M x.py"

    monkeypatch.setattr("satsumaestro.prompt.builder._git_text", fake_run_git)
    await SystemPromptBuilder().snapshot(tmp_path)

    opts = ("--no-optional-locks", "-c", "core.quotePath=false")
    assert seen == [(*opts, "status", "--short"), (*opts, "diff")]


def test_system_prompt_joined_uses_blank_line_separator() -> None:
    assert SystemPrompt(static="A", dynamic="B").joined() == "A\n\nB"
    assert SystemPrompt(static="A", dynamic="").joined() == "A"


def test_api_request_system_text_matches_system_prompt_joined() -> None:
    """``ApiRequest.system_text`` と ``SystemPrompt.joined`` は同じ規則で連結する。"""
    prompt = SystemPrompt(static="STATIC", dynamic="DYNAMIC")
    req = ApiRequest(model="m", system=prompt.static, system_dynamic=prompt.dynamic, messages=[])
    assert req.system_text() == prompt.joined()


def test_openai_sends_the_joined_system_as_one_message() -> None:
    """OpenAI 互換に system の分割送信は無いので 1 本へ連結する。"""
    req = ApiRequest(model="m", system="STATIC", system_dynamic="DYNAMIC", messages=[])
    messages = _to_openai_messages(req)
    assert messages[0] == {"role": "system", "content": "STATIC\n\nDYNAMIC"}


def test_openai_omits_system_when_both_halves_are_empty() -> None:
    assert _to_openai_messages(ApiRequest(model="m", system="", messages=[])) == []


@pytest.mark.parametrize("kind", [ProviderKind.ANTHROPIC, ProviderKind.OPENAI])
async def test_runtime_sends_both_halves(tmp_path: Path, kind: ProviderKind) -> None:
    """runtime は静的前半と動的後半を分けて渡す（プロバイダ種別に依らず）。"""
    provider = SequencedMockProvider(
        [[TextDelta(text="はい"), MessageStop(stop_reason="end_turn")]], kind=kind
    )
    (tmp_path / "CLAUDE.md").write_text("RULE", encoding="utf-8")
    await _runtime(provider, tmp_path).run_turn("こんにちは")

    req = provider.requests[0]
    assert "You are SATSUMAESTRO" in req.system
    assert "# Environment" not in req.system
    assert "RULE" not in req.system
    assert "# Environment" in req.system_dynamic and "RULE" in req.system_dynamic


async def test_count_tokens_sees_the_whole_system(tmp_path: Path) -> None:
    """トークン数見積りは連結後を見る（動的後半を数え落とすと圧縮判定がずれる）。"""
    provider = SequencedMockProvider(
        [[TextDelta(text="はい"), MessageStop(stop_reason="end_turn")]]
    )
    both = await provider.count_tokens(
        ApiRequest(model="m", system="A" * 400, system_dynamic="B" * 400, messages=[])
    )
    static_only = await provider.count_tokens(ApiRequest(model="m", system="A" * 400, messages=[]))
    assert both > static_only

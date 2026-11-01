"""言語サーバを言語ごとに最初に起動する前の確認。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from satsumaestro.config.models import HookConfig, LspConfig, RuntimeConfig
from satsumaestro.engine.events import AssistantEvent, MessageStop, TextDelta, ToolUse
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.engine.turn import DenyPrompter, TurnSummary
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security.allowlist import PermissionResolver
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import PermissionOutcome, PermissionPolicy, default_policy
from satsumaestro.session.models import ToolResultBlock
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from satsumaestro.tools.lsp.registry import LanguageServers
from tests.fixtures.mock_provider import ScriptedFailureProvider

WW = PermissionMode.WORKSPACE_WRITE
SERVERS = LspConfig(servers={"python": ["sh"], "typescript": ["sh"]})
YES = PermissionOutcome(decision="allow", reason="y")
NO = PermissionOutcome(decision="deny", reason="n")


class _FakeServers(LanguageServers):
    """サーバを起動せずに、渡された path を記録して決まった文を返す。"""

    def __init__(self, root: Path) -> None:
        super().__init__(root, LspConfig())
        self.paths: list[str] = []

    async def dispatch(self, action: Any, **kw: Any) -> str:
        path = kw.get("path")
        name = path.name if isinstance(path, Path) else ""
        self.paths.append(name)
        return f"ok {name}".rstrip()


class _Prompter:
    def __init__(self, *decisions: PermissionOutcome) -> None:
        self.calls: list[tuple[str | None, str | None]] = []
        self._decisions = list(decisions) or [NO]

    def decide(
        self, call: ToolUse, required: PermissionMode, warning: str | None = None
    ) -> PermissionOutcome:
        self.calls.append((call.input.get("path"), warning))
        return self._decisions.pop(0) if len(self._decisions) > 1 else self._decisions[0]


@pytest.fixture(autouse=True)
def _files(tmp_path: Path) -> None:
    for name in ("a.py", "b.py", "c.ts", "notes.txt"):
        (tmp_path / name).write_text("x = 1\n", encoding="utf-8")


def _diag(path: str | None = None) -> dict[str, object]:
    return {"action": "diagnostics"} if path is None else {"action": "diagnostics", "path": path}


def _runtime(
    tmp_path: Path,
    inputs: list[dict[str, object]],
    *,
    lsp: LspConfig = SERVERS,
    policy: PermissionPolicy | PermissionResolver | None = None,
    servers: _FakeServers | None = None,
) -> ConversationRuntime:
    steps: list[BaseException | list[AssistantEvent | BaseException]] = [
        [ToolUse(id=f"l{n}", name="lsp", input=inp), MessageStop()] for n, inp in enumerate(inputs)
    ]
    steps.append([TextDelta(text="done"), MessageStop()])
    return ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=ScriptedFailureProvider(steps),  # type: ignore[arg-type]
        tools=default_registry(),
        policy=policy or default_policy(WW),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=RuntimeConfig(max_iterations=8, lsp=lsp),
        lsp=servers if servers is not None else _FakeServers(tmp_path),
    )


async def _run(
    rt: ConversationRuntime, prompter: object
) -> tuple[TurnSummary, list[ToolResultBlock]]:
    summary = await rt.run_turn("go", prompter=prompter)  # type: ignore[arg-type]
    results = [b for m in rt.session.messages for b in m.blocks if isinstance(b, ToolResultBlock)]
    return summary, results


def test_confirm_start_is_on_by_default() -> None:
    assert LspConfig().confirm_start is True


async def test_each_language_asks_once(tmp_path: Path) -> None:
    prompter = _Prompter(YES)
    rt = _runtime(tmp_path, [_diag("a.py"), _diag("b.py"), _diag("c.ts")])
    summary, results = await _run(rt, prompter)
    assert prompter.calls == [
        ("a.py", "first python language server this session: sh"),
        ("c.ts", "first typescript language server this session: sh"),
    ]
    assert [r.output for r in results] == ["ok a.py", "ok b.py", "ok c.ts"]
    assert summary.lsp_start_asks == 2


async def test_declining_asks_again_next_time(tmp_path: Path) -> None:
    prompter = _Prompter(NO, YES)
    _, results = await _run(_runtime(tmp_path, [_diag("a.py"), _diag("b.py")]), prompter)
    assert [path for path, _ in prompter.calls] == ["a.py", "b.py"]
    assert results[0].is_error and results[1].output == "ok b.py"


async def test_noninteractive_refuses_and_names_the_setting(tmp_path: Path) -> None:
    servers = _FakeServers(tmp_path)
    rt = _runtime(tmp_path, [_diag("a.py")], servers=servers)
    summary, [result] = await _run(rt, DenyPrompter())
    assert result.is_error
    assert result.output.startswith("permission required but running non-interactively: ")
    assert "lsp.confirm_start (not a tool argument)" in result.output
    assert servers.paths == [] and summary.lsp_start_asks == 1


async def test_confirm_start_off_does_not_ask(tmp_path: Path) -> None:
    prompter = _Prompter()
    lsp = LspConfig(servers=SERVERS.servers, confirm_start=False)
    summary, [result] = await _run(_runtime(tmp_path, [_diag("a.py")], lsp=lsp), prompter)
    assert prompter.calls == [] and result.output == "ok a.py" and summary.lsp_start_asks == 0


@pytest.mark.parametrize(
    ("inp", "lsp"),
    [
        (_diag(), SERVERS),
        (_diag("notes.txt"), SERVERS),
        (_diag("a.py"), LspConfig(servers={"python": ["no-such-language-server"]})),
    ],
)
async def test_no_card_when_no_server_would_start(
    tmp_path: Path, inp: dict[str, object], lsp: LspConfig
) -> None:
    prompter = _Prompter()
    summary, [result] = await _run(_runtime(tmp_path, [inp], lsp=lsp), prompter)
    assert prompter.calls == [] and not result.is_error and summary.lsp_start_asks == 0


async def test_approval_is_shared_with_runtimes_on_the_same_resolver(tmp_path: Path) -> None:
    resolver = PermissionResolver(base=default_policy(WW))
    await _run(_runtime(tmp_path, [_diag("a.py")], policy=resolver), _Prompter(YES))
    _, [child] = await _run(_runtime(tmp_path, [_diag("b.py")], policy=resolver), DenyPrompter())
    assert not child.is_error and child.output == "ok b.py"

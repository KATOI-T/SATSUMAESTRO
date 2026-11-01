"""会話体験（権限カード・3層許可・状態描画）のテスト。"""

from __future__ import annotations

import io
import re

from rich.console import Console

from satsumaestro.cli.render import (
    RichRenderer,
    _format_tokens,
    action_preview,
    change_preview,
)
from satsumaestro.cli.repl import InteractivePrompter
from satsumaestro.engine.events import TextDelta, Thinking, ToolUse, Usage
from satsumaestro.engine.turn import USER_DENY_ORIGIN, paused, phase
from satsumaestro.security.allowlist import (
    MemoryLayer,
    PermissionResolver,
    signature_for,
)
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import PermissionPolicy, PermissionRule, RuleMatch
from satsumaestro.session.models import ToolResultBlock

_DANGER = PermissionMode.DANGER_FULL_ACCESS


def _console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, width=100)


def _out(console: Console) -> str:
    assert isinstance(console.file, io.StringIO)
    return console.file.getvalue()


def _bash(cmd: str) -> ToolUse:
    return ToolUse(id="1", name="bash", input={"command": cmd})


def test_prompter_yes_no_empty(monkeypatch) -> None:
    p = InteractivePrompter(_console())
    monkeypatch.setattr(p.console, "input", lambda *a, **k: "y")
    assert p.decide(_bash("ls"), _DANGER).decision == "allow"
    monkeypatch.setattr(p.console, "input", lambda *a, **k: "n")
    assert p.decide(_bash("ls"), _DANGER).decision == "deny"
    monkeypatch.setattr(p.console, "input", lambda *a, **k: "")
    assert p.decide(_bash("ls"), _DANGER).decision == "deny"


def test_prompter_always_flags_remember(monkeypatch) -> None:
    p = InteractivePrompter(_console())
    monkeypatch.setattr(p.console, "input", lambda *a, **k: "a")
    out = p.decide(_bash("ls"), _DANGER)
    assert out.decision == "allow" and out.remember is True


def test_prompter_interrupt_denies(monkeypatch) -> None:
    p = InteractivePrompter(_console())

    def _interrupt(*_a: object, **_k: object) -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr(p.console, "input", _interrupt)
    assert p.decide(_bash("ls"), _DANGER).decision == "deny"


def test_prompter_marks_user_denials(monkeypatch) -> None:
    p = InteractivePrompter(_console())
    for answer in ("n", "", "no"):
        monkeypatch.setattr(p.console, "input", lambda *a, _x=answer, **k: _x)
        out = p.decide(_bash("ls"), _DANGER)
        assert out.decision == "deny" and out.origin == USER_DENY_ORIGIN

    def _interrupt(*_a: object, **_k: object) -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr(p.console, "input", _interrupt)
    assert p.decide(_bash("ls"), _DANGER).origin == USER_DENY_ORIGIN
    monkeypatch.setattr(p.console, "input", lambda *a, **k: "y")
    assert p.decide(_bash("ls"), _DANGER).origin is None


def _resolver(base: PermissionPolicy, memory_path=None) -> PermissionResolver:
    layers = [MemoryLayer(memory_path, writable=True)] if memory_path is not None else []
    return PermissionResolver(base, layers)


def test_policy_deny_is_absolute_floor(tmp_path) -> None:
    base = PermissionPolicy(
        mode=_DANGER,
        deny=[PermissionRule(match=RuleMatch(tool="bash", field="command", regex="rm -rf"))],
    )
    mem = tmp_path / ".satsuma" / "memory.md"
    mem.parent.mkdir(parents=True)
    mem.write_text("## Allow\n- bash when command re: rm\n", encoding="utf-8")
    r = _resolver(base, mem)
    assert r.authorize("bash", {"command": "rm -rf /tmp"}, _DANGER).decision == "deny"


def test_memory_overrides_policy_ask(tmp_path) -> None:
    base = PermissionPolicy(mode=PermissionMode.READ_ONLY)
    assert base.authorize("bash", {"command": "ls"}, _DANGER).decision == "ask"
    mem = tmp_path / ".satsuma" / "memory.md"
    mem.parent.mkdir(parents=True)
    mem.write_text("## Allow\n- bash when command re: \\Als\\Z\n", encoding="utf-8")
    r = _resolver(base, mem)
    assert r.authorize("bash", {"command": "ls"}, _DANGER).decision == "allow"
    assert r.authorize("bash", {"command": "whoami"}, _DANGER).decision == "ask"


def test_session_layer_allows(tmp_path) -> None:
    base = PermissionPolicy(mode=PermissionMode.READ_ONLY)
    r = _resolver(base)
    r.session.add(signature_for("bash", {"command": "pwd"}))
    out = r.authorize("bash", {"command": "pwd"}, _DANGER)
    assert out.decision == "allow" and "session" in out.reason


def test_record_persists_and_reloads(tmp_path) -> None:
    base = PermissionPolicy(mode=PermissionMode.READ_ONLY)
    mem = tmp_path / ".satsuma" / "memory.md"
    r = _resolver(base, mem)
    r.record("bash", {"command": "pytest -q"})

    assert r.session.matches("bash", {"command": "pytest -q"})
    text = mem.read_text(encoding="utf-8")
    assert "## Allow" in text and "bash when command re:" in text
    reloaded = MemoryLayer(mem, writable=True)
    assert reloaded.match("bash", {"command": "pytest -q"}).decision == "allow"
    assert reloaded.match("bash", {"command": "pytest -x"}) is None


def test_memory_layer_can_deny_and_ask(tmp_path) -> None:
    base = PermissionPolicy(mode=_DANGER)
    mem = tmp_path / ".satsuma" / "memory.md"
    mem.parent.mkdir(parents=True)
    mem.write_text(
        "## Deny\n- bash when command re: curl\n## Ask\n- bash when command re: git\n",
        encoding="utf-8",
    )
    r = _resolver(base, mem)
    assert r.authorize("bash", {"command": "curl x"}, _DANGER).decision == "deny"
    assert r.authorize("bash", {"command": "git push"}, _DANGER).decision == "ask"


def test_signature_tool_level_for_fieldless() -> None:
    rule = signature_for("some_mcp_tool", {"x": 1})
    assert rule.tool == "some_mcp_tool" and rule.field is None
    assert rule.matches("some_mcp_tool", {"anything": True})


def test_action_preview_bash_shows_command() -> None:
    console = _console()
    console.print(action_preview(_bash("git push origin main"), "danger-full-access"))
    text = _out(console)
    assert "権限確認" in text and "git push" in text


def _edit_call() -> ToolUse:
    return ToolUse(id="1", name="edit_file", input={"path": "a.py", "old": "x = 1", "new": "x = 2"})


def test_action_preview_edit_summarizes_and_details_on_demand() -> None:
    console = _console()
    console.print(action_preview(_edit_call(), "workspace-write"))
    text = _out(console)
    assert "a.py" in text and "+1" in text and "-1" in text
    assert "-x = 1" not in text

    detailed = _console()
    detailed.print(action_preview(_edit_call(), "workspace-write", detailed=True))
    body = _out(detailed)
    assert "-x = 1" in body and "+x = 2" in body


def test_format_tokens_step() -> None:
    assert _format_tokens(0) == "0"
    assert _format_tokens(999) == "999"
    assert _format_tokens(1000) == "1K"
    assert _format_tokens(1234) == "1.23K"
    assert _format_tokens(12345) == "12.3K"


def test_turn_tokens_prefers_actual_usage() -> None:
    r = RichRenderer(console=_console())
    r.start_turn()
    r.on_assistant_event(TextDelta(text="x" * 40))
    assert r._turn_tokens() == 10
    r.on_assistant_event(Usage(output_tokens=57))
    assert r._turn_tokens() == 57
    r._finalize_segment()
    r.on_assistant_event(Usage(output_tokens=13))
    assert r._turn_tokens() == 70


def test_token_hidden_while_zero() -> None:
    r = RichRenderer(console=_console())
    r.start_turn()
    assert r._turn_tokens() == 0
    status = r._status_row()
    from rich.console import Console as _C

    c = _C(file=io.StringIO(), force_terminal=False, width=60)
    c.print(status)
    assert isinstance(c.file, io.StringIO)
    assert "出力" not in c.file.getvalue() and "tok" not in c.file.getvalue()


def test_operations_are_blank_separated() -> None:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=80)
    r = RichRenderer(console=console)
    r.start_turn()
    r.on_assistant_event(Thinking(text="think"))
    r.on_assistant_event(TextDelta(text="# Title"))
    r.on_tool_start(ToolUse(id="1", name="read_file", input={"path": "x"}))
    r.on_tool_result(
        ToolResultBlock(tool_use_id="1", tool_name="read_file", output="body", is_error=False)
    )
    r.on_assistant_event(TextDelta(text="answer"))
    r.end_turn()
    out = buf.getvalue()
    assert "Title" in out and "・読み込み" in out and "answer" in out
    assert "\n\n" in out


def _render_status(r: RichRenderer) -> str:
    """状態行を非 TTY Console へ描いて文字列にする（テスト専用の覗き窓）。"""
    status = r._status_row()
    if status is None:
        return ""
    c = Console(file=io.StringIO(), force_terminal=False, width=80)
    c.print(status)
    assert isinstance(c.file, io.StringIO)
    return c.file.getvalue()


def test_tool_execution_keeps_status_line() -> None:
    r = RichRenderer(console=_console())
    r.start_turn()
    r.on_tool_start(ToolUse(id="c1", name="bash", input={"command": "make test"}))
    assert "コマンド実行中…" in _render_status(r)
    r.on_tool_result(ToolResultBlock(tool_use_id="c1", tool_name="bash", output="ok"))
    assert "コマンド実行中" not in _render_status(r)
    assert "考え中…" in _render_status(r)


def test_phase_label_overrides_state_label() -> None:
    r = RichRenderer(console=_console())
    r.start_turn()
    r.on_assistant_event(TextDelta(text="hello"))
    assert "応答生成中…" in _render_status(r)
    r.on_phase("履歴を要約中…")
    assert "履歴を要約中…" in _render_status(r)
    r.on_phase(None)
    assert "応答生成中…" in _render_status(r)


def test_elapsed_seconds_appear_after_grace(monkeypatch) -> None:
    r = RichRenderer(console=_console())
    r.start_turn()
    r.on_phase("bash 実行中…")
    assert re.search(r"\d+s", _render_status(r)) is None
    import satsumaestro.cli.render as render_mod

    base = r._phase_since
    monkeypatch.setattr(render_mod.time, "monotonic", lambda: base + 12.0)
    assert "12s" in _render_status(r)


def test_phase_is_not_carried_over_between_turns() -> None:
    r = RichRenderer(console=_console())
    r.start_turn()
    r.on_phase("bash 実行中…")
    r.end_turn()
    r.start_turn()
    assert "実行中" not in _render_status(r)


def test_paused_folds_live_for_interactive_input() -> None:
    r = RichRenderer(console=Console(file=io.StringIO(), force_terminal=True, width=80))
    r.start_turn()
    r.on_phase("bash 実行中…")
    assert r._live is not None
    with paused(r):
        assert r._live is None
    assert r._live is not None
    r.end_turn()


def test_phase_and_paused_are_noop_for_plain_renderer() -> None:
    class _Plain:
        def on_assistant_event(self, event: object) -> None: ...
        def on_tool_start(self, call: object) -> None: ...
        def on_tool_result(self, block: object) -> None: ...

    plain = _Plain()
    phase(plain, "履歴を要約中…")
    with paused(plain):
        pass


def test_change_preview_shows_diff_before_execution() -> None:
    console = Console(file=io.StringIO(), force_terminal=True, width=100)
    r = RichRenderer(console=console)
    r.start_turn()
    r.on_tool_start(_edit_call())
    r.end_turn()
    assert isinstance(console.file, io.StringIO)
    text = console.file.getvalue()
    assert "-x = 1" in text and "+x = 2" in text


def test_change_preview_covers_write_file_and_skips_others() -> None:
    write = ToolUse(id="1", name="write_file", input={"path": "b.py", "content": "a\nb\n"})
    body = change_preview(write)
    assert body is not None
    c = Console(file=io.StringIO(), force_terminal=False, width=80)
    c.print(body)
    assert isinstance(c.file, io.StringIO)
    assert "b.py" in c.file.getvalue()
    assert change_preview(_bash("ls")) is None
    assert change_preview(ToolUse(id="1", name="read_file", input={"path": "a.py"})) is None


def test_change_preview_not_emitted_when_not_tty() -> None:
    console = _console()
    r = RichRenderer(console=console)
    r.start_turn()
    r.on_tool_start(_edit_call())
    r.end_turn()
    assert "-x = 1" not in _out(console)

"""永続許可してはいけない操作（削除系など）では、権限カードに a=永続許可 を出さない。"""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from satsumaestro.cli.repl import InteractivePrompter
from satsumaestro.engine.events import ToolUse
from satsumaestro.engine.turn import DETAIL_REQUEST_ORIGIN, PermissionDetail
from satsumaestro.security.allowlist import (
    MemoryLayer,
    PermissionResolver,
    rule_to_line,
    signature_for,
)
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import PermissionPolicy
from satsumaestro.security.remember_guard import (
    DELETION,
    DESTRUCTIVE,
    PRIVILEGE,
    UNPARSABLE,
    remember_blocker,
)

_DANGER = PermissionMode.DANGER_FULL_ACCESS
_FOUR = "[y=許可 / n=拒否 / a=永続許可 / d=詳細]"
_THREE = "[y=許可 / n=拒否 / d=詳細]"
_AFTER_DETAIL = "[y=許可 / n=拒否]"


@pytest.mark.parametrize(
    ("command", "reason"),
    [
        ("rm old.txt", DELETION),
        ("rm -rf build", DELETION),
        ("cd x && rm y", DELETION),
        ("find . -name '*.o' -delete", DELETION),
        ("find . -exec sh -c 'rm {}' ';'", DELETION),
        ("git rm a.py", DELETION),
        ("bash -c 'rm x'", DELETION),
        ("xargs rm", DELETION),
        ("env FOO=1 rm x", DELETION),
        ("timeout 5 rm x", DELETION),
        ("echo $(rm x)", DELETION),
        ("echo `rm x`", DELETION),
        ("cat <(rm x)", DELETION),
        ("sudo apt install x", PRIVILEGE),
        ("sudo ls", PRIVILEGE),
        ("shred f", DESTRUCTIVE),
        ("ls && shred x", DESTRUCTIVE),
        ("mkfs.ext4 /dev/sdb", DESTRUCTIVE),
        ("dd if=/dev/zero of=x", DESTRUCTIVE),
        ("echo 'unclosed", UNPARSABLE),
    ],
)
def test_blocks_deletion_privilege_and_destructive(command: str, reason: str) -> None:
    assert remember_blocker("bash", {"command": command}) == reason


@pytest.mark.parametrize(
    ("command", "sub"),
    [
        ("git push --force", "push"),
        ("git push -f origin x", "push"),
        ("git push --force-with-lease", "push"),
        ("git push origin +main", "push"),
        ("git push origin :old", "push"),
        ("git push -d origin x", "push"),
        ("git -C repo reset --hard", "reset"),
        ("git clean -fd", "clean"),
        ("git branch -D x", "branch"),
        ("git branch -d -f x", "branch"),
        ("git checkout -- a.py", "checkout"),
        ("git checkout .", "checkout"),
        ("git switch -f main", "switch"),
        ("git restore a.py", "restore"),
        ("git stash drop", "stash"),
        ("git stash clear", "stash"),
    ],
)
def test_blocks_irreversible_git(command: str, sub: str) -> None:
    reason = remember_blocker("bash", {"command": command})
    assert reason == f"git:{sub}"


@pytest.mark.parametrize(
    "command",
    [
        "ls",
        "npm test",
        "touch built.txt",
        "grep -rn rm .",
        "echo 'rm x'",
        "cat README.md | grep sudo",
        "git rm --cached a",
        "git push",
        "git push origin main",
        "git push -u origin feat",
        "git reset HEAD~1",
        "git clean -n",
        "git branch -d x",
        "git checkout main",
        "git checkout -b feat",
        "git restore --staged a.py",
        "git stash",
        "git -C repo status",
    ],
)
def test_ordinary_commands_can_be_remembered(command: str) -> None:
    assert remember_blocker("bash", {"command": command}) is None


def test_tools() -> None:
    assert remember_blocker("delete_path", {"path": "x"}) == DELETION
    for name in ("write_file", "edit_file", "append_file", "web_fetch"):
        assert remember_blocker(name, {"path": "a.txt"}) is None
    assert remember_blocker("bash", {}) is None


def _console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, width=120)


def _out(console: Console) -> str:
    assert isinstance(console.file, io.StringIO)
    return console.file.getvalue()


def _answers(monkeypatch: pytest.MonkeyPatch, p: InteractivePrompter, *answers: str) -> list[str]:
    """``console.input`` に答えを順に返させ、出たプロンプトを記録する。"""
    prompts: list[str] = []
    it: Iterator[str] = iter(answers)

    def fake_input(prompt: str = "", **_: Any) -> str:
        prompts.append(prompt)
        return next(it)

    monkeypatch.setattr(p.console, "input", fake_input)
    return prompts


@pytest.mark.parametrize(
    "call",
    [
        ToolUse(id="1", name="delete_path", input={"path": "old.txt"}),
        ToolUse(id="1", name="bash", input={"command": "rm -rf build"}),
        ToolUse(id="1", name="bash", input={"command": "git push --force"}),
    ],
)
def test_card_hides_always_and_reasks(monkeypatch: pytest.MonkeyPatch, call: ToolUse) -> None:
    p = InteractivePrompter(_console())
    prompts = _answers(monkeypatch, p, "a", "always", "y")
    out = p.decide(call, _DANGER)
    assert out.decision == "allow" and out.remember is False
    assert [_THREE in s for s in prompts] == [True, True, True]
    assert not any("a=永続許可" in s for s in prompts)
    shown = _out(p.console)
    assert "永続許可できません）" in shown
    assert shown.count("この操作は永続許可できません。許可するときは y を押してください") == 2


def test_card_reason_line_names_the_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    p = InteractivePrompter(_console())
    _answers(monkeypatch, p, "n")
    p.decide(ToolUse(id="1", name="bash", input={"command": "git reset --hard"}), _DANGER)
    assert "（取り消せない git 操作（git reset）のため永続許可できません）" in _out(p.console)


def test_card_still_denies_on_n(monkeypatch: pytest.MonkeyPatch) -> None:
    p = InteractivePrompter(_console())
    _answers(monkeypatch, p, "a", "n")
    out = p.decide(ToolUse(id="1", name="delete_path", input={"path": "x"}), _DANGER)
    assert out.decision == "deny" and out.remember is False


def test_ordinary_card_keeps_always(monkeypatch: pytest.MonkeyPatch) -> None:
    p = InteractivePrompter(_console())
    prompts = _answers(monkeypatch, p, "a")
    out = p.decide(ToolUse(id="1", name="bash", input={"command": "npm test"}), _DANGER)
    assert out.decision == "allow" and out.remember is True
    assert len(prompts) == 1 and _FOUR in prompts[0]
    assert "永続許可できません" not in _out(p.console)


def _detail() -> PermissionDetail:
    return PermissionDetail(explanation="old.txt を消します。", failure=None)


def test_after_detail_hides_always(monkeypatch: pytest.MonkeyPatch) -> None:
    p = InteractivePrompter(_console())
    call = ToolUse(id="1", name="delete_path", input={"path": "old.txt"})
    prompts = _answers(monkeypatch, p, "d", "a", "y")
    assert p.decide(call, _DANGER).origin == DETAIL_REQUEST_ORIGIN
    out = p.decide_with_detail(call, _DANGER, None, _detail())
    assert out.decision == "allow" and out.remember is False
    assert _THREE in prompts[0]
    assert prompts[1:] == [f"許可しますか？ {_AFTER_DETAIL}: "] * 2


def test_after_detail_ordinary_keeps_always(monkeypatch: pytest.MonkeyPatch) -> None:
    p = InteractivePrompter(_console())
    call = ToolUse(id="1", name="bash", input={"command": "npm test"})
    prompts = _answers(monkeypatch, p, "a")
    out = p.decide_with_detail(call, _DANGER, None, _detail())
    assert out.remember is True
    assert prompts == ["許可しますか？ [y=許可 / n=拒否 / a=永続許可]: "]


def _resolver(memory: Path) -> PermissionResolver:
    base = PermissionPolicy(mode=PermissionMode.READ_ONLY)
    return PermissionResolver(base, [MemoryLayer(memory, writable=True)])


def test_record_skips_blocked_calls(tmp_path: Path) -> None:
    mem = tmp_path / ".satsuma" / "memory.md"
    r = _resolver(mem)
    r.record("bash", {"command": "rm -rf build"})
    r.record("delete_path", {"path": "old.txt"})
    r.record("bash", {"command": "git push --force"})
    assert not r.session.matches("bash", {"command": "rm -rf build"})
    assert not r.session.matches("delete_path", {"path": "old.txt"})
    assert not mem.exists()
    r.record("bash", {"command": "npm test"})
    assert "npm" in mem.read_text(encoding="utf-8")


def _remembered(*calls: tuple[str, dict[str, object]]) -> str:
    """過去の a で memory.md に残った行（今の record では書かれないものも含む）。"""
    lines = [f"- {rule_to_line(signature_for(name, inp))}" for name, inp in calls]
    return "## Allow\n" + "\n".join(lines) + "\n"


def test_remembered_allow_is_ignored_for_blocked_calls(tmp_path: Path) -> None:
    mem = tmp_path / ".satsuma" / "memory.md"
    mem.parent.mkdir(parents=True)
    rm = {"command": "rm -rf build"}
    push = {"command": "git push --force"}
    ls = {"command": "ls"}
    mem.write_text(
        _remembered(("bash", rm), ("bash", push), ("delete_path", {"path": "x"}), ("bash", ls)),
        encoding="utf-8",
    )
    r = _resolver(mem)
    assert r.authorize("bash", rm, _DANGER).decision == "ask"
    assert r.authorize("bash", push, _DANGER).decision == "ask"
    assert r.authorize("delete_path", {"path": "x"}, _DANGER).decision == "ask"
    assert r.authorize("bash", ls, _DANGER).decision == "allow"


def test_session_allow_is_ignored_for_blocked_calls(tmp_path: Path) -> None:
    r = _resolver(tmp_path / "memory.md")
    rm = {"command": "rm old.txt"}
    r.session.add(signature_for("bash", rm))
    assert r.authorize("bash", rm, _DANGER).decision == "ask"


def test_memory_deny_still_applies_to_blocked_calls(tmp_path: Path) -> None:
    mem = tmp_path / "memory.md"
    mem.write_text("## Deny\n- bash when command re: rm\n", encoding="utf-8")
    r = _resolver(mem)
    assert r.authorize("bash", {"command": "rm x"}, _DANGER).decision == "deny"

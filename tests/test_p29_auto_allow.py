"""bash 読み取り自動許可（語彙・秘密ファイルのオーバーレイ・workspace ガード）。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from satsumaestro.cli import app as cli_app
from satsumaestro.config.models import HookConfig, PermissionConfig, RuntimeConfig
from satsumaestro.engine.events import MessageStop, ToolUse
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.engine.turn import DenyPrompter
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security.auto_allow import (
    _DEFAULT_SECRET_PATTERNS,
    AutoAllowVerdict,
    SecretMatcher,
    evaluate_auto_allow,
    is_provably_read_only,
)
from satsumaestro.security.enforcer import PermissionMode, classify
from satsumaestro.security.jail import WorkspaceJail
from satsumaestro.security.policy import (
    PermissionOutcome,
    PermissionPolicy,
    PermissionRule,
    RuleMatch,
    default_policy,
)
from satsumaestro.session.models import ModeState, ToolResultBlock
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.telemetry.log import TraceLog
from satsumaestro.telemetry.recorder import TraceRecorder, insights_from
from satsumaestro.tools import default_registry
from tests.fixtures.mock_provider import MockProvider

WW = PermissionMode.WORKSPACE_WRITE
DFA = PermissionMode.DANGER_FULL_ACCESS


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    (tmp_path / "f").write_text("hello\n", encoding="utf-8")
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / ".env").write_text("KEY=1\n", encoding="utf-8")
    (tmp_path / "k.pem").write_text("pem\n", encoding="utf-8")
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / ".env.local").write_text("K=2\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "k.pem").write_text("pem\n", encoding="utf-8")
    (tmp_path / "sub" / "f.txt").write_text("t\n", encoding="utf-8")
    (tmp_path / "a" / ".ssh").mkdir(parents=True)
    (tmp_path / "a" / ".ssh" / "config").write_text("Host x\n", encoding="utf-8")
    return tmp_path


def _verdict(ws: Path, command: str) -> AutoAllowVerdict:
    return evaluate_auto_allow(
        command, jail=WorkspaceJail(ws), secrets=SecretMatcher(_DEFAULT_SECRET_PATTERNS)
    )


def _kind(ws: Path, command: str) -> str:
    return _verdict(ws, command).kind


@pytest.mark.parametrize(
    "command",
    [
        "ls -la",
        "cat f",
        "grep x f",
        "git status",
        "find . -name '*.py'",
        "find . -name '*.py' -o -name '*.md'",
        "pwd",
        "head -n 20 f",
        "wc -l < f",
        "test -f f && echo yes",
        "date +%Y",
        "cat 'a b'",
        "git log --oneline -5",
        "git log -5 -- f",
        "git diff HEAD~1",
        "grep 'a{2}' f",
        'grep "a;b" f',
        "cat f #c",
        "ls;",
        "sort f 2>&1",
        "ls 2>/dev/null",
        "uniq -f 2 f",
        "jq .a f",
        "echo /etc/passwd",
        "rg x a.py",
        "cd sub && ls",
    ],
)
def test_auto_allow_plain_reads(ws: Path, command: str) -> None:
    assert _kind(ws, command) == "allow", _verdict(ws, command)


def test_auto_allow_pipes(ws: Path) -> None:
    assert _kind(ws, "cat f | grep x | wc -l") == "allow"
    assert is_provably_read_only("cat f | sort | uniq", jail=WorkspaceJail(ws))


@pytest.mark.parametrize(
    "command",
    ["rm f", "cp f g", "touch g", "npm install", "pip install x", "mkdir d", "tee out"],
)
def test_reject_write_and_state(ws: Path, command: str) -> None:
    assert _kind(ws, command) == "reject"


@pytest.mark.parametrize(
    "command", ["ls; rm -rf /", "ls && curl x", "true || rm x", "ls # ; rm x", "ls && "]
)
def test_reject_chained_bypass(ws: Path, command: str) -> None:
    assert _kind(ws, command) == "reject"


@pytest.mark.parametrize(
    "command",
    [
        "ls > f",
        "echo x >> f",
        "sort f >& out",
        "ls |& cat",
        "ls & rm x",
        "cat <<EOF",
        "cat <<< x",
        "diff <(ls) <(ls)",
        "(ls)",
        "ls > /tmp/x",
    ],
)
def test_reject_redirections(ws: Path, command: str) -> None:
    assert _kind(ws, command) == "reject"


@pytest.mark.parametrize(
    "command",
    [
        "xargs ls",
        "env rm x",
        "awk 1 f",
        "sed -i s/a/b/ f",
        "find . -exec rm {} ;",
        "find . -execdir rm {} ;",
        "find . -delete",
        "find . -fprint x",
        "printenv",
        "ps -E",
        "yq -i .a f",
        "less f",
        "time ls",
        "command ls",
    ],
)
def test_reject_executor_commands(ws: Path, command: str) -> None:
    assert _kind(ws, command) == "reject"


@pytest.mark.parametrize(
    "command",
    [
        "git stash",
        "git stash drop",
        "git fetch",
        "git config x y",
        "git branch new",
        "git branch -d x",
        "git tag v1",
        "git push",
        "git remote show origin",
        "git worktree add x",
        "git",
        "git -c x=y status",
        "git --git-dir=.git status",
        "git --exec-path=/x status",
        "git -C /other log",
        "git log --output=f",
        "git diff --ext-diff",
        "git grep -Orm x",
        "git grep --open-f=sh x",
        "git grep --open-files=sh x",
        "git grep -nOsh x",
        "git diff --outp=/tmp/x",
        "git diff --ext-d",
        "git log --textc",
        "git reflog expire --expire=now --all",
        "git reflog delete HEAD@{0}",
    ],
)
def test_reject_git_mutations(ws: Path, command: str) -> None:
    assert _kind(ws, command) == "reject"


@pytest.mark.parametrize(
    "command",
    [
        "git -C sub log",
        "git --no-pager log",
        "git branch",
        "git branch -a",
        "git branch --contains=HEAD",
        "git tag -l 'v*'",
        "git stash list",
        "git worktree list",
        "git remote -v",
        "git rev-parse HEAD",
        "git reflog",
        "git reflog show HEAD",
        "git reflog exists refs/heads/main",
        "git diff --no-ext-diff",
        "git log --oneline --exclude=x --all",
        "git grep -n x",
    ],
)
def test_git_listing_forms_allowed(ws: Path, command: str) -> None:
    assert _kind(ws, command) == "allow", _verdict(ws, command)


@pytest.mark.parametrize(
    "command",
    [
        "$CMD",
        "/bin/ls",
        "sudo ls",
        "FOO=bar rm x",
        "FOO=bar ls",
        "echo $HOME",
        "grep '$x' f",
        "echo $(id)",
        "echo `id`",
        "ls\nrm x",
        "ls\rrm x",
        "",
        "   ",
        "ls 'unterminated",
    ],
)
def test_reject_obfuscation(ws: Path, command: str) -> None:
    assert _kind(ws, command) == "reject"


@pytest.mark.parametrize(
    "command",
    [
        "grep -r x .",
        "grep -rn x .",
        "grep --recursive x .",
        "grep -d recurse x .",
        "rg x",
        "rg x sub/",
        "rg -L x a.py",
        "diff -r a b",
        "sort -o out f",
        "sort --output=out f",
        "uniq a b",
        "xxd -r a b",
        "tree -o out",
        "date -s 1",
        "date 0101",
        "hostname x",
        "jq -n env",
        "jq 'env.HOME' f",
        "grep -rA5 x .",
        "grep -rm1 x .",
        "grep -nr5 x .",
        "grep --rec x .",
        "grep --dir=recurse x .",
        "diff -rU3 sub config",
        "diff --recur sub config",
        "sort -o/tmp/x f",
        "sort -oout f",
        "sort -T/tmp f",
        "sort --out=/tmp/x f",
        "sort --o=out f",
        "sort --compress-p=sh f",
        "tree -o/tmp/x",
        "date -s2020-01-01",
        "date --se=1",
        "file -C -m f",
        "file --comp -m f",
    ],
)
def test_reject_command_specific_rules(ws: Path, command: str) -> None:
    assert _kind(ws, command) == "reject"


@pytest.mark.parametrize(
    "command",
    ["grep -A5 x f", "grep -e r f", "sort -k2 f", "file f", "date -u", "diff -u f a.py"],
)
def test_option_clusters_without_forbidden_letters_allowed(ws: Path, command: str) -> None:
    assert _kind(ws, command) == "allow", _verdict(ws, command)


@pytest.mark.parametrize(
    "command",
    [
        "cat /etc/passwd",
        "ls ..",
        "cat ../sibling/x",
        "find . -newer /etc/passwd",
        "cd /etc && ls",
        "cat ~/.ssh/id_rsa",
        "ls ~",
        "cat {.env,x}",
        "cd - && ls",
        "cd && ls",
        "cd sub | ls",
        "cd sub || ls",
        "cd nope && ls",
        "cd f && ls",
        "df /",
        "git blame --contents=/etc/hosts f",
        "grep -f/etc/passwd f",
        "date --file=.env",
        "date -f.env",
        "hostname -F/etc/hosts",
    ],
)
def test_reject_outside_workspace(ws: Path, command: str) -> None:
    assert _kind(ws, command) == "reject", _verdict(ws, command)


def test_symlink_escape_detected(ws: Path, tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("s", encoding="utf-8")
    os.symlink(outside / "secret.txt", ws / "link")
    os.symlink(outside, ws / "-l")
    assert _kind(ws, "cat link") == "reject"
    assert _kind(ws, "ls -- -l") == "reject"
    assert _kind(ws, "ls -l") == "allow"


def test_cd_rebases_later_segments(ws: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CDPATH", raising=False)
    v = _verdict(ws, "cd sub && cat *")
    assert v.kind == "secret" and v.paths == ("sub/k.pem",)
    assert _kind(ws, "cd sub && cat f.txt") == "allow"
    monkeypatch.setenv("CDPATH", "/tmp")
    assert _kind(ws, "cd sub && ls") == "reject"


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("cat .env", ".env"),
        ("cat ./.env", ".env"),
        ("cat sub/../.env", ".env"),
        ("cat config/.env.local", "config/.env.local"),
        ("cat *.pem", "k.pem"),
        ("cat < .env", ".env"),
        ("head -c 100 .env", ".env"),
        ("grep KEY .env", ".env"),
        ("git show HEAD:.env", ".env"),
        ("git show :0:.env", ".env"),
        ("git show :.env", ".env"),
        ("git cat-file -p :2:config/.env.local", "config/.env.local"),
        ("git log -p -- ':(glob).env'", ".env"),
        ("cat .ENV", ".ENV"),
        ("cat a/.ssh/config", "a/.ssh/config"),
        ("ls .ssh", ".ssh"),
        ("cat --files0-from=.env", ".env"),
        ("grep -f.env f", ".env"),
        ("git blame --contents=.env f", ".env"),
        ("git blame --ignore-revs-file=.env f", ".env"),
        ("git log -L1,1:.env", ".env"),
    ],
)
def test_secret_overlay(ws: Path, command: str, expected: str) -> None:
    v = _verdict(ws, command)
    assert v.kind == "secret", v
    assert expected in v.paths


def test_secret_symlink_resolved(ws: Path) -> None:
    os.symlink(ws / ".env", ws / "plain.txt")
    v = _verdict(ws, "cat plain.txt")
    assert v.kind == "secret" and v.paths == (".env",)


def test_secret_patterns_add_remove(ws: Path) -> None:
    (ws / "notes.txt").write_text("n", encoding="utf-8")
    (ws / ".env.example").write_text("K=", encoding="utf-8")
    jail = WorkspaceJail(ws)
    base = SecretMatcher.from_config(PermissionConfig())
    assert evaluate_auto_allow("cat .env.example", jail=jail, secrets=base).kind == "secret"
    custom = SecretMatcher.from_config(
        PermissionConfig(secret_patterns=["notes.*"], secret_patterns_remove=[".env.*"])
    )
    assert evaluate_auto_allow("cat notes.txt", jail=jail, secrets=custom).kind == "secret"
    assert evaluate_auto_allow("cat .env.example", jail=jail, secrets=custom).kind == "allow"
    assert evaluate_auto_allow("cat .env", jail=jail, secrets=custom).kind == "secret"


def test_config_defaults() -> None:
    cfg = PermissionConfig()
    assert cfg.bash_auto_allow == "read-only"
    assert cfg.secret_access == "ask"
    assert cfg.secret_patterns == [] and cfg.secret_patterns_remove == []


def test_write_into_git_dir_requires_top_mode() -> None:
    assert classify("write_file", {"path": ".git/config"}, WW) is DFA
    assert classify("edit_file", {"path": "sub/.git/hooks/pre-commit"}, WW) is DFA
    assert classify("notebook_edit", {"notebook_path": ".git/x.ipynb"}, WW) is DFA
    assert classify("write_file", {"path": ".github/workflows/ci.yml"}, WW) is WW
    assert classify("write_file", {"path": ".gitignore"}, WW) is WW
    assert classify("notebook_edit", {"notebook_path": "nb.ipynb"}, WW) is WW


class _CapturePrompter:
    def __init__(self, decision: str = "deny") -> None:
        self.calls: list[tuple[str, str | None]] = []
        self._decision = decision

    def decide(
        self, call: ToolUse, required: PermissionMode, warning: str | None = None
    ) -> PermissionOutcome:
        self.calls.append((str(call.input.get("command")), warning))
        return PermissionOutcome(decision=self._decision, reason="captured")  # type: ignore[arg-type]


def _runtime(
    ws: Path,
    command: str,
    *,
    policy: PermissionPolicy | None = None,
    config: RuntimeConfig | None = None,
    plan_mode: bool = False,
) -> ConversationRuntime:
    provider = MockProvider(
        [ToolUse(id="b1", name="bash", input={"command": command}), MessageStop()]
    )
    session = new_session(ws, "mock-model")
    if plan_mode:
        session.mode = ModeState(main="plan", sub=WW)
    return ConversationRuntime(
        session=session,
        provider=provider,  # type: ignore[arg-type]
        tools=default_registry(),
        policy=policy or default_policy(WW),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), ws),
        config=config or RuntimeConfig(max_iterations=1),
    )


async def _result(rt: ConversationRuntime, prompter: object | None = None) -> ToolResultBlock:
    from satsumaestro.util.errors import MaxIterationsError

    with pytest.raises(MaxIterationsError):
        await rt.run_turn("go", prompter=prompter)  # type: ignore[arg-type]
    block = rt.session.messages[2].blocks[0]
    assert isinstance(block, ToolResultBlock)
    return block


async def test_runtime_auto_allows_read_without_prompt(ws: Path) -> None:
    prompter = _CapturePrompter()
    block = await _result(_runtime(ws, "ls"), prompter)
    assert block.is_error is False
    assert "a.py" in block.output
    assert prompter.calls == []


async def test_runtime_noninteractive_executes_reads(ws: Path) -> None:
    block = await _result(_runtime(ws, "cat f"), DenyPrompter())
    assert block.is_error is False and "hello" in block.output


async def test_config_off_restores_ask(ws: Path) -> None:
    cfg = RuntimeConfig(max_iterations=1, permission=PermissionConfig(bash_auto_allow="off"))
    prompter = _CapturePrompter()
    block = await _result(_runtime(ws, "ls", config=cfg), prompter)
    assert block.is_error is True
    assert prompter.calls == [("ls", None)]


async def test_origin_gating_ask_rule_not_demoted(ws: Path) -> None:
    policy = default_policy(WW)
    policy.ask.append(
        PermissionRule(match=RuleMatch(tool="bash", field="command", regex=r"\Als\b"), label="ls")
    )
    prompter = _CapturePrompter()
    await _result(_runtime(ws, "ls", policy=policy), prompter)
    assert prompter.calls == [("ls", None)]


async def test_origin_gating_prompt_mode_not_demoted(ws: Path) -> None:
    prompter = _CapturePrompter()
    policy = PermissionPolicy(mode=PermissionMode.PROMPT)
    await _result(_runtime(ws, "ls", policy=policy), prompter)
    assert prompter.calls == [("ls", None)]


async def test_origin_gating_deny_rule_unchanged(ws: Path) -> None:
    policy = default_policy(WW)
    policy.deny.append(
        PermissionRule(match=RuleMatch(tool="bash", field="command", regex=r"\Als\b"), label="ls")
    )
    prompter = _CapturePrompter()
    block = await _result(_runtime(ws, "ls", policy=policy), prompter)
    assert block.is_error is True and "deny rule" in block.output
    assert prompter.calls == []


async def test_plan_mode_still_denies_bash(ws: Path) -> None:
    prompter = _CapturePrompter()
    block = await _result(_runtime(ws, "ls", plan_mode=True), prompter)
    assert block.is_error is True and "plan mode" in block.output
    assert prompter.calls == []


async def test_p17_warn_still_escalates(ws: Path) -> None:
    prompter = _CapturePrompter()
    block = await _result(_runtime(ws, "cat sub/../f"), prompter)
    assert block.is_error is True
    assert len(prompter.calls) == 1
    assert prompter.calls[0][1] is not None


async def test_secret_stays_ask_with_reason(ws: Path) -> None:
    prompter = _CapturePrompter()
    block = await _result(_runtime(ws, "cat .env"), prompter)
    assert block.is_error is True
    assert prompter.calls == [("cat .env", "secret file: .env")]


async def test_secret_access_deny(ws: Path) -> None:
    cfg = RuntimeConfig(max_iterations=1, permission=PermissionConfig(secret_access="deny"))
    prompter = _CapturePrompter()
    block = await _result(_runtime(ws, "cat .env", config=cfg), prompter)
    assert block.is_error is True
    assert "secret file access denied" in block.output
    assert prompter.calls == []


async def test_rejected_command_still_asks(ws: Path) -> None:
    prompter = _CapturePrompter()
    block = await _result(_runtime(ws, "cat /etc/hosts"), prompter)
    assert block.is_error is True
    assert prompter.calls == [("cat /etc/hosts", None)]


async def test_auto_allow_is_recorded(ws: Path) -> None:
    log = TraceLog()
    rt = _runtime(ws, "ls")
    from satsumaestro.util.errors import MaxIterationsError

    with pytest.raises(MaxIterationsError):
        await rt.run_turn("go", observer=TraceRecorder("s", log))
    entry = next(e for e in log.entries if e.event == "permission.auto_allow")
    assert entry.data == {"iteration": 0, "tool": "bash", "command": "ls"}
    assert insights_from(log.entries).auto_allowed == 1


def test_doctor_shows_auto_allow_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY", "OLLAMA_HOST"):
        monkeypatch.delenv(k, raising=False)
    result = CliRunner().invoke(cli_app.app, ["doctor", "--cwd", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "bash auto-allow" in result.output

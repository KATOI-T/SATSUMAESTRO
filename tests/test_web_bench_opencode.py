"""Web-Bench ドライバ（scripts/eval_web_bench.py）の OpenCode の回し方のテスト。"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_web_bench_driver import _load

_EVENTS = [
    {"type": "step_start", "part": {"type": "step-start"}},
    {
        "type": "tool_use",
        "part": {"tool": "write", "state": {"status": "completed", "input": {"path": "a.js"}}},
    },
    {
        "type": "step_finish",
        "part": {"reason": "tool-calls", "tokens": {"input": 6969, "output": 112, "reasoning": 8}},
    },
    {
        "type": "tool_use",
        "part": {
            "tool": "shell",
            "state": {"status": "error", "error": "Permission denied: shell"},
        },
    },
    {
        "type": "step_finish",
        "part": {"reason": "tool-calls", "tokens": {"input": 7090, "output": 42}},
    },
    {"type": "text", "part": {"text": "done"}},
    {"type": "step_finish", "part": {"reason": "stop", "tokens": {"input": 7200, "output": 30}}},
]


def _jsonl(events: list[dict[str, object]]) -> str:
    return "\n".join(json.dumps(e) for e in events) + "\n"


def test_config_points_at_ollama_and_denies_deletion() -> None:
    mod = _load()
    cfg = mod.opencode_config("gpt-oss:120b", {"OLLAMA_HOST": "http://192.0.2.10:11434/v1"})
    ollama = cfg["provider"]["ollama"]
    assert ollama["options"]["baseURL"] == "http://192.0.2.10:11434/v1"
    assert ollama["models"] == {"gpt-oss:120b": {"name": "gpt-oss:120b"}}
    shell = cfg["permission"]["bash"]
    assert list(shell)[0] == "*" and shell["*"] == "allow"
    for pattern in ("rm", "rm *", "rmdir *", "find * -delete*", "git clean*", "sudo *"):
        assert shell[pattern] == "deny"
    assert cfg["permission"]["external_directory"] == "deny"


def test_config_and_records_live_next_to_the_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load()
    monkeypatch.setattr(mod, "ollama_env", lambda: {"OLLAMA_HOST": "http://localhost:11434"})
    home = tmp_path / "logs" / "opencode-home"
    env = mod.opencode_env(home, "m")
    assert env["XDG_CONFIG_HOME"] == str(home / "config")
    assert env["XDG_DATA_HOME"] == str(home / "data")
    written = json.loads((home / "config" / "opencode" / "opencode.json").read_text())
    assert written["provider"]["ollama"]["options"]["baseURL"] == "http://localhost:11434/v1"


def test_events_become_the_common_result() -> None:
    mod = _load()
    payload, error = mod.parse_opencode("not json\n" + _jsonl(_EVENTS))
    assert error == ""
    assert payload == {
        "iterations": 3,
        "tool_calls": 2,
        "permission_denied": 1,
        "usage": {"input_tokens": 6969 + 7090 + 7200, "output_tokens": 112 + 8 + 42 + 30},
        "stop_reason": "stop",
        "session_id": "",
    }


def test_no_step_is_an_error() -> None:
    mod = _load()
    payload, error = mod.parse_opencode("")
    assert payload["iterations"] == 0 and payload["stop_reason"] == "error"
    assert "no LLM response" in error
    _, error = mod.parse_opencode(_jsonl([{"type": "error", "error": {"name": "APIError"}}]))
    assert "APIError" in error


def test_run_opencode_calls_the_cli_without_touching_the_workdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load()
    seen = tmp_path / "seen.txt"
    events = tmp_path / "events.jsonl"
    events.write_text(_jsonl(_EVENTS), encoding="utf-8")
    fake = tmp_path / "opencode"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import os, pathlib, sys\n"
        f"pathlib.Path({str(seen)!r}).write_text('\\n'.join(sys.argv[1:] + [\n"
        "    'XDG_CONFIG_HOME=' + os.environ['XDG_CONFIG_HOME'], 'PWD=' + os.environ['PWD']]))\n"
        f"sys.stdout.write(pathlib.Path({str(events)!r}).read_text())\n",
        encoding="utf-8",
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(mod, "OPENCODE_BIN", str(fake))
    monkeypatch.setattr(mod, "ollama_env", lambda: {"OLLAMA_HOST": "http://localhost:11434"})
    monkeypatch.setenv("PWD", str(tmp_path))
    workdir = tmp_path / "work"
    workdir.mkdir()
    log = tmp_path / "logs" / "task-1-1.log"
    payload, error, _ = mod.run_agent("opencode", "Do it.", workdir, "gpt-oss:120b", 30, log)
    assert error == "" and payload["iterations"] == 3
    args = seen.read_text().splitlines()
    assert args[:4] == ["run", "--standalone", "--auto", "--format"]
    assert "ollama/gpt-oss:120b" in args and args[-3] == "Do it."
    assert args[-2] == f"XDG_CONFIG_HOME={tmp_path / 'logs' / 'opencode-home' / 'config'}"
    assert args[-1] == f"PWD={workdir}"
    assert list(workdir.iterdir()) == []


def test_a_failed_run_keeps_the_stderr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load()
    fake = tmp_path / "opencode"
    fake.write_text("#!/bin/sh\necho 'boom' >&2\nexit 3\n", encoding="utf-8")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(mod, "OPENCODE_BIN", str(fake))
    monkeypatch.setattr(mod, "ollama_env", lambda: {})
    workdir = tmp_path / "work"
    workdir.mkdir()
    _, error, _ = mod.run_opencode("x", workdir, "m", 30, tmp_path / "logs" / "a.log")
    assert error == "exit 3: boom"


def test_writes_outside_the_workdir_stop_the_run(tmp_path: Path) -> None:
    mod = _load()
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / ".gitignore").write_text(".satsuma/\n")
    (repo / "mine.txt").write_text("the user's own change\n")
    before = mod.outside_changes(repo)
    (repo / ".satsuma" / "work").mkdir(parents=True)
    (repo / ".satsuma" / "work" / "a.js").write_text("x")
    mod.check_no_escape(before, mod.outside_changes(repo), tmp_path / "a.log")
    (repo / "design.js").write_text("x")
    with pytest.raises(SystemExit, match="design.js"):
        mod.check_no_escape(before, mod.outside_changes(repo), tmp_path / "a.log")
    assert os.path.isdir(repo)


def test_config_carries_the_reasoning_effort_only_when_given() -> None:
    mod = _load()
    env = {"OLLAMA_HOST": "http://h:11434"}
    plain = mod.opencode_config("m", env)["provider"]["ollama"]["models"]["m"]
    high = mod.opencode_config("m", env, "high")["provider"]["ollama"]["models"]["m"]
    assert plain == {"name": "m"}
    assert high == {"name": "m", "options": {"reasoningEffort": "high"}}


def test_copied_settings_carry_the_reasoning_effort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load()
    repo = tmp_path / "repo"
    (repo / ".satsuma").mkdir(parents=True)
    (repo / ".satsuma" / "settings.local.json").write_text('{"provider": {"default_model": "m"}}')
    monkeypatch.setattr(mod, "REPO", repo)
    work = tmp_path / "w"
    work.mkdir()
    mod.copy_settings(work, "high")
    local = json.loads((work / ".satsuma" / "settings.local.json").read_text())
    assert local["provider"] == {"default_model": "m", "sampling": {"reasoning_effort": "high"}}
    other = tmp_path / "w2"
    other.mkdir()
    mod.copy_settings(other)
    assert (
        "sampling"
        not in json.loads((other / ".satsuma" / "settings.local.json").read_text())["provider"]
    )


def test_replies_are_counted_from_the_exported_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``run --format json`` が最後の ``step_finish`` を出さなくても、書き出した会話から数える。"""
    mod = _load()
    sid = "ses_1"
    events = [
        {"type": "step_start", "sessionID": sid, "part": {}},
        {
            "type": "step_finish",
            "sessionID": sid,
            "part": {"reason": "tool-calls", "tokens": {"input": 100, "output": 5}},
        },
        {"type": "step_start", "sessionID": sid, "part": {}},
        {"type": "text", "sessionID": sid, "part": {"text": "done"}},
    ]
    stream = tmp_path / "events.jsonl"
    stream.write_text(_jsonl(events), encoding="utf-8")
    fake = tmp_path / "opencode"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, sys, time\n"
        "if sys.argv[1:3] == ['session', 'export']:\n"
        "    now = time.time() * 1000\n"
        "    old = {'type': 'assistant', 'time': {'created': 1},\n"
        "           'tokens': {'input': 9, 'output': 9}}\n"
        "    def reply(i, o, r, finish):\n"
        "        return {'type': 'assistant', 'time': {'created': now},\n"
        "                'tokens': {'input': i, 'output': o, 'reasoning': r}, 'finish': finish}\n"
        "    msgs = [old, {'type': 'user'},\n"
        "            reply(100, 5, 0, 'tool-calls'), reply(120, 3, 2, 'stop')]\n"
        "    print(json.dumps({'info': {}, 'messages': msgs}))\n"
        "else:\n"
        f"    sys.stdout.write(pathlib.Path({str(stream)!r}).read_text())\n",
        encoding="utf-8",
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(mod, "OPENCODE_BIN", str(fake))
    monkeypatch.setattr(mod, "ollama_env", lambda: {"OLLAMA_HOST": "http://localhost:11434"})
    workdir = tmp_path / "work"
    workdir.mkdir()
    log = tmp_path / "logs" / "a.log"
    payload, error, _ = mod.run_opencode("x", workdir, "m", 30, log)
    assert error == ""
    assert payload["iterations"] == 2
    assert payload["usage"] == {"input_tokens": 220, "output_tokens": 10}
    assert payload["stop_reason"] == "stop" and payload["session_id"] == sid
    assert log.with_suffix(".export.json").is_file()

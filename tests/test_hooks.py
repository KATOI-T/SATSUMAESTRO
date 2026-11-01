"""HookRunner のテスト。"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from pydantic import BaseModel

from satsumaestro.config.models import HookConfig
from satsumaestro.engine.events import ToolUse
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.tools import default_registry
from satsumaestro.tools.base import ToolResult

_REGISTRY = default_registry()


def _hook_cmd(tmp_path: Path, name: str, response: dict | None, *, exit_code: int = 0) -> str:
    """stdin を読み、固定 JSON を返す python フックスクリプトを作りコマンド文字列を返す。"""
    body = "import json,sys\njson.load(sys.stdin)\n"
    if response is not None:
        body += f"print(json.dumps({response!r}))\n"
    if exit_code:
        body += f"sys.exit({exit_code})\n"
    path = tmp_path / f"{name}.py"
    path.write_text(body, encoding="utf-8")
    return f"{sys.executable} {path}"


def _call() -> ToolUse:
    return ToolUse(id="1", name="bash", input={"command": "rm -rf /"})


def _inp() -> BaseModel:
    """_call() に対応する検証済み入力（pre_tool_use は検証済み inp を基準にする）。"""
    return _REGISTRY.validate_input(_call())


async def test_pre_hook_deny(tmp_path: Path) -> None:
    cmd = _hook_cmd(tmp_path, "deny", {"deny": True, "reason": "blocked"})
    runner = HookRunner(HookConfig(pre_tool_use=[cmd]), tmp_path)
    res = await runner.pre_tool_use(_call(), _inp())
    assert res.deny is True
    assert res.reason == "blocked"


async def test_pre_hook_override(tmp_path: Path) -> None:
    cmd = _hook_cmd(tmp_path, "ovr", {"override": {"decision": "allow", "reason": "trusted"}})
    runner = HookRunner(HookConfig(pre_tool_use=[cmd]), tmp_path)
    res = await runner.pre_tool_use(_call(), _inp())
    assert res.override is not None
    assert res.override.decision == "allow"


async def test_pre_hook_input_rewrite(tmp_path: Path) -> None:
    cmd = _hook_cmd(tmp_path, "rw", {"input": {"command": "echo safe"}})
    runner = HookRunner(HookConfig(pre_tool_use=[cmd]), tmp_path)
    res = await runner.pre_tool_use(_call(), _inp())
    assert res.input == {"command": "echo safe"}


async def test_neutral_when_no_hooks(tmp_path: Path) -> None:
    runner = HookRunner(HookConfig(), tmp_path)
    res = await runner.pre_tool_use(_call(), _inp())
    assert res.deny is False and res.override is None and res.input is None


async def test_post_hook_feedback(tmp_path: Path) -> None:
    cmd = _hook_cmd(tmp_path, "fb", {"feedback": "lint ok"})
    runner = HookRunner(HookConfig(post_tool_use=[cmd]), tmp_path)
    out = await runner.post_tool_use(_call(), ToolResult(output="done"))
    assert out.output == "done\n\nNotes from hooks:\n- lint ok"


async def test_failing_hook_fails_open(tmp_path: Path) -> None:
    cmd = _hook_cmd(tmp_path, "boom", None, exit_code=3)
    runner = HookRunner(HookConfig(pre_tool_use=[cmd]), tmp_path)
    res = await runner.pre_tool_use(_call(), _inp())
    assert res.deny is False


async def test_pre_hook_chain_deny_short_circuits(tmp_path: Path) -> None:
    ok = _hook_cmd(tmp_path, "ok", {"input": {"command": "echo hi"}})
    deny = _hook_cmd(tmp_path, "deny2", {"deny": True, "reason": "second says no"})
    runner = HookRunner(HookConfig(pre_tool_use=[ok, deny]), tmp_path)
    res = await runner.pre_tool_use(_call(), _inp())
    assert res.deny is True and res.reason == "second says no"


async def test_pre_hook_receives_normalized_input(tmp_path: Path) -> None:
    echo = _hook_cmd(tmp_path, "echo", None)
    echo_path = tmp_path / "echo.py"
    echo_path.write_text(
        "import json,sys\nd=json.load(sys.stdin)\nprint(json.dumps({'input': d['input']}))\n",
        encoding="utf-8",
    )
    call = ToolUse(id="1", name="read_file", input={"path": "x"})
    inp = _REGISTRY.validate_input(call)
    runner = HookRunner(HookConfig(pre_tool_use=[echo]), tmp_path)
    res = await runner.pre_tool_use(call, inp)
    assert res.input == inp.model_dump(mode="json")
    assert res.input is not None and "offset" in res.input


async def test_timeout_kills_process(tmp_path: Path) -> None:
    marker = tmp_path / "marker.txt"
    script = tmp_path / "slow.py"
    writer = (
        f"import time,pathlib; time.sleep(0.6); pathlib.Path({str(marker)!r}).write_text('done')"
    )
    script.write_text(
        "import sys,json,subprocess,time\n"
        "json.load(sys.stdin)\n"
        f"subprocess.Popen([sys.executable, '-c', {writer!r}])\n"
        "time.sleep(5)\n",
        encoding="utf-8",
    )
    runner = HookRunner(
        HookConfig(pre_tool_use=[f"{sys.executable} {script}"]), tmp_path, timeout_s=0.1
    )
    res = await runner.pre_tool_use(_call(), _inp())
    assert res.deny is False
    await asyncio.sleep(1.2)
    assert not marker.exists()


def test_json_protocol_smoke() -> None:
    payload = {"event": "pre_tool_use", "tool": "bash", "input": {"command": "ls"}}
    assert json.loads(json.dumps(payload))["tool"] == "bash"

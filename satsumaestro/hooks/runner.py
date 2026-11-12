"""pre/post ツールフックランナー。"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from satsumaestro.config.models import HookConfig, SandboxConfig
from satsumaestro.security.policy import PermissionOutcome
from satsumaestro.security.sandbox import child_environment
from satsumaestro.tools.base import ToolResult
from satsumaestro.util.proc import terminate

if TYPE_CHECKING:
    from satsumaestro.engine.events import ToolUse

_log = logging.getLogger(__name__)


class HookResult(BaseModel):
    """pre_tool_use の結果。"""

    deny: bool = False
    reason: str = ""
    override: PermissionOutcome | None = None
    input: dict[str, Any] | None = None


class HookRunner:
    """コマンド実行型の pre/post フックランナー。"""

    def __init__(
        self,
        config: HookConfig,
        cwd: Path,
        timeout_s: float = 30.0,
        *,
        sandbox: SandboxConfig | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._config = config
        self._cwd = cwd
        self._timeout_s = timeout_s
        self._sandbox = sandbox if sandbox is not None else SandboxConfig()
        self._env = env

    async def pre_tool_use(self, call: ToolUse, inp: BaseModel) -> HookResult:
        """設定された pre フックを順に実行し、集約した :class:`HookResult` を返す。"""
        result = HookResult()
        current_input: dict[str, Any] = inp.model_dump(mode="json")
        for cmd in self._config.pre_tool_use:
            payload = {
                "event": "pre_tool_use",
                "tool": call.name,
                "input": current_input,
            }
            out = await self._run(cmd, payload)
            if out is None:
                continue
            if out.get("deny"):
                return HookResult(deny=True, reason=str(out.get("reason", "denied by hook")))
            if isinstance(out.get("override"), dict):
                result.override = _parse_override(out["override"])
            if isinstance(out.get("input"), dict):
                current_input = out["input"]
                result.input = current_input
        return result

    async def post_tool_use(self, call: ToolUse, res: ToolResult) -> ToolResult:
        """設定された post フックを順に実行し、出力・is_error を反映して返す。"""
        return await self._run_post("post_tool_use", self._config.post_tool_use, call, res)

    async def post_tool_use_failure(self, call: ToolUse, res: ToolResult) -> ToolResult:
        """``res.is_error`` が真のときだけ失敗専用フックを実行する。"""
        if not res.is_error:
            return res
        return await self._run_post(
            "post_tool_use_failure", self._config.post_tool_use_failure, call, res
        )

    async def _run_post(
        self, event: str, cmds: list[str], call: ToolUse, res: ToolResult
    ) -> ToolResult:
        """post 系フック共通の実行ループ（post_tool_use / post_tool_use_failure）。"""
        output, is_error = res.output, res.is_error
        notes: list[str] = []
        for cmd in cmds:
            payload = {
                "event": event,
                "tool": call.name,
                "output": output,
                "is_error": is_error,
            }
            out = await self._run(cmd, payload)
            if out is None:
                continue
            if isinstance(out.get("output"), str):
                output = out["output"]
            if isinstance(out.get("is_error"), bool):
                is_error = out["is_error"]
            if isinstance(out.get("feedback"), str) and out["feedback"]:
                notes.append(out["feedback"])
        return ToolResult(
            output=append_hook_notes(output, notes, is_error),
            is_error=is_error,
            usage=res.usage,
        )

    async def _run(self, cmd: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        """1 コマンドを実行し stdout の JSON を返す。"""
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._cwd),
                env=child_environment(self._sandbox, self._env),
                start_new_session=True,
            )
        except OSError as e:
            _log.warning("hook failed to spawn (%s): %s", cmd, e)
            return None

        stdin = json.dumps(payload, ensure_ascii=False).encode()
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(stdin), timeout=self._timeout_s
            )
        except TimeoutError:
            _log.warning("hook timed out: %s", cmd)
            await terminate(proc)
            return None

        if proc.returncode != 0:
            _log.warning(
                "hook exited %s: %s (%s)", proc.returncode, cmd, stderr.decode(errors="replace")
            )
            return None
        body = stdout.decode(errors="replace").strip()
        if not body:
            return None
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            _log.warning("hook produced non-JSON output: %s", cmd)
            return None
        return parsed if isinstance(parsed, dict) else None


def append_hook_notes(output: str, notes: list[str], failed: bool) -> str:
    """フックが返した指摘を、ツールの出力の後ろに箇条書きで足す。"""
    if not notes:
        return output
    heading = "Notes from hooks (the call failed):" if failed else "Notes from hooks:"
    items = ["- " + note.strip().replace("\n", "\n  ") for note in notes]
    section = "\n".join([heading, *items])
    return f"{output}\n\n{section}" if output.strip() else section


def _parse_override(raw: dict[str, Any]) -> PermissionOutcome | None:
    """``{"decision","reason"}`` を :class:`PermissionOutcome` へ。"""
    decision = raw.get("decision")
    if decision not in ("allow", "deny", "ask"):
        return None
    return PermissionOutcome(decision=decision, reason=str(raw.get("reason", "hook override")))

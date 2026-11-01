"""ツール実行の表示内容（動作ラベル・対象・出すタイミング・結果の付記）。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from satsumaestro.engine.events import ToolUse
from satsumaestro.i18n.text import ui
from satsumaestro.security.jail import (
    OUTSIDE_WORKSPACE,
    PATH_ESCAPE_PREFIX,
    STATE_DIR_HIDDEN,
    STATE_DIR_UNDELETABLE,
)
from satsumaestro.session.models import ToolResultBlock


def _short_reasons() -> tuple[tuple[str, str], ...]:
    """決まった頭で始まる失敗の、画面に出す短い文。"""
    return (
        (
            STATE_DIR_HIDDEN,
            ui("cli.tool_display.the_agent_s_state_folder"),
        ),
        (
            STATE_DIR_UNDELETABLE,
            ui("cli.tool_display.files_in_the_agent_s"),
        ),
        (
            f"{PATH_ESCAPE_PREFIX}{OUTSIDE_WORKSPACE}",
            ui("cli.tool_display.outside_the_working_directory"),
        ),
    )


_DETAIL_LIMIT = 80


@dataclass(frozen=True)
class ToolDisplay:
    """1 回のツール呼び出しの表示内容。"""

    label: str
    items: tuple[str, ...]
    at_start: bool
    active: str = ""

    def status(self) -> str:
        """実行中の状態行の文。"""
        return self.active or ui("cli.tool_display.active_fallback", label=self.label)


_Entry = tuple[str, str, str]


def _on_result() -> dict[str, _Entry]:
    """参照系（結果が返ったときに出す）。"""
    return {
        "read_file": (ui("cli.tool_display.read"), ui("cli.tool_display.reading"), "path"),
        "glob_search": (
            ui("cli.tool_display.find_files"),
            ui("cli.tool_display.finding_files"),
            "pattern",
        ),
        "list_directory": (ui("cli.tool_display.list"), ui("cli.tool_display.listing"), "path"),
        "web_fetch": (
            ui("cli.tool_display.web_fetch"),
            ui("cli.tool_display.fetching"),
            "url",
        ),
        "web_search": (
            ui("cli.tool_display.web_search"),
            ui("cli.tool_display.searching_the_web"),
            "query",
        ),
        "rag_search": (
            ui("cli.tool_display.index_search"),
            ui("cli.tool_display.searching_the_index"),
            "query",
        ),
        "tool_search": (
            ui("cli.tool_display.tool_search"),
            ui("cli.tool_display.searching_tools"),
            "query",
        ),
        "bash_output": (
            ui("cli.tool_display.background_output"),
            ui("cli.tool_display.checking_background_output"),
            "task_id",
        ),
        "skill": (
            ui("cli.tool_display.load_skill"),
            ui("cli.tool_display.loading_skill"),
            "skill",
        ),
    }


def _on_start() -> dict[str, _Entry]:
    """開始時に出す（変更系・コマンド・サブエージェント・オーケストレーション・ワーカー）。"""
    return {
        "write_file": (ui("cli.tool_display.write"), ui("cli.tool_display.writing"), "path"),
        "edit_file": (ui("cli.tool_display.edit"), ui("cli.tool_display.editing"), "path"),
        "append_file": (ui("cli.tool_display.append"), ui("cli.tool_display.appending"), "path"),
        "delete_path": (ui("cli.tool_display.delete"), ui("cli.tool_display.deleting"), "path"),
        "notebook_edit": (
            ui("cli.tool_display.notebook_edit"),
            ui("cli.tool_display.editing_notebook"),
            "notebook_path",
        ),
        "bash": (
            ui("cli.tool_display.run_command"),
            ui("cli.tool_display.running_command"),
            "command",
        ),
        "agent": (
            ui("cli.tool_display.subagent"),
            ui("cli.tool_display.running_subagent"),
            "description",
        ),
        "enter_plan_mode": (
            ui("cli.tool_display.enter_plan_mode"),
            ui("cli.tool_display.entering_plan_mode"),
            "",
        ),
        "exit_plan_mode": (
            ui("cli.tool_display.exit_plan_mode"),
            ui("cli.tool_display.exiting_plan_mode"),
            "",
        ),
        "task_create": (
            ui("cli.tool_display.create_task"),
            ui("cli.tool_display.creating_task"),
            "description",
        ),
        "task_get": (
            ui("cli.tool_display.check_task"),
            ui("cli.tool_display.checking_task"),
            "task_id",
        ),
        "task_list": (
            ui("cli.tool_display.list_tasks"),
            ui("cli.tool_display.listing_tasks"),
            "status",
        ),
        "task_stop": (
            ui("cli.tool_display.stop_task"),
            ui("cli.tool_display.stopping_task"),
            "task_id",
        ),
        "task_update": (
            ui("cli.tool_display.update_task"),
            ui("cli.tool_display.updating_task"),
            "task_id",
        ),
        "task_output": (
            ui("cli.tool_display.task_output"),
            ui("cli.tool_display.checking_task_output"),
            "task_id",
        ),
        "team_create": (
            ui("cli.tool_display.create_team"),
            ui("cli.tool_display.creating_team"),
            "name",
        ),
        "team_delete": (
            ui("cli.tool_display.delete_team"),
            ui("cli.tool_display.deleting_team"),
            "team_id",
        ),
        "cron_create": (
            ui("cli.tool_display.add_schedule"),
            ui("cli.tool_display.adding_schedule"),
            "description",
        ),
        "cron_delete": (
            ui("cli.tool_display.delete_schedule"),
            ui("cli.tool_display.deleting_schedule"),
            "cron_id",
        ),
        "cron_list": (
            ui("cli.tool_display.list_schedules"),
            ui("cli.tool_display.listing_schedules"),
            "",
        ),
        "worker_create": (
            ui("cli.tool_display.create_worker"),
            ui("cli.tool_display.creating_worker"),
            "cwd",
        ),
        "worker_get": (
            ui("cli.tool_display.check_worker"),
            ui("cli.tool_display.checking_worker"),
            "worker_id",
        ),
        "worker_await_ready": (
            ui("cli.tool_display.wait_for_worker"),
            ui("cli.tool_display.waiting_for_worker"),
            "worker_id",
        ),
        "worker_resolve_trust": (
            ui("cli.tool_display.resolve_worker_trust"),
            ui("cli.tool_display.resolving_worker_trust"),
            "worker_id",
        ),
        "worker_send_prompt": (
            ui("cli.tool_display.send_to_worker"),
            ui("cli.tool_display.sending_to_worker"),
            "worker_id",
        ),
        "worker_restart": (
            ui("cli.tool_display.restart_worker"),
            ui("cli.tool_display.restarting_worker"),
            "worker_id",
        ),
        "worker_terminate": (
            ui("cli.tool_display.terminate_worker"),
            ui("cli.tool_display.terminating_worker"),
            "worker_id",
        ),
    }


_TODO_MARKS = {"pending": "☐", "in_progress": "▶", "completed": "☑"}

_MCP_PREFIX = "mcp__"


def known_tool_names() -> frozenset[str]:
    """個別の表示を定義済みのツール名（テストで登録簿との突き合わせに使う）。"""
    return frozenset({*_on_result(), *_on_start(), "grep_search", "lsp", "todo_write"})


def _clip(text: str) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= _DETAIL_LIMIT else text[:_DETAIL_LIMIT] + "…"


def _field(inp: dict[str, Any], key: str) -> str:
    value = inp.get(key)
    return _clip(value) if value not in (None, "") else ""


def describe(call: ToolUse) -> ToolDisplay:
    """ツール呼び出しの見出し・対象・出すタイミングを返す。"""
    name, inp = call.name, call.input
    on_result = _on_result()
    if name in on_result:
        label, active, key = on_result[name]
        return ToolDisplay(label, _items(_field(inp, key)), at_start=False, active=active)
    on_start = _on_start()
    if name in on_start:
        label, active, key = on_start[name]
        detail = _field(inp, key) if key else ""
        if name == "task_create" and not detail:
            detail = _field(inp, "prompt")
        if name == "cron_create" and not detail:
            detail = _field(inp, "schedule")
        return ToolDisplay(label, _items(detail), at_start=True, active=active)
    if name == "grep_search":
        pattern = _field(inp, "pattern")
        path = _field(inp, "path")
        detail = f'"{pattern}"' + (f" in {path}" if path else "")
        return ToolDisplay(
            ui("cli.tool_display.search"),
            (detail,),
            at_start=False,
            active=ui("cli.tool_display.searching"),
        )
    if name == "lsp":
        action, path = _field(inp, "action"), _field(inp, "path")
        line = inp.get("line")
        where = f"{path}:{line}" if path and line is not None else path or _field(inp, "query")
        return ToolDisplay(
            ui("cli.tool_display.code_analysis"),
            _items(" ".join(p for p in (action, where) if p)),
            at_start=False,
            active=ui("cli.tool_display.analyzing_code"),
        )
    if name == "todo_write":
        todos = inp.get("todos") or []
        items = tuple(
            f"{_TODO_MARKS.get(str(t.get('status')), '☐')} {_clip(t.get('content', ''))}"
            for t in todos
            if isinstance(t, dict)
        )
        return ToolDisplay(
            ui("cli.tool_display.update_todo"),
            items,
            at_start=True,
            active=ui("cli.tool_display.updating_todo"),
        )
    if name.startswith(_MCP_PREFIX):
        server, _, tool = name[len(_MCP_PREFIX) :].partition("__")
        return ToolDisplay(ui("cli.tool_display.mcp", server=server), _items(tool), at_start=False)
    lines = name.strip().splitlines() or [""]
    return ToolDisplay(lines[0] + (" …" if len(lines) > 1 else ""), (), at_start=False)


def _items(detail: str) -> tuple[str, ...]:
    return (detail,) if detail else ()


_CONTENT_MATCH = re.compile(r"^.+?:\d+:", re.MULTILINE)


def result_note(call: ToolUse, block: ToolResultBlock) -> str | None:
    """結果に添える短い付記（件数など）。"""
    if block.is_error:
        return None
    output = block.output
    truncated = "... (truncated" in output
    if call.name == "glob_search":
        n = len([ln for ln in output.splitlines() if ln.strip() and not ln.startswith("...")])
        return _count(n, truncated) if n else _no_match()
    if call.name == "list_directory":
        n = len(
            [ln for ln in output.splitlines() if ln.strip() and not ln.startswith(("(", "..."))]
        )
        return _count(n, truncated, entries=True) if n else ui("cli.tool_display.empty_directory")
    if call.name == "grep_search":
        mode = call.input.get("output_mode") or "content"
        lines = [ln for ln in output.splitlines() if ln.strip() and not ln.startswith("...")]
        if mode == "files_with_matches":
            n = len(lines)
            if not n:
                return _no_match()
            if truncated:
                return ui("cli.tool_display.files_or_more", n=n)
            return ui("cli.tool_display.files", n=n)
        if mode == "count":
            n = sum(int(m.group(1)) for ln in lines if (m := re.search(r":(\d+)$", ln)))
        else:
            n = len(_CONTENT_MATCH.findall("\n".join(ln for ln in lines if ln != "--")))
        return _count(n, truncated) if n else _no_match()
    return None


def _count(n: int, truncated: bool, *, entries: bool = False) -> str:
    """件数の付記（``12 件`` / ``12 matches``。打ち切られていれば「以上」/ ``+``）。"""
    if entries and truncated:
        return ui("cli.tool_display.entries_or_more", n=n)
    if entries:
        return ui("cli.tool_display.entries", n=n)
    if truncated:
        return ui("cli.tool_display.matches_or_more", n=n)
    return ui("cli.tool_display.matches", n=n)


def _no_match() -> str:
    return ui("cli.tool_display.no_matches")


def _short_reason(output: str) -> str | None:
    head = output.strip()
    for prefix, short in _short_reasons():
        if head.startswith(prefix):
            return short
    return None


def error_reason(block: ToolResultBlock) -> str:
    """失敗理由の 1 行目（表示用に切り詰め）。"""
    short = _short_reason(block.output)
    if short is not None:
        return short
    first = block.output.strip().splitlines()[0] if block.output.strip() else "(no output)"
    return _clip(first)


_ERROR_DETAIL_LINES = 10
_ERROR_DETAIL_CHARS = 400
_ERROR_NOISE_PREFIXES = ("For further information visit",)
ERROR_TRUNCATED = "… (truncated)"


def error_detail(block: ToolResultBlock) -> list[str]:
    """失敗理由の 2 行目以降（表示用）。"""
    if _short_reason(block.output) is not None:
        return []
    first, *rest = block.output.strip().splitlines() or [""]
    lines = [first, *rest] if len(" ".join(first.split())) > _DETAIL_LIMIT else rest
    kept: list[str] = []
    budget = _ERROR_DETAIL_CHARS
    truncated = False
    for raw in lines:
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith(_ERROR_NOISE_PREFIXES):
            continue
        if len(kept) >= _ERROR_DETAIL_LINES or budget <= 0:
            truncated = True
            break
        if len(line) > budget:
            kept.append(line[:budget])
            truncated = True
            break
        kept.append(line)
        budget -= len(line)
    if truncated:
        kept.append(ERROR_TRUNCATED)
    return kept

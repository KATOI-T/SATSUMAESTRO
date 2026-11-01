"""状態トリガ式 system-reminder 生成。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum

from satsumaestro.engine.repetition import RepetitionVerdict
from satsumaestro.security.deletion_guard import PROHIBITED_DELETION_PREFIX
from satsumaestro.security.output_guard import UNREQUESTED_DELETION_PREFIX
from satsumaestro.security.policy import BLOCKED_TOOL_SUFFIX
from satsumaestro.security.state_write_guard import PROHIBITED_STATE_WRITE_PREFIX
from satsumaestro.session.models import Session, ToolResultBlock, ToolUseBlock
from satsumaestro.util import chunking
from satsumaestro.util.lang import language_name

FAILURE_STREAK_THRESHOLD = 2
TODO_STALE_ITERATIONS = 10

PLAN_MODE_DENY_PREFIX = "plan mode is active:"

_POLICY_DENY_PREFIXES = ("deny rule:", "secret file access denied")
_USER_DENY_PREFIX = "denied by user"


def _deny_kind(result: ToolResultBlock) -> str | None:
    """拒否の種類（``policy`` / ``prohibited`` / ``user``）。"""
    if not result.is_error:
        return None
    first = result.output.split("\n", 1)[0]
    if first.startswith(
        (PROHIBITED_DELETION_PREFIX, UNREQUESTED_DELETION_PREFIX, PROHIBITED_STATE_WRITE_PREFIX)
    ):
        return "prohibited"
    if first.startswith(_POLICY_DENY_PREFIXES) or first.endswith(BLOCKED_TOOL_SUFFIX):
        return "policy"
    if first.startswith(_USER_DENY_PREFIX):
        return "user"
    return None


_EDIT_TOOL = "edit_file"
EDIT_FILE_STREAK_HINT = (
    "For edit_file: do not guess `old` again. Use the line numbers in the last error, "
    "read_file that range, and copy those lines verbatim (including indentation) into `old`."
)

_TODO_TOOL = "todo_write"


class ReminderTrigger(StrEnum):
    POST_COMPACTION = "post-compaction"
    TOOL_FAILURE_STREAK = "failure-streak"
    PLAN_MODE_ACTIVE = "plan-mode"
    POLICY_DENY = "policy-deny"
    POLICY_DENY_REPEATED = "policy-deny-repeated"
    USER_DENY = "user-deny"
    TODO_STALE = "todo-stale"
    EMPTY_FINAL = "empty-final"
    LANGUAGE_MISMATCH = "language-mismatch"
    TIMEOUT_RECOVERY = "timeout-recovery"
    TOOL_CALL_REWRITE = "tool-call-rewrite"
    PLAN_EXIT_NUDGE = "plan-exit-nudge"
    PLAN_EXIT_NO_PLAN = "plan-exit-no-plan"
    REPETITION_ABORT = "repetition-abort"
    BACKGROUND_TASK_DONE = "background-task-done"
    ARGS_ONLY_REPLY = "args-only-reply"
    NO_CHANGE_REPORT = "no-change-report"


_TEMPLATES: dict[ReminderTrigger, str] = {
    ReminderTrigger.POST_COMPACTION: (
        "Earlier conversation history was just compacted into a summary. Details that are "
        "not in the summary are no longer visible; re-read the relevant files instead of "
        "relying on memory."
    ),
    ReminderTrigger.TOOL_FAILURE_STREAK: (
        "The tool `{tool}` has failed {count} times in a row. Stop retrying with the same "
        "arguments; read the error message and fix the cause first."
    ),
    ReminderTrigger.PLAN_MODE_ACTIVE: (
        "Plan mode is still active: tools beyond read-only are denied (except writing the "
        "plan file). Finish investigating with read-only tools, write the plan file, then "
        "call exit_plan_mode to request the user's approval."
    ),
    ReminderTrigger.POLICY_DENY: (
        "The call to `{tool}` was blocked by a permission rule; it did not fail. Reissuing the "
        "same call will be blocked again every time, so do not retry it. If the operation is "
        "still needed, use a more suitable tool (for deleting files or directories, use "
        "delete_path) or explain to the user what you need and why."
    ),
    ReminderTrigger.POLICY_DENY_REPEATED: (
        "You reissued a call to `{tool}` that had already been blocked, and it was blocked "
        "again. Stop retrying it and its variations (other flags, other path spellings). Tell "
        "the user what you were trying to do and ask how they want to proceed."
    ),
    ReminderTrigger.USER_DENY: (
        "The user declined the call to `{tool}` and the turn was stopped there. Do not retry "
        "it or work around it unless the user's next message asks for it."
    ),
    ReminderTrigger.TODO_STALE: (
        "The TODO list has not been updated for many steps. Bring it up to date with "
        "todo_write: mark finished items completed and drop items that no longer apply."
    ),
    ReminderTrigger.EMPTY_FINAL: (
        "Your previous message ended with an empty body: internal reasoning is not shown, "
        "so nothing was delivered. If the request needs files to be created or changed, do it "
        "now with tool calls; otherwise state your final answer as plain message text."
    ),
    ReminderTrigger.ARGS_ONLY_REPLY: (
        "Your previous message was only a JSON object ({keys}) that looks like the arguments "
        "of the tool {tool}. It was sent as message text, so nothing ran. Call {tool} as a "
        "tool call now instead of writing its arguments."
    ),
    ReminderTrigger.NO_CHANGE_REPORT: (
        "Your previous message reports work on {paths}, but this turn changed nothing: no file "
        "was created, edited or deleted{missing}. If the work is still needed, do it now with "
        "tool calls (write_file / edit_file / append_file). If no change is needed, say so and "
        "explain why."
    ),
    ReminderTrigger.LANGUAGE_MISMATCH: (
        "Your previous message was not written in {language}. Rewrite the same answer in "
        "{language} now: same content, same structure, nothing added or removed. Keep code "
        "blocks, file paths, identifiers and command output byte-identical. Do not call any "
        "tools; reply with the rewritten answer only."
    ),
    ReminderTrigger.TIMEOUT_RECOVERY: (
        "Your previous response was cancelled: nothing arrived for {seconds} seconds and the "
        "request timed out. Two causes are common: one tool call tried to carry too much text at "
        "once (for example a `write_file` `content` of several hundred lines), or the server "
        "spent that time re-reading a long conversation before it could start answering. If you "
        "were about to write a lot, do not repeat that call unchanged: produce the same result "
        "in pieces — `write_file` with the first part (at most {limit}), then `append_file` for "
        "each following part, one piece per call. Otherwise, simply continue with the next "
        "step; the conversation has been read by now, so the next request is usually faster. "
        "Call the next tool now — do not reply with a description of what you are about to do. "
        "Work that already succeeded is still done; do not redo it."
    ),
    ReminderTrigger.TOOL_CALL_REWRITE: (
        "Your previous tool call was rejected by the server again, even after the hint to fix "
        "it, because its arguments were not valid JSON — for example a raw line break or an "
        'unescaped `"` inside a string, or an invalid backslash escape. Do not resend the same '
        "call unchanged. Write the call again with valid JSON: inside strings, write line breaks "
        'as `\\n` and double quotes as `\\"`. If the content is long, write it in pieces: '
        "`write_file` with the first part, then `append_file` for each following part. Work "
        "that already succeeded is still done; do not redo it."
    ),
    ReminderTrigger.PLAN_EXIT_NUDGE: (
        "Plan mode is still active: your previous message ended without calling "
        "exit_plan_mode, so the user was never asked to approve the plan and nothing in it "
        "can be carried out yet. Asking for approval in the message text does not start the "
        "approval, and neither does writing the name exit_plan_mode as text — only the tool "
        "call does. Call exit_plan_mode now as a tool call, with no other tools. The plan "
        "file `{plan_file}` is shown to the user along with it, so do not repeat the plan in "
        "the message text."
    ),
    ReminderTrigger.PLAN_EXIT_NO_PLAN: (
        "Plan mode is still active: your previous message ended without calling "
        "exit_plan_mode, and no plan has been written yet, so the user was never asked to "
        "approve anything. Write the plan to `{plan_file}` with write_file (its directory "
        "already exists), then call exit_plan_mode as a tool call to ask for approval. "
        "Describing the plan in the message text does not ask for approval."
    ),
    ReminderTrigger.REPETITION_ABORT: (
        "Your previous response was cut off and discarded because it kept repeating itself: "
        'the same {unit} came back {repeats} times (it starts with: "{snippet}") and almost '
        "nothing new was being added. None of it was delivered. Do not write it again. If you "
        "were reasoning about something you are not sure of, stop reasoning and check it with "
        "one tool call (read the file, run the command). Otherwise state your conclusion once, "
        "in a few lines, and stop."
    ),
    ReminderTrigger.BACKGROUND_TASK_DONE: (
        "Background task {task_id} (command: {preview}) has exited with code {code}. "
        'Read its remaining output with bash_output(task_id="{task_id}") if you still '
        "need it."
    ),
}


def background_done_note(task_id: str, code: int | None, preview: str) -> str:
    """バックグラウンド bash の完了通知（発火は runtime、1 回制御は台帳側）。"""
    return _TEMPLATES[ReminderTrigger.BACKGROUND_TASK_DONE].format(
        task_id=task_id, code=code, preview=preview
    )


def empty_final_note() -> str:
    """EMPTY_FINAL の定型文（発火判定は runtime 側）。"""
    return _TEMPLATES[ReminderTrigger.EMPTY_FINAL]


_ARGS_ONLY_REPLY_MULTI = (
    "Your previous message was only a JSON object ({keys}) that looks like the arguments of "
    "one of these tools: {tools}. It was sent as message text, so nothing ran. Call the right "
    "one as a tool call now instead of writing its arguments."
)


def no_change_report_note(paths: tuple[str, ...], missing: tuple[str, ...]) -> str:
    """NO_CHANGE_REPORT の定型文（触れたパスと、無いパスの注記。判定は runtime 側）。"""
    shown = ", ".join(f"`{p}`" for p in paths)
    note = ""
    if missing:
        gone = ", ".join(f"`{p}`" for p in missing)
        note = f" ({gone} {'does' if len(missing) == 1 else 'do'} not exist)"
    return _TEMPLATES[ReminderTrigger.NO_CHANGE_REPORT].format(paths=shown, missing=note)


def args_only_reply_note(keys: tuple[str, ...], tools: tuple[str, ...]) -> str:
    """ARGS_ONLY_REPLY の定型文（候補 1 本なら名指し、複数なら列挙。判定は runtime 側）。"""
    shown = ", ".join(f"`{k}`" for k in keys)
    if len(tools) == 1:
        return _TEMPLATES[ReminderTrigger.ARGS_ONLY_REPLY].format(keys=shown, tool=f"`{tools[0]}`")
    return _ARGS_ONLY_REPLY_MULTI.format(keys=shown, tools=", ".join(f"`{t}`" for t in tools))


_EMPTY_FINAL_IN_PLAN = (
    "Your previous message ended with an empty body: internal reasoning is not shown, so "
    "nothing was done. Continue with tool calls: write the plan to `{plan_file}` with "
    "write_file if you have not yet, then call exit_plan_mode as a tool call."
)
_EMPTY_FINAL_AFTER_APPROVAL = (
    "Your previous message ended with an empty body: internal reasoning is not shown, so "
    "nothing was done. The plan is approved: carry it out now with tool calls (create and "
    "edit files with write_file / append_file / edit_file). Only if every change is already "
    "made, state your final answer as plain message text."
)


def empty_final_plan_note(plan_file: str) -> str:
    """plan 中の EMPTY_FINAL（計画を書いて exit_plan_mode を呼べ）。"""
    return _EMPTY_FINAL_IN_PLAN.format(plan_file=plan_file)


def empty_final_approved_note() -> str:
    """計画の承認直後の EMPTY_FINAL（承認済みの計画をツールで実行せよ）。"""
    return _EMPTY_FINAL_AFTER_APPROVAL


def language_mismatch_note(language: str) -> str:
    """LANGUAGE_MISMATCH の定型文（発火判定は runtime 側）。"""
    return _TEMPLATES[ReminderTrigger.LANGUAGE_MISMATCH].format(language=language_name(language))


def timeout_note(seconds: float, lines: int | None = None) -> str:
    """TIMEOUT_RECOVERY の定型文（発火判定は runtime 側）。"""
    limit = f"~{lines} lines" if lines else chunking.LINES
    return _TEMPLATES[ReminderTrigger.TIMEOUT_RECOVERY].format(
        seconds=f"{seconds:.0f}", limit=limit
    )


def tool_call_rewrite_note() -> str:
    """TOOL_CALL_REWRITE の定型文（発火判定は runtime 側）。"""
    return _TEMPLATES[ReminderTrigger.TOOL_CALL_REWRITE]


def plan_exit_nudge_note(plan_file: str) -> str:
    """PLAN_EXIT_NUDGE の定型文（発火判定は runtime 側）。"""
    return _TEMPLATES[ReminderTrigger.PLAN_EXIT_NUDGE].format(plan_file=plan_file)


def plan_missing_plan_note(plan_file: str) -> str:
    """PLAN_EXIT_NO_PLAN の定型文（同上）。"""
    return _TEMPLATES[ReminderTrigger.PLAN_EXIT_NO_PLAN].format(plan_file=plan_file)


def repetition_note(verdict: RepetitionVerdict) -> str:
    """REPETITION_ABORT の定型文（発火判定は runtime 側）。"""
    return _TEMPLATES[ReminderTrigger.REPETITION_ABORT].format(
        unit=verdict.unit, repeats=verdict.repeats, snippet=verdict.snippet
    )


def language_note(language: str) -> str:
    """他の定型文へ併記する短い言語指示（``language`` が空なら空文字）。"""
    if not language:
        return ""
    return f"Reply in {language_name(language)}."


def format_reminder(notes: list[str]) -> str:
    """発火した定型文を 1 つの ``<system-reminder>`` タグへ整形する。"""
    return "<system-reminder>\n" + "\n".join(notes) + "\n</system-reminder>"


def _todo_list_note(session: Session) -> str:
    """圧縮後の知らせに添える今の TODO 一覧（無ければ空文字）。"""
    if not session.todos:
        return ""
    from satsumaestro.tools.builtins.todo_write import format_todos

    return "\n\nCurrent TODO list:\n" + format_todos(session.todos)


@dataclass
class ReminderState:
    """1 ターン分のトリガ状態（``run_turn`` ごとに 1 個）。"""

    todo_stale_after: int = TODO_STALE_ITERATIONS
    _fired: set[ReminderTrigger] = field(default_factory=set)
    _streaks: dict[str, int] = field(default_factory=dict)
    _streak_fired: set[str] = field(default_factory=set)
    _pending_compaction: bool = False
    _iterations_since_todo: int = 0
    _denials: dict[str, int] = field(default_factory=dict)
    emitted: int = 0

    def note_empty_final_retry(self) -> None:
        """EMPTY_FINAL 再試行の注入を計測へ含める（注入と 1 回制御は runtime が行う）。"""
        self.emitted += 1

    def note_language_mismatch(self) -> None:
        """LANGUAGE_MISMATCH 再生成の注入を計測へ含める（同上）。"""
        self.emitted += 1

    def note_timeout_recovery(self) -> None:
        """TIMEOUT_RECOVERY 作り直しの注入を計測へ含める（同上）。"""
        self.emitted += 1

    def note_tool_call_rewrite(self) -> None:
        """TOOL_CALL_REWRITE の注入を計測へ含める（同上）。"""
        self.emitted += 1

    def note_plan_exit_nudge(self) -> None:
        """plan の承認へ繋ぐ差し戻しの注入を計測へ含める（同上）。"""
        self.emitted += 1

    def note_args_only_reply(self) -> None:
        """引数だけの JSON の本文への差し戻しの注入を計測へ含める（同上）。"""
        self.emitted += 1

    def note_no_change_report(self) -> None:
        """編集 0 回の完了報告への差し戻しの注入を計測へ含める（同上）。"""
        self.emitted += 1

    def note_repetition_abort(self) -> None:
        """反復の打ち切り後の差し戻しの注入を計測へ含める（同上）。"""
        self.emitted += 1

    def note_compaction(self) -> None:
        """圧縮の発生を記録する（次のツール結果注入で 1 回だけ発火）。"""
        self._pending_compaction = True

    def collect(
        self,
        calls: list[ToolUseBlock],
        results: list[ToolResultBlock],
        session: Session,
    ) -> list[str]:
        """このイテレーションのツール結果を観測し、発火した定型文を返す。"""
        notes: list[str] = []

        if self._pending_compaction:
            self._pending_compaction = False
            notes.append(_TEMPLATES[ReminderTrigger.POST_COMPACTION] + _todo_list_note(session))

        for r in results:
            if r.output.startswith(PLAN_MODE_DENY_PREFIX) or _deny_kind(r) is not None:
                continue
            if r.is_error:
                self._streaks[r.tool_name] = self._streaks.get(r.tool_name, 0) + 1
            else:
                self._streaks[r.tool_name] = 0
        for name, count in self._streaks.items():
            if count >= FAILURE_STREAK_THRESHOLD:
                if name not in self._streak_fired:
                    self._streak_fired.add(name)
                    note = _TEMPLATES[ReminderTrigger.TOOL_FAILURE_STREAK].format(
                        tool=name, count=count
                    )
                    if name == _EDIT_TOOL:
                        note += " " + EDIT_FILE_STREAK_HINT
                    notes.append(note)
            else:
                self._streak_fired.discard(name)

        plan_active = session.mode.main == "plan"
        denied = any(r.is_error and r.output.startswith(PLAN_MODE_DENY_PREFIX) for r in results)
        if plan_active and denied:
            if ReminderTrigger.PLAN_MODE_ACTIVE not in self._fired:
                self._fired.add(ReminderTrigger.PLAN_MODE_ACTIVE)
                notes.append(_TEMPLATES[ReminderTrigger.PLAN_MODE_ACTIVE])
        elif not plan_active:
            self._fired.discard(ReminderTrigger.PLAN_MODE_ACTIVE)

        inputs = {c.id: c.input for c in calls}
        for r in results:
            kind = _deny_kind(r)
            if kind is None:
                continue
            key = r.tool_name + json.dumps(inputs.get(r.tool_use_id), sort_keys=True, default=str)
            count = self._denials[key] = self._denials.get(key, 0) + 1
            if count >= 2:
                if count == 2:
                    notes.append(
                        _TEMPLATES[ReminderTrigger.POLICY_DENY_REPEATED].format(tool=r.tool_name)
                    )
            elif kind == "policy":
                notes.append(_TEMPLATES[ReminderTrigger.POLICY_DENY].format(tool=r.tool_name))
            elif kind == "user":
                notes.append(_TEMPLATES[ReminderTrigger.USER_DENY].format(tool=r.tool_name))

        if any(c.name == _TODO_TOOL for c in calls):
            self._iterations_since_todo = 0
            self._fired.discard(ReminderTrigger.TODO_STALE)
        else:
            self._iterations_since_todo += 1
        has_open_todo = any(t.status != "completed" for t in session.todos)
        if has_open_todo and self._iterations_since_todo >= self.todo_stale_after:
            if ReminderTrigger.TODO_STALE not in self._fired:
                self._fired.add(ReminderTrigger.TODO_STALE)
                notes.append(_TEMPLATES[ReminderTrigger.TODO_STALE])

        self.emitted += len(notes)
        return notes

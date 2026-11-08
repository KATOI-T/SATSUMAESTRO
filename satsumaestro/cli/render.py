"""Rich によるストリーミング描画・状態表示。"""

from __future__ import annotations

import difflib
import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.segment import Segment
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.text import Text

from satsumaestro.cli.live_region import LiveRegion
from satsumaestro.cli.tool_display import (
    ToolDisplay,
    describe,
    error_detail,
    error_reason,
    result_note,
)
from satsumaestro.engine.events import (
    AssistantEvent,
    MessageStop,
    TextDelta,
    Thinking,
    ToolUse,
    ToolUseDelta,
    Usage,
)
from satsumaestro.engine.repetition import RepetitionVerdict
from satsumaestro.i18n.text import ui
from satsumaestro.providers.detect import LocalKind
from satsumaestro.providers.notices import ProviderNotice
from satsumaestro.providers.notices import describe as describe_notice
from satsumaestro.session.models import PlanStepRecord, TokenUsage, ToolResultBlock
from satsumaestro.tools import edit_match
from satsumaestro.util.partial_json import partial_json_string

_TOOL_RESULT_LIMIT = 2000
_WRITE_PREVIEW_LINES = 20
_DIFF_PREVIEW_LINES = 40
_STREAM_ARG_LINES = 12
_ITEM_INDENT = 4

_LIVE_RESERVE_LINES = 3
_LIVE_MIN_LINES = 3
_FENCE_THEME = "monokai"
_FENCE_CONTEXT_LINES = 50


def _state_label(state: str | None) -> str | None:
    """状態のラベル（呼ぶたびに画面の言語で組み立てる）。"""
    labels = {
        "waiting": ui("cli.render.thinking"),
        "waiting_next": ui("cli.render.thinking_about_the_next_step"),
        "thinking": ui("cli.render.organizing_thoughts"),
        "responding": ui("cli.render.writing_the_reply"),
        "recovering": ui("cli.render.waiting_for_the_retry"),
    }
    return labels.get(state or "")


_ELAPSED_AFTER_S = 2.0

_LOAD_STALL_S = 180.0


def _load_stall_label(server: LocalKind) -> str:
    if server == "lmstudio":
        return ui("cli.render.the_model_has_not_finished_lmstudio")
    if server == "llamacpp":
        return ui("cli.render.the_model_has_not_finished_llamacpp")
    return ui("cli.render.the_model_has_not_finished")


_MODEL_WAIT_STATES = ("waiting", "waiting_next")
_LOAD_BAR_WIDTH = 40

_STEP_PREVIEW_CHARS = 60


class _StatusLine:
    """状態行（スピナー＋経過秒＋トークン数）を**描画のたびに組み立てる**renderable。"""

    def __init__(
        self,
        spinner: Spinner,
        label: str,
        since: float,
        tokens: Callable[[], str],
        *,
        stalled: tuple[float, str] | None = None,
        pulse_bar: bool = False,
    ) -> None:
        self._spinner = spinner
        self._label = label
        self._since = since
        self._tokens = tokens
        self._stalled = stalled
        self._pulse_bar = pulse_bar

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        elapsed = time.monotonic() - self._since
        label = self._label
        stalled = self._stalled is not None and elapsed >= self._stalled[0]
        if self._stalled is not None and stalled:
            label = self._stalled[1]
        text = Text(label, style="dim")
        if elapsed >= _ELAPSED_AFTER_S:
            text.append(f"  ·  {elapsed:.0f}s", style="dim")
        tok = self._tokens()
        if tok:
            text.append(f"  ·  {tok}", style="dim")
        self._spinner.update(text=text)
        yield self._spinner
        if self._pulse_bar and not stalled:
            yield Padding(
                ProgressBar(total=None, width=_LOAD_BAR_WIDTH, pulse_style="cyan"), (0, 0, 0, 2)
            )


class _TailLines:
    """Live に載せる本文を末尾 N 行に切る renderable。"""

    def __init__(self, inner: RenderableType, *, reserve: int) -> None:
        self._inner = inner
        self._reserve = reserve

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        lines = console.render_lines(self._inner, options, pad=False)
        max_lines = max(_LIVE_MIN_LINES, options.size.height - self._reserve)
        if len(lines) > max_lines:
            lines = lines[-(max_lines - 1) :]
            yield Text("⋮", style="dim")
        for line in lines:
            yield from line
            yield Segment.line()


@dataclass
class _FenceStream:
    """画面の高さを超えたコードフェンスを行ごとに確定している間の状態。"""

    lexer: str
    printed: bool = False
    context: list[str] = field(default_factory=list)


class RichRenderer:
    """ストリーミング Markdown を描画し、最下部に状態スピナー＋トークン数を出す ``Renderer``。"""

    def __init__(
        self,
        console: Console | None = None,
        *,
        show_thinking: bool = False,
        stream_tool_args: bool = True,
        loading_model: str | None = None,
        local_server: LocalKind = "ollama",
        cwd: Path | None = None,
        context_limit: int | Callable[[], int] = 0,
    ) -> None:
        self.console = console or Console()
        self._cwd = cwd
        self._context_limit = context_limit
        self._loading_model = loading_model
        self._local_server: LocalKind = local_server
        self._tty = self.console.is_terminal
        self._show_thinking = show_thinking
        self._live: LiveRegion | None = None
        self._text: list[str] = []
        self._committed_len = 0
        self._seg_committed = False
        self._thinking: list[str] = []
        self._state: str | None = None
        self._out_done = 0
        self._est_done = 0
        self._out_cur = 0
        self._cur_ascii = 0
        self._cur_other = 0
        self._in_total = 0
        self._requests = 0
        self._ctx_last = 0
        self._emitted = False
        self._phase_label: str | None = None
        self._phase_since = time.monotonic()
        self._spinner = Spinner("dots", style="cyan")
        self._stream_tool_args = stream_tool_args
        self._tool_args: dict[int, str] = {}
        self._tool_arg_names: dict[int, str] = {}
        self._tool_arg_active: int | None = None
        self._fence: _FenceStream | None = None
        self._gap_before_body = False
        self._gap_pending = False
        self._running: dict[str, tuple[ToolUse, ToolDisplay, float]] = {}
        self._group: str | None = None

    def start_turn(self) -> None:
        self._reset_segment()
        self._out_done = 0
        self._est_done = 0
        self._out_cur = 0
        self._cur_ascii = 0
        self._cur_other = 0
        self._in_total = 0
        self._requests = 0
        self._ctx_last = 0
        self._emitted = False
        self._group = None
        self._running.clear()
        self._phase_label = None
        self._phase_since = time.monotonic()
        self._set_state("waiting")
        self._refresh_live()

    def use_local_server(self, kind: LocalKind, loading_model: str | None) -> None:
        """切り替えたサーバの種類と、まだ載っていないと分かったモデルを表示に使う。"""
        self._local_server = kind
        self._loading_model = loading_model

    def end_turn(self, *, delegated: TokenUsage | None = None) -> None:
        """ターンを閉じ、TTY ならトークンの内訳を 1 行残す。"""
        self._phase_label = None
        self._loading_model = None
        self._finalize_segment()
        self._stop_live()
        if self._tty:
            line = self._turn_summary_line(delegated)
            if line is not None:
                self.console.print(Text(line, style="dim"))
            self.console.print()

    def on_assistant_event(self, event: AssistantEvent) -> None:
        self._loading_model = None
        if isinstance(event, TextDelta):
            self._text.append(event.text)
            self._count_chars(event.text)
            self._set_state("responding")
            if self._tty:
                self._flush_committable()
            self._refresh_live()
        elif isinstance(event, Thinking):
            self._count_chars(event.text)
            if not self._show_thinking:
                self._set_state("thinking")
                self._refresh_live()
            elif self._tty:
                self._emit(Text(event.text, style="dim italic"))
                self._set_state("thinking")
                self._refresh_live()
            else:
                self._thinking.append(event.text)
        elif isinstance(event, ToolUseDelta):
            self._count_chars(event.args_delta)
            if self._stream_tool_args and self._tty:
                self._tool_args[event.index] = (
                    self._tool_args.get(event.index, "") + event.args_delta
                )
                if event.name:
                    self._tool_arg_names[event.index] = event.name
                self._tool_arg_active = event.index
                self._set_state("responding")
                self._refresh_live()
        elif isinstance(event, ToolUse):
            self._clear_tool_args()
        elif isinstance(event, Usage):
            self._out_cur = event.output_tokens
            self._in_total += event.input_tokens
            self._requests += 1
            self._ctx_last = event.input_tokens + event.output_tokens
            self._refresh_live()
        elif isinstance(event, MessageStop):
            self._close_message()

    def on_tool_start(self, call: ToolUse) -> None:
        self._finalize_segment()
        display = describe(call)
        self._running[call.id] = (call, display, time.monotonic())
        if display.at_start:
            self._emit_tool(display.label, [_tool_item(d) for d in display.items])
            if self._tty:
                preview = change_preview(call, self._cwd)
                if preview is not None:
                    self._emit(Padding(preview, (0, 0, 0, _ITEM_INDENT)), separated=False)
        self.on_phase(display.status())

    def on_tool_result(self, block: ToolResultBlock) -> None:
        call, display, started = self._running.pop(
            block.tool_use_id,
            (ToolUse(id=block.tool_use_id, name=block.tool_name, input={}), None, None),
        )
        if display is None:
            display = describe(call)
        if not display.at_start:
            lines = [_tool_item(d) for d in display.items]
            if block.is_error:
                lines = lines or [Text(" " * _ITEM_INDENT)]
                lines[-1].append(f"  ✗ {error_reason(block)}", style="red")
                lines.extend(_error_detail_lines(block))
            elif lines and (note := result_note(call, block)) is not None:
                lines[-1].append(ui("cli.render.note_in_parentheses", note=note), style="dim")
            self._emit_tool(display.label, lines)
        elif block.is_error:
            self._emit_tool(
                display.label, [_failure_text(display, block), *_error_detail_lines(block)]
            )
        else:
            if block.tool_name == "bash" and block.output.strip():
                panel = Padding(_bash_output_panel(block.output), (0, 0, 0, _ITEM_INDENT))
                self._emit(panel, separated=False)
            self._emit(Text(_done_line(started), style="dim"), separated=False)
        if self._running:
            _, (_, remaining, _) = next(reversed(self._running.items()))
            self.on_phase(remaining.status())
            return
        self._phase_label = None
        self._set_state("waiting_next")
        self._phase_since = time.monotonic()
        self._refresh_live()

    def on_stream_aborted(self, timeout_recovery_s: float | None = None) -> None:
        """応答が途中で断たれたとき、そこまでの本文を確定して生成中表示を畳む。"""
        self._finalize_segment()
        if timeout_recovery_s is None:
            self._emit(
                Text(
                    ui("cli.render.the_response_was_cut_off"),
                    style="yellow",
                )
            )
            return
        seconds = f"{timeout_recovery_s:.0f}"
        self._emit(
            Text(
                ui("cli.render.stopped_no_response", seconds=seconds),
                style="yellow",
            )
        )
        self._recovering()

    def on_step_timeout_resume(
        self, timeout_s: float, index: int, total: int, attempt: int, limit: int
    ) -> None:
        """段の中で打ち切りが続き、段を小さく分けて再開するよう依頼したことを 1 行で出す。"""
        self._finalize_segment()
        final = attempt >= limit
        seconds = f"{timeout_s:.0f}"
        self._emit(
            Text(
                ui(
                    "cli.render.step_resume_last_attempt",
                    seconds=seconds,
                    index=index,
                    total=total,
                    attempt=attempt,
                )
                if final
                else ui(
                    "cli.render.step_resume_attempt",
                    seconds=seconds,
                    index=index,
                    total=total,
                    attempt=attempt,
                ),
                style="yellow",
            )
        )
        self._recovering()

    def _recovering(self) -> None:
        """打ち切りの後、出し直しを待つ状態行を点け直す。"""
        self._phase_label = None
        self._set_state("recovering")
        self._phase_since = time.monotonic()
        self._refresh_live()

    def on_provider_notice(self, notice: ProviderNotice) -> None:
        """再試行・モデルの切り替えを、ユーザー向けの 1 行で伝える。"""
        self._finalize_segment()
        self._emit(Text(f"— {describe_notice(notice)}", style="yellow"))
        self._phase_label = None
        self._set_state("recovering")
        self._phase_since = time.monotonic()
        self._refresh_live()

    def on_repetition_aborted(self, verdict: RepetitionVerdict, recovering: bool) -> None:
        """反復を検知して応答を打ち切ったとき、そこまでの本文を確定して 1 行で伝える。"""
        self._finalize_segment()
        paragraph = verdict.unit == "paragraph"
        what = ui("cli.render.the_same_paragraph") if paragraph else ui("cli.render.the_same_line")
        n = verdict.repeats
        if not recovering:
            stopped = ui("cli.render.detected_repeated_content_times_so", what=what, n=n)
            self._emit(Text(stopped, style="yellow"))
            return
        asked = ui("cli.render.detected_repeated_content_times_it", what=what, n=n)
        self._emit(Text(asked, style="yellow"))
        self._phase_label = None
        self._set_state("recovering")
        self._phase_since = time.monotonic()
        self._refresh_live()

    def on_plan_step(self, index: int, total: int, text: str) -> None:
        """承認済み計画の段に入ったことを 1 行で出す。"""
        head = text if len(text) <= _STEP_PREVIEW_CHARS else text[: _STEP_PREVIEW_CHARS - 1] + "…"
        line = ui("cli.render.step", index=index, total=total, head=head)
        self._emit(Text(line, style="cyan", no_wrap=True, overflow="ellipsis"))

    def on_plan_sendback(self, summary: str) -> None:
        """計画を予算超過で差し戻したことを 1 行で出す。"""
        self._emit(
            Text(
                ui("cli.render.plan_sent_back", summary=summary),
                style="yellow",
            )
        )

    def on_plan_step_done(self, record: PlanStepRecord) -> None:
        """段を終えたときの見積と実績を 1 行で出す（分割の根拠をその場で確かめられるように）。"""
        r = record
        estimate = r.estimated_lines
        longest = f"{r.max_response_s:.0f}"
        fields = [
            ui("cli.render.estimate_lines", n=estimate)
            if estimate is not None
            else ui("cli.render.estimate_none"),
            ui("cli.render.actual_lines", n=r.actual_lines),
            ui("cli.render.largest_write_lines_over_budget", n=r.max_write_lines)
            if r.over_budget
            else ui("cli.render.largest_write_lines", n=r.max_write_lines),
            ui("cli.render.longest_response_s", longest=longest),
        ]
        if r.edits:
            fields.append(ui("cli.render.edited_lines", n=r.edited_lines))
        head = ui("cli.render.step_done", index=r.index, total=r.total)
        line = head + " / ".join(fields)
        self._emit(Text(line, style="dim cyan"))

    def on_context_notice(self, text: str) -> None:
        """文脈の量に合わせた手当て（ツールの結果を切った等）を 1 行で出す。"""
        self._emit(Text(f"— {text}", style="yellow"))

    def _reset_segment(self) -> None:
        self._text = []
        self._committed_len = 0
        self._seg_committed = False
        self._thinking = []
        self._clear_tool_args()
        self._fence = None
        self._gap_before_body = False
        self._gap_pending = False

    def _clear_tool_args(self) -> None:
        self._tool_args.clear()
        self._tool_arg_names.clear()
        self._tool_arg_active = None

    def _set_state(self, state: str | None) -> None:
        if state != self._state and self._phase_label is None:
            self._phase_since = time.monotonic()
        self._state = state

    def on_phase(self, label: str | None) -> None:
        """いま何をしているかを状態行へ出す（``None`` でフェーズを閉じる）。"""
        if label != self._phase_label:
            self._phase_label = label
            self._phase_since = time.monotonic()
        self._refresh_live()

    def pause(self) -> None:
        """対話入力（権限カード）の間だけ Live を畳む（``console.input`` との競合回避）。"""
        self._stop_live()

    def resume(self) -> None:
        """対話入力が終わったら状態行を戻す。"""
        self._refresh_live()

    def _tail(self) -> str:
        """未 commit の本文（Live に残す末尾ブロック）。"""
        return "".join(self._text)[self._committed_len :]

    def _print_body(self, renderable: RenderableType) -> None:
        """本文チャンクをスクロールバックへ。"""
        if not self._seg_committed:
            self._open_result()
            self._emit(renderable)
            self._seg_committed = True
        else:
            if self._gap_before_body:
                self.console.print()
            self.console.print(renderable)
        self._gap_before_body = False

    def _open_result(self) -> None:
        """「考え中」フェーズ（スピナー/思考）と最初の結果の間に 1 行だけ空ける。"""
        if not self._emitted:
            self.console.print()

    def _flush_committable(self) -> None:
        """親子階層を壊さない安全境界まで本文を確定し、スクロールバックへ出す。"""
        if self._fence is None:
            if self._gap_pending:
                self._resolve_gap()
            commit, _remaining = _committable_prefix(self._tail())
            if commit:
                self._committed_len += len(commit)
                self._print_body(Markdown(commit))
            self._maybe_open_fence()
        if self._fence is not None:
            self._stream_fence(final=False)

    def _live_budget(self) -> int:
        """Live に載せられる本文の行数（端末の高さ − 状態行と余白）。"""
        return max(_LIVE_MIN_LINES, self.console.size.height - _LIVE_RESERVE_LINES)

    def _maybe_open_fence(self) -> None:
        """未確定の tail に画面の高さを超えた開きフェンスがあれば、行ごとの確定に切り替える。"""
        tail = self._tail()
        opened = _tall_open_fence(tail, self._live_budget())
        if opened is None:
            return
        start, body_start, lexer = opened
        before = tail[:start]
        self._committed_len += body_start
        self._fence = _FenceStream(lexer=lexer)
        if before.strip():
            self._print_body(Markdown(before))
            self.console.print()

    def _stream_fence(self, *, final: bool) -> None:
        """流しているフェンスの本体のうち、確定できる行をスクロールバックへ出す。"""
        fence = self._fence
        if fence is None:
            return
        body = self._tail()
        head, sep, partial = body.rpartition("\n")
        lines = head.split("\n") if sep else []
        closer = next((i for i, ln in enumerate(lines) if _FENCE_CLOSE_LINE.fullmatch(ln)), None)
        if closer is not None:
            consumed = sum(len(ln) + 1 for ln in lines[:closer]) + len(lines[closer])
            self._committed_len += consumed
            self._fence = None
            self._print_fence_chunk(fence, lines[:closer], last=True)
            self._close_fence(body[consumed:])
            return
        if final:
            code_lines = lines if _FENCE_CLOSE_LINE.fullmatch(partial) else [*lines, partial]
            self._committed_len += len(body)
            self._fence = None
            self._print_fence_chunk(fence, code_lines, last=True)
            self._close_fence(None)
            return
        last_nonblank = next((i for i in range(len(lines) - 1, -1, -1) if lines[i].strip()), None)
        if not last_nonblank:
            return
        chunk = lines[:last_nonblank]
        self._committed_len += sum(len(ln) + 1 for ln in chunk)
        self._print_fence_chunk(fence, chunk, last=False)

    def _print_fence_chunk(self, fence: _FenceStream, lines: list[str], *, last: bool) -> None:
        """コード行を ``Syntax`` で出す。"""
        if last:
            code = "\n".join(lines).rstrip()
            lines = code.split("\n") if code else []
        elif not lines:
            return
        pad = (0 if fence.printed else 1, 1, 1 if last else 0, 1)
        context = fence.context[-_FENCE_CONTEXT_LINES:] if lines else []
        syntax = Syntax(
            "\n".join([*context, *lines]),
            fence.lexer,
            theme=_FENCE_THEME,
            word_wrap=True,
            padding=pad,
            line_range=(len(context) + 1, None) if context else None,
        )
        self._print_body(syntax)
        fence.printed = True
        fence.context = [*fence.context, *lines][-_FENCE_CONTEXT_LINES:]

    def _close_fence(self, rest: str | None) -> None:
        """流しを終える。"""
        self._fence = None
        self._gap_before_body = False
        self._gap_pending = False
        if rest is not None:
            self._resolve_gap(rest)

    def _resolve_gap(self, rest: str | None = None) -> None:
        """閉じフェンスの後ろ ``rest``（省略時は今の tail）から、次の本文の前の 1 行を決める。"""
        rest = self._tail() if rest is None else rest
        after = rest[1:] if rest.startswith("\n") else rest
        if not after:
            self._gap_pending = True
            return
        self._gap_pending = False
        self._gap_before_body = not after.startswith("\n")

    def _count_chars(self, text: str) -> None:
        """現メッセージの文字数を足す（実測が届くまでの概算の材料）。"""
        if text.isascii():
            self._cur_ascii += len(text)
            return
        ascii_n = sum(1 for ch in text if ch.isascii())
        self._cur_ascii += ascii_n
        self._cur_other += len(text) - ascii_n

    def _close_message(self) -> None:
        """メッセージの境目: 現メッセージの数を確定分へ繰り入れ、現メッセージの数を 0 に戻す。"""
        if self._out_cur > 0:
            self._out_done += self._out_cur
        else:
            self._est_done += _rough_tokens(self._cur_ascii, self._cur_other)
        self._out_cur = 0
        self._cur_ascii = 0
        self._cur_other = 0

    def _turn_tokens(self) -> int:
        """ターンの出力トークン数。"""
        rough = _rough_tokens(self._cur_ascii, self._cur_other)
        current = self._out_cur if self._out_cur > 0 else rough
        return self._out_done + self._est_done + current

    def _output_label(self) -> str:
        """状態行のトークン区間（``出力 1.2K``。0 なら空）。"""
        tok = self._turn_tokens()
        if tok <= 0:
            return ""
        shown = _format_tokens(tok)
        return ui("cli.render.output_shown", shown=shown)

    def _generating(self) -> bool:
        """モデルが生成中（思考・本文が流れていて、出力の数が動く区間）か。"""
        return self._phase_label is None and self._state in ("thinking", "responding")

    def _turn_summary_line(self, delegated: TokenUsage | None) -> str | None:
        """ターン末尾の内訳 1 行。"""
        tok = self._turn_tokens()
        if tok <= 0 and self._requests == 0:
            return None
        out = _format_tokens(tok)
        parts = [ui("cli.render.summary_output", out=out)]
        if self._requests > 0:
            inp, n = _format_tokens(self._in_total), self._requests
            parts.append(ui("cli.render.input_requests", n=n, inp=inp))
        if delegated is not None and (delegated.input_tokens > 0 or delegated.output_tokens > 0):
            d_in = _format_tokens(delegated.input_tokens)
            d_out = _format_tokens(delegated.output_tokens)
            parts.append(ui("cli.render.delegated_input_output", d_in=d_in, d_out=d_out))
        if self._requests > 0:
            ctx = _format_tokens(self._ctx_last)
            limit = self._context_limit() if callable(self._context_limit) else self._context_limit
            if limit > 0:
                ctx += f"/{_format_tokens(limit)}"
            parts.append(ui("cli.render.context", ctx=ctx))
        return "— " + " · ".join(parts)

    def waiting_since(self) -> float | None:
        """モデルの最初のイベントを待っている区間の始まり（単調時計）。"""
        if self._phase_label is not None or self._loading_model is not None:
            return None
        if self._state not in _MODEL_WAIT_STATES:
            return None
        return self._phase_since

    def on_model_loading(self, model: str, since: float) -> None:
        """待ちの途中で、モデルがサーバに載っていないと分かった（ロードから始まっている）。"""
        if self.waiting_since() != since:
            return
        self._loading_model = model
        self._refresh_live()

    def _status_row(self) -> RenderableType | None:
        if (
            self._phase_label is None
            and self._state in _MODEL_WAIT_STATES
            and self._loading_model is not None
        ):
            model = self._loading_model
            return _StatusLine(
                self._spinner,
                ui("cli.render.loading", model=model),
                self._phase_since,
                _no_tokens,
                stalled=(_LOAD_STALL_S, _load_stall_label(self._local_server)),
                pulse_bar=True,
            )
        label = self._phase_label or _state_label(self._state)
        if label is None:
            return None
        tokens = self._output_label if self._generating() else _no_tokens
        return _StatusLine(self._spinner, label, self._phase_since, tokens)

    def _tool_arg_panel(self) -> RenderableType | None:
        """生成中の tool_call 引数から、書き込まれる本文の末尾を出す。"""
        active = self._tool_arg_active
        if active is None:
            return None
        done: list[Text] = []
        for index in sorted(self._tool_args):
            if index == active:
                continue
            streamed = _streamed_write(self._tool_args[index])
            if streamed is not None:
                path, value = streamed
                label = path or self._tool_arg_names.get(index) or "…"
                n = len(value.splitlines())
                done.append(Text(ui("cli.render.done_with_lines", n=n, label=label), style="dim"))
        current = _streamed_write(self._tool_args[active])
        if current is None and not done:
            return None
        name = self._tool_arg_names.get(active) or "edit"
        title = ui("cli.render.generating", name=name)
        if len(self._tool_args) > 1:
            title += ui("cli.render.call_number", number=active + 1)
        body: list[RenderableType] = [*done]
        if current is not None:
            path, value = current
            if path:
                title += f" {path}"
            lines = value.splitlines() or [""]
            title += ui("cli.render.title_lines", n=len(lines))
            body.append(Text("\n".join(lines[-_STREAM_ARG_LINES:]), style="dim"))
        return Panel(Group(*body), title=title, border_style="cyan", expand=False)

    def _live_renderable(self) -> RenderableType:
        body: list[RenderableType] = []
        tail = self._tail()
        if self._fence is not None:
            shown = tail.rstrip("\n")
            if shown:
                lexer = self._fence.lexer
                body.append(
                    Syntax(shown, lexer, theme=_FENCE_THEME, word_wrap=True, padding=(0, 1))
                )
        elif tail.strip():
            body.append(Markdown(tail))
        panel = self._tool_arg_panel()
        if panel is not None:
            body.append(panel)
        parts: list[RenderableType] = []
        if body:
            parts.append(_TailLines(Group(*body), reserve=_LIVE_RESERVE_LINES))
        status = self._status_row()
        if status is not None:
            parts.append(status)
        return Group(*parts)

    def _refresh_live(self) -> None:
        if not self._tty:
            return
        if self._live is None:
            self._live = LiveRegion(self.console, fps=12.5, on_print=self._sync_region)
            self._live.start()
        self._live.update(self._live_renderable())

    def _sync_region(self) -> None:
        """確定内容を出す直前に、領域の内容をいまの状態に合わせる。"""
        if self._live is not None:
            self._live.update(self._live_renderable())

    def _stop_live(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    def _emit(self, renderable: RenderableType, *, separated: bool = True) -> None:
        """確定コンテンツをスクロールバックへ出す。"""
        if separated and self._emitted:
            self.console.print()
        self.console.print(renderable)
        self._emitted = True
        self._group = None

    def _emit_tool(self, label: str, lines: list[Text]) -> None:
        """ツールの見出しと項目を出す。"""
        self._open_result()
        if self._group != label:
            self._emit(Text.assemble((ui("cli.render.bullet"), "cyan bold"), (label, "cyan bold")))
        for line in lines:
            self.console.print(line)
        self._emitted = True
        self._group = label

    def _finalize_segment(self) -> None:
        self._close_message()
        if self._tty:
            self._stop_live()
            if self._fence is not None:
                self._stream_fence(final=True)
            if self._gap_pending:
                self._resolve_gap()
            tail = self._tail()
            if tail.strip():
                self._print_body(Markdown(tail))
        elif "".join(self._text).strip() or self._thinking:
            parts: list[RenderableType] = []
            if self._thinking:
                parts.append(Text("".join(self._thinking), style="dim italic"))
            if "".join(self._text).strip():
                parts.append(Markdown("".join(self._text)))
            self._emit(Group(*parts))
        self._reset_segment()


def _done_line(started: float | None) -> str:
    """ツール完了行。"""
    elapsed = None if started is None else time.monotonic() - started
    if elapsed is None or elapsed < 0.1:
        return " " * _ITEM_INDENT + ui("cli.render.done")
    shown = f"{elapsed:.1f}s" if elapsed < 10 else f"{elapsed:.0f}s"
    return " " * _ITEM_INDENT + ui("cli.render.done_with_summary", shown=shown)


def _rough_tokens(ascii_chars: int, other_chars: int) -> int:
    """表示専用の粗いトークン概算。"""
    return ascii_chars // 4 + other_chars


def _no_tokens() -> str:
    """状態行にトークンを出さない区間（ツール実行中・次の応答の待ち）用。"""
    return ""


def _format_tokens(tok: int) -> str:
    """トークン数を整形する。"""
    if tok < 1000:
        return str(tok)
    if tok < 999_500:
        return f"{tok / 1000:.3g}K"
    return f"{tok / 1_000_000:.3g}M"


_BLANK_RUN = re.compile(r"\n[ \t]*\n+")
_LIST_MARKER = re.compile(r"(?:[-*+]|\d+[.)])\s")


def _block_is_listish(block: str) -> bool:
    """ブロックがリスト/インデント継続（親子階層を持つ）かどうか。"""
    for line in block.splitlines():
        if line.strip():
            return bool(_LIST_MARKER.match(line.lstrip())) or line[:1] in (" ", "\t")
    return False


def _committable_prefix(s: str) -> tuple[str, str]:
    """``s`` の先頭から、階層を壊さず確定できる範囲 ``(commit, remaining)`` を返す。"""
    cuts = [m.end() for m in _BLANK_RUN.finditer(s)]
    if not cuts:
        return "", s
    starts = [0, *cuts]
    blocks = [s[starts[i] : starts[i + 1]] for i in range(len(starts) - 1)] + [s[starts[-1] :]]
    best = 0
    for i in range(len(blocks) - 1):
        cut = starts[i + 1]
        if not blocks[i + 1].strip():
            continue
        if s[:cut].count("```") % 2 != 0:
            continue
        if _block_is_listish(blocks[i]) and _block_is_listish(blocks[i + 1]):
            continue
        best = cut
    return s[:best], s[best:]


_FENCE_OPEN_LINE = re.compile(r"```(\S*)")
_FENCE_CLOSE_LINE = re.compile(r" {0,3}```[ \t]*")


def _fence_opener(s: str) -> tuple[int, int, str] | None:
    """``s`` の中で閉じていない列 0 の開きフェンスの ``(開始位置, 本体の開始位置, lexer)``。"""
    offset = 0
    opened: tuple[int, int, str] | None = None
    for line in s.splitlines(keepends=True):
        bare = line.rstrip("\r\n")
        if opened is None:
            m = _FENCE_OPEN_LINE.match(bare)
            if m is not None:
                if not line.endswith("\n"):
                    return None
                lexer = m.group(1).partition(" ")[0] or "text"
                opened = (offset, offset + len(line), lexer)
        elif _FENCE_CLOSE_LINE.fullmatch(bare):
            opened = None
        offset += len(line)
    return opened


def _tall_open_fence(s: str, budget: int) -> tuple[int, int, str] | None:
    """本体の完成行が ``budget`` 行以上ある開きフェンス（:func:`_fence_opener` の形）。"""
    opened = _fence_opener(s)
    if opened is None or s.count("\n", opened[1]) < budget:
        return None
    return opened


def _streamed_write(buf: str) -> tuple[str | None, str] | None:
    """生成中の tool_call 引数から ``(path, 本文)`` を取り出す。"""
    for key in ("new", "content"):
        value = partial_json_string(buf, key)
        if value is not None:
            return partial_json_string(buf, "path"), value
    return None


def _tool_item(detail: str) -> Text:
    """見出しの下の 1 項目（``    - <対象>``）。"""
    return Text.assemble((" " * _ITEM_INDENT + "- ", "dim"), (detail, ""))


def _failure_text(display: ToolDisplay, block: ToolResultBlock) -> Text:
    """開始時に出したツールの失敗行。"""
    target = f"{display.items[0]} — " if display.items else ""
    return Text(" " * _ITEM_INDENT + f"✗ {target}{error_reason(block)}", style="red")


def _error_detail_lines(block: ToolResultBlock) -> list[Text]:
    """失敗理由の 2 行目以降。"""
    indent = " " * (_ITEM_INDENT + 2)
    return [Text(indent + line, style="red dim") for line in error_detail(block)]


def _bash_output_panel(output: str) -> RenderableType:
    body = (
        output
        if len(output) <= _TOOL_RESULT_LIMIT
        else output[:_TOOL_RESULT_LIMIT] + "\n… (truncated)"
    )
    return Panel(Text(body.rstrip(), style="dim"), border_style="green", expand=False)


def action_preview(
    call: ToolUse, required: str, detailed: bool = False, warning: str | None = None
) -> RenderableType:
    """権限確認カード: 「何を実行するか」をツール別に整形する。"""
    title = ui("cli.render.permission", name=call.name, required=required)
    body = _preview_body(call, detailed)
    if warning is not None:
        body = Group(Text(f"⚠ {warning}", style="yellow bold"), body)
    return Panel(body, title=title, border_style="yellow", expand=False)


def change_preview(call: ToolUse, cwd: Path | None = None) -> RenderableType | None:
    """ファイル変更ツールの「これから何を変えるか」。"""
    inp = call.input
    path = str(inp.get("path", ""))
    if call.name == "edit_file":
        old, new = str(inp.get("old", "")), str(inp.get("new", ""))
        current = _read_current(cwd, path)
        if isinstance(current, str) and old and old in current:
            after = (
                current.replace(old, new)
                if inp.get("replace_all")
                else current.replace(old, new, 1)
            )
            return _numbered_diff(current, after)
        if isinstance(current, str) and old:
            shifted = edit_match.reindent_edit(current, old, new)
            if shifted is not None:
                return _numbered_diff(current, current.replace(shifted.old, shifted.new, 1))
        return _diff_preview(path, old, new, False)
    if call.name == "delete_path":
        return _delete_preview(path, cwd)
    if call.name == "write_file":
        content = str(inp.get("content", ""))
        current = _read_current(cwd, path)
        if current is _MISSING:
            return _numbered_new_file(content)
        if isinstance(current, str):
            return _numbered_diff(current, content)
        return _write_preview(path, content, False)
    if call.name == "append_file":
        content = str(inp.get("content", ""))
        current = _read_current(cwd, path)
        if current is _MISSING:
            return _numbered_new_file(content)
        if isinstance(current, str):
            return _numbered_diff(current, current + content)
        return _write_preview(path, content, False)
    return None


_PREVIEW_READ_LIMIT = 1_000_000
_MISSING = object()


def _read_current(cwd: Path | None, path: str) -> str | object | None:
    """変更前の内容。"""
    if cwd is None or not path:
        return None
    target = Path(path) if Path(path).is_absolute() else cwd / path
    try:
        if not target.resolve().is_relative_to(cwd.resolve()):
            return None
        if not target.exists():
            return _MISSING
        if not target.is_file() or target.stat().st_size > _PREVIEW_READ_LIMIT:
            return None
        return target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _numbered_diff(before: str, after: str) -> RenderableType:
    """ファイル内の行番号付きで、変わる箇所（前後 2 行の文脈つき）だけを出す。"""
    a, b = before.splitlines(), after.splitlines()
    width = len(str(max(len(a), len(b), 1)))
    rows: list[tuple[str, str]] = []
    matcher = difflib.SequenceMatcher(None, a, b)
    for n, group in enumerate(matcher.get_grouped_opcodes(2)):
        if n:
            rows.append((" " * width + " ⋮", "dim"))
        for tag, i1, i2, j1, j2 in group:
            if tag == "equal":
                rows.extend((f"{i + 1:>{width}}   {a[i]}", "dim") for i in range(i1, i2))
                continue
            rows.extend((f"{i + 1:>{width}} - {a[i]}", "red") for i in range(i1, i2))
            rows.extend((f"{j + 1:>{width}} + {b[j]}", "green") for j in range(j1, j2))
    if not rows:
        return Text(ui("cli.render.no_changes"), style="dim")
    text = Text()
    for line, style in rows[:_DIFF_PREVIEW_LINES]:
        text.append(line + "\n", style=style)
    if len(rows) > _DIFF_PREVIEW_LINES:
        text.append(_rest_omitted(), style="dim italic")
    return text


def _rest_omitted() -> str:
    return ui("cli.render.rest_omitted")


_DELETE_PREVIEW_ENTRIES = 10


def _delete_preview(path: str, cwd: Path | None) -> RenderableType | None:
    """削除されるディレクトリの中身（先頭 10 件＋件数）。"""
    if cwd is None or not path:
        return None
    target = Path(path) if Path(path).is_absolute() else cwd / path
    try:
        if target.is_symlink() or not target.is_dir():
            return None
        if not target.resolve().is_relative_to(cwd.resolve()):
            return None
        entries: list[str] = []
        total = 0
        for root, dirs, files in os.walk(target):
            dirs.sort()
            for name in sorted(dirs) + sorted(files):
                total += 1
                if len(entries) < _DELETE_PREVIEW_ENTRIES:
                    rel = (Path(root) / name).relative_to(target).as_posix()
                    entries.append(rel + ("/" if name in dirs else ""))
    except OSError:
        return None
    if total == 0:
        return Text(ui("cli.render.empty_directory"), style="dim")
    text = Text(
        ui("cli.render.deletes_everything_inside_entries", n=total),
        style="red",
    )
    for entry in entries:
        text.append(f"  {entry}\n", style="dim")
    if total > len(entries):
        rest = total - len(entries)
        text.append(ui("cli.render.and_more", rest=rest), style="dim italic")
    return text


def _numbered_new_file(content: str) -> RenderableType:
    """新規ファイルは先頭を行番号付きで出す。"""
    lines = content.splitlines()
    width = len(str(max(min(len(lines), _WRITE_PREVIEW_LINES), 1)))
    text = Text(ui("cli.render.new_file"), style="bold")
    for i, line in enumerate(lines[:_WRITE_PREVIEW_LINES]):
        text.append(f"{i + 1:>{width}} + {line}\n", style="green")
    if len(lines) > _WRITE_PREVIEW_LINES:
        text.append(_rest_omitted(), style="dim italic")
    return text


def _diff_stats(old: str, new: str) -> tuple[int, int]:
    """追加行数・削除行数（権限カードの要約用）。"""
    add = rm = 0
    for ln in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm=""):
        if ln.startswith("+") and not ln.startswith("+++"):
            add += 1
        elif ln.startswith("-") and not ln.startswith("---"):
            rm += 1
    return add, rm


def _preview_body(call: ToolUse, detailed: bool) -> RenderableType:
    inp = call.input
    if call.name == "bash":
        return Syntax(str(inp.get("command", "")), "bash", theme="ansi_dark", word_wrap=True)
    if call.name == "edit_file":
        path, old, new = (
            str(inp.get("path", "")),
            str(inp.get("old", "")),
            str(inp.get("new", "")),
        )
        if detailed:
            return _diff_preview(path, old, new, True)
        add, rm = _diff_stats(old, new)
        return Text.assemble(
            (f"{path}  ", "bold"),
            (f"+{add} ", "green"),
            (f"-{rm}", "red"),
            (
                ui("cli.render.diff_shown_above_d_for"),
                "dim",
            ),
        )
    if call.name in ("write_file", "append_file"):
        path, content = str(inp.get("path", "")), str(inp.get("content", ""))
        if detailed:
            return _write_preview(path, content, True)
        append = call.name == "append_file"
        n, size = len(content.splitlines()), len(content)
        return Text.assemble(
            (f"{path}", "bold"),
            (
                ui("cli.render.append_lines_bytes", n=n, size=size)
                if append
                else ui("cli.render.card_lines_bytes", n=n, size=size),
                "green",
            ),
            (
                ui("cli.render.content_shown_above_d_for"),
                "dim",
            ),
        )
    if call.name == "delete_path":
        recursive = bool(inp.get("recursive"))
        return Text.assemble(
            (str(inp.get("path", "")), "bold"),
            (
                ui("cli.render.deletes_the_directory_and_everything") if recursive else "",
                "red",
            ),
            (
                ui("cli.render.deletion_cannot_be_undone_this"),
                "dim",
            ),
        )
    if call.name in ("web_fetch", "web_search"):
        return Text(str(inp.get("url") or inp.get("query", "")))
    try:
        return Text(json.dumps(inp, ensure_ascii=False, indent=2))
    except (TypeError, ValueError):
        return Text(str(inp))


def _diff_preview(path: str, old: str, new: str, detailed: bool) -> RenderableType:
    lines = list(
        difflib.unified_diff(
            old.splitlines(), new.splitlines(), fromfile=path, tofile=path, lineterm=""
        )
    )
    truncated = not detailed and len(lines) > _DIFF_PREVIEW_LINES
    shown = lines[:_DIFF_PREVIEW_LINES] if truncated else lines
    text = Text()
    for ln in shown:
        style = "dim"
        if ln.startswith("+") and not ln.startswith("+++"):
            style = "green"
        elif ln.startswith("-") and not ln.startswith("---"):
            style = "red"
        elif ln.startswith("@@"):
            style = "cyan"
        text.append(ln + "\n", style=style)
    if truncated:
        text.append(_lines_omitted(len(lines) - _DIFF_PREVIEW_LINES) + "\n", style="dim italic")
    return text


def _lines_omitted(n: int) -> str:
    return ui("cli.render.lines_omitted_d_for_full", n=n)


def _write_preview(path: str, content: str, detailed: bool) -> RenderableType:
    lines = content.splitlines()
    truncated = not detailed and len(lines) > _WRITE_PREVIEW_LINES
    shown = lines[:_WRITE_PREVIEW_LINES] if truncated else lines
    n, size = len(lines), len(content)
    text = Text(
        ui("cli.render.preview_lines_bytes", n=n, path=path, size=size),
        style="bold",
    )
    text.append("\n".join(shown), style="dim")
    if truncated:
        text.append("\n" + _lines_omitted(len(lines) - _WRITE_PREVIEW_LINES), style="dim italic")
    return text

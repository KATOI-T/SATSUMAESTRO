"""Rich によるストリーミング描画・状態表示。"""

from __future__ import annotations

import difflib
import functools
import json
import math
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from rich.cells import cell_len
from rich.color import Color
from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.padding import Padding
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.segment import Segment
from rich.spinner import Spinner
from rich.style import Style
from rich.syntax import Syntax
from rich.text import Text

from satsumaestro.cli.diff_view import diff_view, new_file_view, shown_context
from satsumaestro.cli.live_region import LiveRegion
from satsumaestro.cli.markdown import ReplyMarkdown, code_block
from satsumaestro.cli.tool_display import (
    ToolDisplay,
    describe,
    error_detail,
    error_reason,
    result_note,
)
from satsumaestro.config.models import ToolOutputMode
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
from satsumaestro.session.redact import (
    redact_lines,
    redact_secrets,
    redact_shown_input,
    secret_values,
)
from satsumaestro.tools import edit_match
from satsumaestro.util.partial_json import partial_json_string

_TOOL_RESULT_LIMIT = 2000
# 畳んだ bash の出力を /expand で出し直せるよう持っておく件数（古いものから捨てる）
COLLAPSED_KEEP = 20
# この行数以下の bash の出力は畳まずに出す
SHORT_OUTPUT_LINES = 3
_WRITE_PREVIEW_LINES = 20
_DIFF_PREVIEW_LINES = 40
_NEW_FILE_PREVIEW_LINES = 10
_STREAM_ARG_LINES = 12
# 本文とツールの行の頭の印、結果の行の頭の印（2 行目からは同じ幅だけ下げる）
_MARK = "⏺ "
_RESULT_MARK = "  ⎿  "

# 状態行の上の 1 行・スピナー・ロード中の帯・余白
_LIVE_RESERVE_LINES = 4
_LIVE_MIN_LINES = 3
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
# ラベルの上を左から右へ流れる光（中心からの距離で灰から白へ混ぜる）
_SHIMMER_SPEED = 24.0  # 桁/秒
_SHIMMER_RADIUS = 4.0
_SHIMMER_GAP_S = 0.6
_SHIMMER_BASE = (118, 118, 118)
_SHIMMER_PEAK = (242, 242, 242)
_SHIMMER_LEVELS = 16

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


def _shimmer(label: str, now: float) -> Text:
    """ラベルを灰で描き、光の中心に近い字ほど明るくする（光の位置は時刻だけで決まる）。"""
    width = cell_len(label)
    period = (width + 2 * _SHIMMER_RADIUS) / _SHIMMER_SPEED + _SHIMMER_GAP_S
    center = (now % period) * _SHIMMER_SPEED - _SHIMMER_RADIUS
    text = Text()
    col = 0
    for ch in label:
        w = cell_len(ch)
        d = abs(col + w / 2 - center)
        glow = 0.5 + 0.5 * math.cos(math.pi * d / _SHIMMER_RADIUS) if d < _SHIMMER_RADIUS else 0.0
        text.append(ch, style=_glow_style(round(glow * _SHIMMER_LEVELS)))
        col += w
    return text


@functools.cache
def _glow_style(level: int) -> Style:
    t = level / _SHIMMER_LEVELS
    rgb = (round(b + (p - b) * t) for b, p in zip(_SHIMMER_BASE, _SHIMMER_PEAK, strict=True))
    return Style(color=Color.from_rgb(*rgb))


class _StatusLine:
    """状態行（スピナー・光の流れるラベル・経過）を**描画のたびに組み立てる**renderable。"""

    def __init__(
        self,
        spinner: Spinner,
        label: str,
        since: float,
        *,
        stalled: tuple[float, str] | None = None,
        pulse_bar: bool = False,
    ) -> None:
        self._spinner = spinner
        self._label = label
        self._since = since
        self._stalled = stalled
        self._pulse_bar = pulse_bar

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        now = time.monotonic()
        elapsed = now - self._since
        label = self._label
        stalled = self._stalled is not None and elapsed >= self._stalled[0]
        if self._stalled is not None and stalled:
            label = self._stalled[1]
        # 停滞した後は光も帯も止め、進んでいるように見せない
        text = Text(label, style="dim") if stalled else _shimmer(label, now)
        if elapsed >= _ELAPSED_AFTER_S:
            text.append("  " + _format_elapsed(elapsed), style="dim")
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


class _Gutter:
    """中身の見える最初の行の頭に印を置き、ほかの行を印と同じ幅だけ下げる renderable。"""

    def __init__(self, inner: RenderableType, mark: Text) -> None:
        self._inner = inner
        self._mark = mark

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        width = self._mark.cell_len
        inner = options.update_width(max(1, options.max_width - width))
        lines = console.render_lines(self._inner, inner, pad=False)
        # Rich のリストやコードブロックは空行から始まるので、印は空でない最初の行に置く
        first = next((i for i, ln in enumerate(lines) if "".join(s.text for s in ln).strip()), 0)
        blank = Segment(" " * width)
        for i, line in enumerate(lines):
            if i == first:
                yield from self._mark.render(console, end="")
            elif Segment.get_line_length(line) > 0:
                yield blank
            yield from line
            yield Segment.line()


@dataclass
class _Running:
    """実行中のツール（``call`` は画面に出す写し・``shown`` は結果より先に行を出したか）。"""

    call: ToolUse
    display: ToolDisplay
    started: float
    preview: RenderableType | None = None
    shown: bool = False


@dataclass
class _FenceStream:
    """画面の高さを超えたコードフェンスを行ごとに確定している間の状態。"""

    lang: str  # info 文字列の先頭語（無ければ空）
    printed: bool = False
    context: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CollapsedOutput:
    """畳んだ bash の出力（``number`` は ``/expand <number>`` の番号）。"""

    number: int
    command: str
    output: str


class RichRenderer:
    """ストリーミング Markdown を描画し、最下部に状態行を出す ``Renderer``。"""

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
        redact: bool = False,
        tool_output: ToolOutputMode = "full",
    ) -> None:
        self.console = console or Console()
        self._cwd = cwd
        self._redact = redact
        # summary は SHORT_OUTPUT_LINES を超える bash の出力を畳み、直近 COLLAPSED_KEEP 件を持つ
        self._tool_output = tool_output
        self._collapsed: dict[int, CollapsedOutput] = {}
        self._collapsed_count = 0
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
        self._turn_since: float | None = None
        self._spinner = Spinner("dots", style="cyan")
        self._stream_tool_args = stream_tool_args and not redact
        self._tool_args: dict[int, str] = {}
        self._tool_arg_names: dict[int, str] = {}
        self._tool_arg_active: int | None = None
        self._fence: _FenceStream | None = None
        self._gap_before_body = False
        self._gap_pending = False
        self._running: dict[str, _Running] = {}
        # 結果の行の無いツールが同じラベルで続く間は、空行を挟まない
        self._group: str | None = None
        self._last_tool: str | None = None

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
        self._last_tool = None
        self._running.clear()
        self._phase_label = None
        self._phase_since = time.monotonic()
        self._turn_since = self._phase_since
        self._set_state("waiting")
        self._refresh_live()

    def use_local_server(self, kind: LocalKind, loading_model: str | None) -> None:
        """切り替えたサーバの種類と、まだ載っていないと分かったモデルを表示に使う。"""
        self._local_server = kind
        self._loading_model = loading_model

    def end_turn(self, *, delegated: TokenUsage | None = None) -> None:
        """ターンを閉じ、TTY ならトークンの内訳を 1 行残す。"""
        self._phase_label = None
        turn_since, self._turn_since = self._turn_since, None
        elapsed = 0.0 if turn_since is None else time.monotonic() - turn_since
        self._loading_model = None
        self._finalize_segment()
        self._show_pending()
        self._stop_live()
        if self._tty:
            line = self._turn_summary_line(delegated, elapsed)
            if line is not None:
                self.console.print()
                self.console.print(Text(" " * len(_MARK) + line, style="dim"))
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
        shown = _redacted_call(call) if self._redact else call
        display = describe(shown)
        entry = _Running(shown, display, time.monotonic())
        self._running[call.id] = entry
        if display.at_start and self._tty:
            # 差分は実行の前に作り、結果と一緒か権限カードの前に出す
            entry.preview = change_preview(call, self._cwd, redact=self._redact)
            if isinstance(entry.preview, Text):
                entry.preview.rstrip()
        elif display.at_start:
            self._show_header(entry, "")
        self.on_phase(display.status())

    def on_tool_result(self, block: ToolResultBlock) -> None:
        entry = self._running.pop(block.tool_use_id, None)
        if entry is None:
            call = ToolUse(id=block.tool_use_id, name=block.tool_name, input={})
            entry = _Running(call, describe(call), time.monotonic())
        rows = self._result_rows(entry, block)
        if entry.shown:
            if rows:
                self._emit(_result_gutter(rows), separated=False)
                self._last_tool = entry.call.id
        else:
            self._emit_tool(entry, "red" if block.is_error else "green", rows)
        if self._running:
            remaining = next(reversed(self._running.values()))
            self.on_phase(remaining.display.status())
            return
        self._phase_label = None
        self._set_state("waiting_next")
        self._phase_since = time.monotonic()
        self._refresh_live()

    def _result_rows(self, entry: _Running, block: ToolResultBlock) -> list[RenderableType]:
        """``⎿`` に続ける結果の行（行をまだ出していなければ、対象の一覧と差分を先に置く）。"""
        call, display = entry.call, entry.display
        rows: list[RenderableType] = [] if entry.shown else _lead_rows(entry)
        if block.is_error:
            # 先に出した行から離れたとき（並列実行など）は、どれの失敗かを添える
            apart = entry.shown and self._last_tool != call.id
            target = f"{display.items[0]} — " if apart and display.items else ""
            rows.append(Text(f"✗ {target}{error_reason(block)}", style="red"))
            rows.extend(_error_detail_lines(block))
            return rows
        if not display.at_start:
            note = result_note(call, block)
            if note is not None:
                rows.append(Text(note, style="dim"))
            return rows
        done = _done_line(entry.started)
        if block.tool_name == "bash" and block.output.strip():
            n = len(block.output.rstrip("\n").split("\n"))
            if self._tool_output == "summary" and n > SHORT_OUTPUT_LINES:
                kept = self._keep_collapsed(call, block.output)
                done += ui("cli.render.collapsed_output", n=n, number=kept.number)
            else:
                rows.append(_bash_output_text(block.output))
        rows.append(Text(done, style="dim"))
        return rows

    def _keep_collapsed(self, call: ToolUse, output: str) -> CollapsedOutput:
        self._collapsed_count += 1
        command = str(call.input.get("command", ""))
        entry = CollapsedOutput(self._collapsed_count, command, output)
        self._collapsed[entry.number] = entry
        if len(self._collapsed) > COLLAPSED_KEEP:
            del self._collapsed[next(iter(self._collapsed))]
        return entry

    def collapsed_output(self, number: int | None = None) -> CollapsedOutput | None:
        """畳んだ bash の出力（``number`` を省くと最後の 1 つ。持っていなければ None）。"""
        if number is None:
            return next(reversed(self._collapsed.values()), None)
        return self._collapsed.get(number)

    def _show_header(self, entry: _Running, dot: str) -> None:
        """結果より先にツールの行を出す（結果は後で ``⎿`` の行だけを足す）。"""
        self._emit_tool(entry, dot, _lead_rows(entry))
        entry.shown = True

    def _show_pending(self) -> None:
        """結果の来ていないツールの行を灰の印で出す（権限カードの前・ターンの終わり）。"""
        for entry in self._running.values():
            if not entry.shown:
                self._show_header(entry, "dim")

    def _pending_rows(self) -> list[RenderableType]:
        """最下部の領域に出す、実行中のツールの行。"""
        pending = [entry for entry in self._running.values() if not entry.shown]
        rows: list[RenderableType] = [_tool_header(entry.display, "dim") for entry in pending]
        if rows:
            rows.insert(0, Text(""))  # 確定したときの区切りの 1 行と位置を揃える
        return rows

    def on_stream_aborted(self, timeout_recovery_s: float | None = None) -> None:
        """応答が断たれたとき、本文を確定して生成中表示を畳む（何も届いていなければ切れた行は出さない）。"""
        received = bool(self._cur_ascii or self._cur_other or self._out_cur)
        self._finalize_segment()
        if timeout_recovery_s is None:
            if not received:
                return
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
        """対話入力（権限カード）の間だけ Live を畳み、実行中のツールの行は先に確定する。"""
        self._show_pending()
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
            self._emit(self._body(renderable))
            self._seg_committed = True
        else:
            if self._gap_before_body:
                self.console.print()
            self.console.print(self._body(renderable))
        self._gap_before_body = False

    def _body(self, renderable: RenderableType) -> RenderableType:
        """本文をツールの行と同じ位置に下げ、まとまりの最初のチャンクにだけ ``⏺`` を付ける。"""
        mark = " " * len(_MARK) if self._seg_committed else _MARK
        return _Gutter(renderable, Text(mark))

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
                self._print_body(ReplyMarkdown(commit))
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
        start, body_start, lang = opened
        before = tail[:start]
        self._committed_len += body_start
        self._fence = _FenceStream(lang=lang)
        if before.strip():
            self._print_body(ReplyMarkdown(before))
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
        """コード行を、Markdown のコードブロックと同じ描き方で出す。"""
        if last:
            code = "\n".join(lines).rstrip()
            lines = code.split("\n") if code else []
        if not lines:
            return
        context = fence.context[-_FENCE_CONTEXT_LINES:]
        self._print_body(
            code_block("\n".join(lines), fence.lang, head=not fence.printed, context=context)
        )
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

    def _turn_summary_line(self, delegated: TokenUsage | None, elapsed: float = 0.0) -> str | None:
        """ターン末尾の内訳 1 行（最後に合計時間）。"""
        tok = self._turn_tokens()
        total = (
            ui("cli.render.turn_total", shown=_format_elapsed(elapsed))
            if elapsed >= _ELAPSED_AFTER_S
            else ""
        )
        if tok <= 0 and self._requests == 0:
            return "— " + total if total else None
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
        if total:
            parts.append(total)
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
                stalled=(_LOAD_STALL_S, _load_stall_label(self._local_server)),
                pulse_bar=True,
            )
        label = self._phase_label or _state_label(self._state)
        if label is None:
            return None
        return _StatusLine(self._spinner, label, self._phase_since)

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
                fence = self._fence
                body.append(self._body(code_block(shown, fence.lang, head=not fence.printed)))
        elif tail.strip():
            body.append(self._body(ReplyMarkdown(tail)))
        panel = self._tool_arg_panel()
        if panel is not None:
            body.append(panel)
        parts: list[RenderableType] = []
        if body:
            parts.append(_TailLines(Group(*body), reserve=_LIVE_RESERVE_LINES))
        parts.extend(self._pending_rows())
        status = self._status_row()
        if status is not None:
            parts.extend((Text(), status))
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

    def _emit_tool(self, entry: _Running, dot: str, rows: list[RenderableType]) -> None:
        """ツールの行と、``⎿`` に続く結果の行を出す。"""
        self._open_result()
        label = entry.display.label
        packed = self._group == label and not rows
        header = _tool_header(entry.display, dot)
        # 1 回の print で出す（間で最下部の領域が描き直されないように）
        self._emit(Group(header, _result_gutter(rows)) if rows else header, separated=not packed)
        self._group = None if rows else label
        self._last_tool = entry.call.id

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
                self._print_body(ReplyMarkdown(tail))
        elif "".join(self._text).strip() or self._thinking:
            parts: list[RenderableType] = []
            if self._thinking:
                parts.append(Text("".join(self._thinking), style="dim italic"))
            if "".join(self._text).strip():
                parts.append(ReplyMarkdown("".join(self._text)))
            self._emit(Group(*parts))
        self._reset_segment()


def _done_line(started: float | None) -> str:
    """ツール完了行。"""
    elapsed = None if started is None else time.monotonic() - started
    if elapsed is None or elapsed < 0.1:
        return ui("cli.render.done")
    shown = f"{elapsed:.1f}s" if elapsed < 10 else f"{elapsed:.0f}s"
    return ui("cli.render.done_with_summary", shown=shown)


def _rough_tokens(ascii_chars: int, other_chars: int) -> int:
    """表示専用の粗いトークン概算。"""
    return ascii_chars // 4 + other_chars


def _format_elapsed(seconds: float) -> str:
    """経過時間を ``45s`` / ``1m12s`` / ``1h02m`` の形にする（秒は切り捨て）。"""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


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
    """``s`` の中で閉じていない列 0 の開きフェンスの ``(開始位置, 本体の開始位置, 言語名)``。"""
    offset = 0
    opened: tuple[int, int, str] | None = None
    for line in s.splitlines(keepends=True):
        bare = line.rstrip("\r\n")
        if opened is None:
            m = _FENCE_OPEN_LINE.match(bare)
            if m is not None:
                if not line.endswith("\n"):
                    return None
                opened = (offset, offset + len(line), m.group(1))
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


def _tool_header(display: ToolDisplay, dot: str) -> Text:
    """``⏺ 動作(対象)``（対象が 2 つ以上なら括弧は付けず、結果の行に並べる）。"""
    header = Text.assemble((_MARK, dot), (display.label, "bold"))
    if len(display.items) == 1:
        header.append(f"({display.items[0]})")
    return header


def _lead_rows(entry: _Running) -> list[RenderableType]:
    """ツールの行のすぐ下に置く行（2 つ以上の対象の一覧と、ファイル変更の差分）。"""
    items = entry.display.items
    rows: list[RenderableType] = [Text(item) for item in items] if len(items) > 1 else []
    if entry.preview is not None:
        rows.append(entry.preview)
    return rows


def _result_gutter(rows: list[RenderableType]) -> RenderableType:
    """結果の行を ``  ⎿  `` で始め、2 行目からは同じ幅だけ下げる。"""
    return _Gutter(Group(*rows), Text(_RESULT_MARK, style="dim"))


def _error_detail_lines(block: ToolResultBlock) -> list[Text]:
    """失敗理由の 2 行目以降。"""
    return [Text("  " + line, style="red dim") for line in error_detail(block)]


def _bash_output_text(output: str) -> Text:
    body = (
        output
        if len(output) <= _TOOL_RESULT_LIMIT
        else output[:_TOOL_RESULT_LIMIT] + "\n… (truncated)"
    )
    return Text(body.rstrip(), style="dim")


def collapsed_output_panel(entry: CollapsedOutput) -> RenderableType:
    """``/expand`` で出し直す全文（見出しは番号とコマンドの 1 行目）。"""
    first = entry.command.strip().split("\n", 1)[0]
    return Panel(
        Text(entry.output.rstrip(), style="dim"),
        title=Text(f"#{entry.number} $ {first}"),
        title_align="left",
        border_style="green",
        expand=False,
    )


def action_preview(
    call: ToolUse,
    required: str,
    detailed: bool = False,
    warning: str | None = None,
    *,
    redact: bool = False,
) -> RenderableType:
    """権限確認カード: 「何を実行するか」をツール別に整形する（``redact`` なら出す所を伏せる）。"""
    title = ui("cli.render.permission", name=call.name, required=required)
    if redact:
        body = _preview_body(_redacted_call(call), detailed, real=call, split=_shown_lines)
    else:
        body = _preview_body(call, detailed)
    if warning is not None:
        body = Group(Text(f"⚠ {warning}", style="yellow bold"), body)
    return Panel(body, title=title, border_style="yellow", expand=False)


def _redacted_call(call: ToolUse) -> ToolUse:
    """入力の秘密を、描く時点の鍵の値で伏せた写し。"""
    shown = redact_shown_input(call.name, call.input, values=secret_values())
    return call.model_copy(update={"input": shown})


def change_preview(
    call: ToolUse, cwd: Path | None = None, *, redact: bool = False
) -> RenderableType | None:
    """ファイル変更ツールの「これから何を変えるか」（``redact`` なら出す行だけを伏せる）。"""
    inp = call.input
    path = str(inp.get("path", ""))
    split = _shown_lines if redact else str.splitlines
    shown_path = redact_secrets(path, values=secret_values()) if redact else path
    if call.name == "edit_file":
        old, new = str(inp.get("old", "")), str(inp.get("new", ""))
        current = _read_current(cwd, path)
        if isinstance(current, str) and old and old in current:
            after = (
                current.replace(old, new)
                if inp.get("replace_all")
                else current.replace(old, new, 1)
            )
            return diff_view(current, after, path, max_rows=_DIFF_PREVIEW_LINES, split=split)
        if isinstance(current, str) and old:
            shifted = edit_match.reindent_edit(current, old, new)
            if shifted is not None:
                after = current.replace(shifted.old, shifted.new, 1)
                return diff_view(current, after, path, max_rows=_DIFF_PREVIEW_LINES, split=split)
        return _diff_preview(shown_path, old, new, False, split)
    if call.name == "delete_path":
        return _delete_preview(path, cwd)
    if call.name == "write_file":
        content = str(inp.get("content", ""))
        current = _read_current(cwd, path)
        if current is _MISSING:
            return new_file_view(content, path, max_rows=_NEW_FILE_PREVIEW_LINES, split=split)
        if isinstance(current, str):
            return diff_view(current, content, path, max_rows=_DIFF_PREVIEW_LINES, split=split)
        return _write_preview(shown_path, content, False, split)
    if call.name == "append_file":
        content = str(inp.get("content", ""))
        current = _read_current(cwd, path)
        if current is _MISSING:
            return new_file_view(content, path, max_rows=_NEW_FILE_PREVIEW_LINES, split=split)
        if isinstance(current, str):
            after = current + content
            return diff_view(current, after, path, max_rows=_DIFF_PREVIEW_LINES, split=split)
        return _write_preview(shown_path, content, False, split)
    return None


def _shown_lines(text: str) -> list[str]:
    """画面に出す行（行の数を変えずに秘密を伏せる）。"""
    return redact_lines(text, values=secret_values())


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


def _diff_stats(old: str, new: str) -> tuple[int, int]:
    """追加行数・削除行数（権限カードの要約用）。"""
    add = rm = 0
    for ln in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm=""):
        if ln.startswith("+") and not ln.startswith("+++"):
            add += 1
        elif ln.startswith("-") and not ln.startswith("---"):
            rm += 1
    return add, rm


def _preview_body(
    call: ToolUse,
    detailed: bool,
    *,
    real: ToolUse | None = None,
    split: Callable[[str], list[str]] = str.splitlines,
) -> RenderableType:
    """カードの本文（``real`` があれば、ファイルの変更の差分と行数は元の値で数える）。"""
    inp = call.input
    raw = (real or call).input
    if call.name == "bash":
        return Syntax(str(inp.get("command", "")), "bash", theme="ansi_dark", word_wrap=True)
    if call.name == "edit_file":
        path, old, new = (
            str(inp.get("path", "")),
            str(raw.get("old", "")),
            str(raw.get("new", "")),
        )
        if detailed:
            return _diff_preview(path, old, new, True, split)
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
        path, content = str(inp.get("path", "")), str(raw.get("content", ""))
        if detailed:
            return _write_preview(path, content, True, split)
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


def _diff_preview(
    path: str,
    old: str,
    new: str,
    detailed: bool,
    split: Callable[[str], list[str]] = str.splitlines,
) -> RenderableType:
    a, b = old.splitlines(), new.splitlines()
    lines = list(difflib.unified_diff(a, b, fromfile=path, tofile=path, lineterm=""))
    if split is not str.splitlines:
        lines = _shown_unified(lines, a, b, split(old), split(new))
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


def _shown_unified(
    lines: list[str], a: list[str], b: list[str], sa: list[str], sb: list[str]
) -> list[str]:
    """``unified_diff`` の本文の行を、伏せた行に置き換える（見出しと ``@@`` は残す）。"""
    out = lines[:2]
    for group in difflib.SequenceMatcher(None, a, b).get_grouped_opcodes(3):
        out.append(lines[len(out)])
        for tag, i1, i2, j1, j2 in group:
            if tag == "equal":
                out.extend(
                    " " + shown_context(a[i], sa[i], sb[j])
                    for i, j in zip(range(i1, i2), range(j1, j2), strict=True)
                )
                continue
            out.extend("-" + sa[i] for i in range(i1, i2))
            out.extend("+" + sb[j] for j in range(j1, j2))
    return out


def _lines_omitted(n: int) -> str:
    return ui("cli.render.lines_omitted_d_for_full", n=n)


def _write_preview(
    path: str, content: str, detailed: bool, split: Callable[[str], list[str]] = str.splitlines
) -> RenderableType:
    lines = split(content)
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

"""段の予算: ハーネスが測り、検算し、数字で差し戻す。"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from satsumaestro.engine.plan_steps import PlanStep
from satsumaestro.i18n.text import ui
from satsumaestro.util import chunking

DEFAULT_RATE_TOK_S = 20.0
MAX_SENDBACKS = 2
SENDBACK_LIMITS: dict[str, int] = {"over": MAX_SENDBACKS, "no_estimates": 1, "no_steps": 0}

BudgetSource = Literal["measured", "default", "pinned"]

WINDOW_REPLY_SHARE = 0.2


@dataclass(frozen=True)
class PlanBudget:
    """1 段（＝1 回の書き込み）の上限と、その導出。"""

    read_timeout_s: float
    rate_tok_s: float
    source: BudgetSource
    samples: int
    tokens: int
    lines: int
    window: int = 0
    speed_lines: int = 0

    def explain_en(self) -> str:
        """差し戻し文に入れる導出（モデル向け・英語）。"""
        if self.source == "pinned":
            return f"{self.lines} lines (pinned by settings: guidance.step_budget_lines)"
        rate = (
            f"{self.rate_tok_s:.0f} tokens/s measured in this conversation"
            if self.source == "measured"
            else f"{self.rate_tok_s:.0f} tokens/s (default; not yet measured in this conversation)"
        )
        if self.window:
            return (
                f"{self.window:,}-token context window × {WINDOW_REPLY_SHARE} (the room left for "
                f"one reply) × {chunking.STEP_BUDGET_SAFETY} ≈ {self.tokens:,} tokens ≈ "
                f"{self.lines} lines at ~{chunking.CHARS_PER_TOKEN} chars/token, "
                f"~{chunking.CHARS_PER_LINE} chars/line (smaller than the {self.speed_lines} "
                f"lines from {rate})"
            )
        return (
            f"{self.read_timeout_s:.0f} s × {rate} × {chunking.STEP_BUDGET_SAFETY} ≈ "
            f"{self.tokens:,} tokens ≈ {self.lines} lines at ~{chunking.CHARS_PER_TOKEN} "
            f"chars/token, ~{chunking.CHARS_PER_LINE} chars/line"
        )

    def explain_card(self) -> str:
        """承認カードに出す導出（ユーザー向け・画面の言語）。"""
        lines = self.lines
        if self.source == "pinned":
            return ui("engine.plan_budget.step_limit_lines_fixed_by", lines=lines)
        if self.window:
            return ui(
                "engine.plan_budget.step_limit_lines_window",
                lines=lines,
                window=f"{self.window:,}",
                share=WINDOW_REPLY_SHARE,
                safety=chunking.STEP_BUDGET_SAFETY,
                tokens=f"{self.tokens:,}",
                speed_lines=self.speed_lines,
            )
        rate = f"{self.rate_tok_s:.0f}"
        if self.source == "measured":
            rate_text = ui("engine.plan_budget.tok_s_measured_in_this", rate=rate)
        else:
            rate_text = ui("engine.plan_budget.default_tok_s_not_yet", rate=rate)
        seconds, safety, tokens = (
            f"{self.read_timeout_s:.0f}",
            chunking.STEP_BUDGET_SAFETY,
            f"{self.tokens:,}",
        )
        return ui(
            "engine.plan_budget.step_limit_lines_s_safety",
            lines=lines,
            rate_text=rate_text,
            seconds=seconds,
            safety=safety,
            tokens=tokens,
        )


def measured_rate(samples: Sequence[tuple[int, float]]) -> tuple[float, int] | None:
    """応答ごとの ``(出力トークン, 所要秒)`` から生成速度の下限（中央値）と標本数を返す。"""
    rates = [
        tokens / seconds
        for tokens, seconds in samples
        if tokens >= chunking.MIN_RATE_SAMPLE_TOKENS and seconds > 0
    ]
    if not rates:
        return None
    return statistics.median(rates), len(rates)


def lines_for(tokens: float) -> int:
    """トークン数を行数へ換算し、10 の倍数へ切り下げて上下限に収める。"""
    lines = tokens * chunking.CHARS_PER_TOKEN / chunking.CHARS_PER_LINE
    rounded = int(lines // 10) * 10
    return max(chunking.STEP_BUDGET_MIN_LINES, min(chunking.STEP_BUDGET_MAX_LINES, rounded))


def window_reply_tokens(window: int) -> int | None:
    """窓から決めた 1 回の書き込みの上限（トークン）。"""
    if window <= 0:
        return None
    return int(window * WINDOW_REPLY_SHARE * chunking.STEP_BUDGET_SAFETY)


def compute_plan_budget(
    read_timeout_s: float,
    samples: Sequence[tuple[int, float]],
    pin: int | None = None,
    window: int = 0,
) -> PlanBudget:
    """予算を出す。"""
    measured = measured_rate(samples)
    source: BudgetSource
    if measured is None:
        rate, count, source = DEFAULT_RATE_TOK_S, 0, "default"
    else:
        rate, count = measured
        source = "measured"
    tokens = int(read_timeout_s * rate * chunking.STEP_BUDGET_SAFETY)
    if pin is not None:
        return PlanBudget(read_timeout_s, rate, "pinned", count, tokens, pin)
    lines = lines_for(tokens)
    room = window_reply_tokens(window)
    if room is not None and lines_for(room) < lines:
        return PlanBudget(read_timeout_s, rate, source, count, room, lines_for(room), window, lines)
    return PlanBudget(read_timeout_s, rate, source, count, tokens, lines)


VerdictKind = Literal["over", "no_estimates", "no_steps"]


@dataclass(frozen=True)
class BudgetVerdict:
    """検算に落ちた理由。"""

    kind: VerdictKind
    budget: PlanBudget
    over: tuple[tuple[int, int], ...] = ()

    def note(self, plan_file: str) -> str:
        limit = f"{self.budget.lines} lines ({self.budget.explain_en()})"
        split = (
            f"into steps of at most {self.budget.lines} lines each — write_file for the first "
            "part, append_file for the rest (write_file replaces the whole file; once a file "
            "exists, add to it with append_file) —"
        )
        if self.kind == "over":
            if len(self.over) == 1:
                index, estimate = self.over[0]
                head = (
                    f"Step {index} is estimated at {estimate} lines; the limit per step is "
                    f"{limit}. Split it {split}"
                )
            else:
                listed = " and ".join(f"{i} ({n} lines)" for i, n in self.over)
                head = (
                    f"Steps {listed} exceed the limit per step of {limit}. Split each of them "
                    f"{split}"
                )
            return (
                f"{head} keep the (~N lines) estimate on every step, update `{plan_file}`, "
                "then call exit_plan_mode again."
            )
        if self.kind == "no_estimates":
            return (
                "The plan's steps carry no size estimates, so they cannot be checked against "
                f"the limit per step of {limit}. End every step with the new content it creates "
                "as (~N lines) — (~0 lines) for steps that only read, run or verify — split any "
                f"step over {self.budget.lines} lines (write_file for the first part, append_file "
                "for the rest; write_file replaces the whole file, so once a file exists, add to "
                f"it with append_file), update `{plan_file}`, then call exit_plan_mode again."
            )
        return (
            'The plan has no "## Steps" section, so it cannot be handed back one step at a time '
            f'or checked against the limit per step of {limit}. Add a "## Steps" section: a '
            "numbered list of the work you will do with tools after approval, each step ending "
            "with the new content it creates as (~N lines) (~0 lines for steps that only read, "
            f"run or verify), no step over {self.budget.lines} lines. Update `{plan_file}`, then "
            "call exit_plan_mode again."
        )

    @property
    def sendback_limit(self) -> int:
        """この種別を 1 ターンで差し戻せる回数（0 = 記録に留める）。"""
        return SENDBACK_LIMITS[self.kind]

    @property
    def sends_back(self) -> bool:
        """差し戻す種別か（節無しは記録に留める）。"""
        return self.sendback_limit > 0

    def card(self) -> str:
        """承認カードに添える 1 行（差し戻さなかった／差し戻し上限に達した理由。画面の言語）。"""
        limit = self.budget.lines
        if self.kind == "over":
            summary = self.summary()
            return ui("engine.plan_budget.and_the_send_back_limit", summary=summary)
        if self.kind == "no_estimates":
            return ui("engine.plan_budget.the_steps_still_have_no", limit=limit)
        return ui("engine.plan_budget.no_steps_section_card", limit=limit)

    def summary(self) -> str:
        """REPL の差し戻しの 1 行（画面の言語）。"""
        limit = self.budget.lines
        if self.kind == "over":
            if len(self.over) == 1:
                index, estimate = self.over[0]
                return ui(
                    "engine.plan_budget.step_is_over_budget_lines",
                    index=index,
                    estimate=estimate,
                    limit=limit,
                )
            steps = ui("engine.plan_budget.step_separator").join(str(i) for i, _ in self.over)
            sizes = ui("engine.plan_budget.size_separator").join(
                ui("engine.plan_budget.size", n=n) for _, n in self.over
            )
            return ui(
                "engine.plan_budget.steps_are_over_budget_lines",
                steps=steps,
                sizes=sizes,
                limit=limit,
            )
        if self.kind == "no_estimates":
            return ui("engine.plan_budget.the_steps_have_no_estimate", limit=limit)
        return ui("engine.plan_budget.no_steps_section", limit=limit)


def check_plan_budget(steps: Sequence[PlanStep], budget: PlanBudget) -> BudgetVerdict | None:
    """計画の段を予算と検算する。"""
    if not steps:
        return BudgetVerdict("no_steps", budget)
    if all(s.estimated_lines is None for s in steps):
        return BudgetVerdict("no_estimates", budget)
    over = tuple(
        (i, s.estimated_lines)
        for i, s in enumerate(steps, 1)
        if s.estimated_lines is not None and s.estimated_lines > budget.lines
    )
    if over:
        return BudgetVerdict("over", budget, over)
    return None

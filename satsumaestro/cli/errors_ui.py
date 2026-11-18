"""ターンの失敗をユーザー向けの言葉にする。"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from satsumaestro.i18n.text import ui
from satsumaestro.providers.choice import CHOICE_LABELS
from satsumaestro.providers.failures import account_failure
from satsumaestro.providers.key_check import KeyCheck
from satsumaestro.providers.retry import is_read_timeout, is_tool_call_parse_error
from satsumaestro.util.errors import (
    ContextOverflowError,
    ContextTruncatedError,
    MaxIterationsError,
    PlanExitLoopError,
    ProviderError,
    RepetitionLoopError,
    RetriesExhaustedError,
    SatsumaError,
    UnknownToolLoopError,
)

_log = logging.getLogger(__name__)


def failure_head() -> str:
    """失敗の 1 行目の見出し（煙テストの転写が失敗行として数える）。"""
    return ui("cli.errors_ui.the_turn_could_not_be")


def resume_hint() -> str:
    return ui("cli.errors_ui.resume_hint")


def broken_output_hint() -> str:
    return ui("cli.errors_ui.broken_output_hint")


@dataclass(frozen=True)
class TurnFailure:
    """落ちた理由（1 行目の「: 」の後）と、次の一手（2 行目）。"""

    reason: str
    hint: str = ""
    broken_output: bool = False
    reason_only: bool = False

    @property
    def headline(self) -> str:
        return f"{failure_head()}: {self.reason}"

    @property
    def next_step(self) -> str:
        if self.reason_only:
            return ""
        if self.hint:
            return self.hint
        return broken_output_hint() if self.broken_output else resume_hint()


def turn_failure(exc: SatsumaError) -> TurnFailure:
    """ターンを終えた例外を、ユーザー向けの理由と次の一手にする（元の文言はログへ）。"""
    _log.warning("turn failed: %s", exc)
    if isinstance(exc, MaxIterationsError):
        limit = exc.limit
        if exc.step is not None:
            index, total = exc.step
            return TurnFailure(
                ui(
                    "cli.errors_ui.reached_the_iteration_limit_inside",
                    limit=limit,
                    index=index,
                    total=total,
                )
            )
        return TurnFailure(ui("cli.errors_ui.reached_the_iteration_limit", limit=limit))
    if isinstance(exc, RetriesExhaustedError) and is_tool_call_parse_error(exc):
        return TurnFailure(
            ui("cli.errors_ui.the_model_s_output_tool"),
            broken_output=True,
        )
    if isinstance(exc, ProviderError) and is_read_timeout(exc):
        t = f"{exc.timeout_s:.0f}"
        timed = exc.timeout_s > 0
        retried = exc.timeout_recoveries > 0
        if timed and retried:
            reason = ui("cli.errors_ui.no_response_for_s_after_resend", seconds=t)
        elif timed:
            reason = ui("cli.errors_ui.no_response_for_s", seconds=t)
        elif retried:
            reason = ui("cli.errors_ui.no_response_after_resend")
        else:
            reason = ui("cli.errors_ui.no_response")
        return TurnFailure(reason)
    if isinstance(exc, ProviderError):
        refused = account_turn_failure(exc)
        if refused is not None:
            return refused
        return TurnFailure(ui("cli.errors_ui.communication_with_the_model_failed", exc=exc))
    if isinstance(exc, PlanExitLoopError):
        n = exc.calls
        return TurnFailure(ui("cli.errors_ui.exit_plan_mode_was_called", n=n))
    if isinstance(exc, UnknownToolLoopError):
        name, n = exc.name, exc.calls
        return TurnFailure(ui("cli.errors_ui.the_nonexistent_tool_was_called", name=name, n=n))
    if isinstance(exc, ContextOverflowError):
        est, cap = f"{exc.estimate:,}", f"{exc.limit:,}"
        sizes = ui("cli.errors_ui.estimated_about_limit_about_tokens", est=est, cap=cap)
        if exc.dropped_request:
            return TurnFailure(
                ui("cli.errors_ui.the_request_does_not_fit", sizes=sizes),
                ui("cli.errors_ui.the_request_was_not_sent"),
            )
        return TurnFailure(
            ui("cli.errors_ui.the_request_to_the_model", sizes=sizes),
            ui("cli.errors_ui.fold_the_conversation_with_compact"),
        )
    if isinstance(exc, ContextTruncatedError):
        sent, read = f"{exc.sent:,}", f"{exc.read:,}"
        return TurnFailure(
            ui("cli.errors_ui.the_server_truncated_the_request", sent=sent, read=read),
            ui("cli.errors_ui.the_reply_was_not_used"),
        )
    if isinstance(exc, RepetitionLoopError):
        n = exc.repeats
        return TurnFailure(ui("cli.errors_ui.the_same_content_kept_repeating", n=n))
    return TurnFailure(str(exc))


def account_turn_failure(exc: BaseException) -> TurnFailure | None:
    """残高切れ・利用の上限・鍵の拒否を、理由だけの失敗にする（ほかは ``None``）。"""
    failure = account_failure(exc)
    if failure is None:
        return None
    vendor = CHOICE_LABELS.get(failure.vendor) or failure.host or "API"
    status, message = failure.status, failure.message
    if failure.kind == "no_credits":
        reason = ui("cli.errors_ui.no_credits", vendor=vendor, status=status, message=message)
    elif failure.kind == "rate_limited":
        reason = ui("cli.errors_ui.rate_limited", vendor=vendor, status=status, message=message)
    else:
        reason = ui("cli.errors_ui.key_rejected", vendor=vendor, status=status, message=message)
    return TurnFailure(reason, reason_only=True)


def rejected_key_text(check: KeyCheck) -> str:
    """確かめた鍵が断られた理由（止められた xAI の鍵は、止まっている項目の名前）。"""
    vendor = CHOICE_LABELS.get(check.choice, check.choice)
    if check.status == 200:
        return ui("cli.errors_ui.key_blocked", vendor=vendor, flags=check.message)
    status, message = check.status, check.message
    return ui("cli.errors_ui.key_rejected", vendor=vendor, status=status, message=message)

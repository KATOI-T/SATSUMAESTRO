"""プロバイダ層の出来事をユーザー向けの 1 行として画面へ届ける経路。"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal

from satsumaestro.i18n.text import ui

NoticeKind = Literal[
    "retry", "fallback", "rewrite", "tools_dropped", "effort_forced", "effort_adjusted"
]
RetryReason = Literal["tool_parse", "transient"]


@dataclass(frozen=True)
class ProviderNotice:
    kind: NoticeKind
    attempt: int = 0
    limit: int = 0
    delay_s: float = 0.0
    reason: RetryReason = "transient"
    from_model: str = ""
    to_model: str = ""
    model: str = ""
    requested: str = ""
    sent: str = ""


NoticeSink = Callable[[ProviderNotice], None]
_sink: ContextVar[NoticeSink | None] = ContextVar("satsuma_provider_notice_sink", default=None)


def notify(notice: ProviderNotice) -> None:
    """通知先が張られていれば渡す（無ければ何もしない。描画の失敗でプロバイダを止めない）。"""
    sink = _sink.get()
    if sink is None:
        return
    try:
        sink(notice)
    except Exception:  # noqa: BLE001
        return


@contextmanager
def bind(sink: NoticeSink) -> Iterator[None]:
    """この区間でプロバイダから届く通知を ``sink`` へ渡す。"""
    token = _sink.set(sink)
    try:
        yield
    finally:
        _sink.reset(token)


def describe(notice: ProviderNotice) -> str:
    """ユーザー向けの 1 行（先頭の「— 」は描画側が付ける）。"""
    if notice.kind == "fallback":
        old, new = notice.from_model, notice.to_model
        return ui("providers.notices.is_not_responding_so_switching", old=old, new=new)
    if notice.kind == "rewrite":
        return ui("providers.notices.the_model_s_output_kept")
    if notice.kind == "tools_dropped":
        return ui("providers.notices.tools_dropped_for_model", model=notice.model)
    if notice.kind == "effort_forced":
        return ui("providers.notices.effort_forced_to_none", model=notice.model)
    if notice.kind == "effort_adjusted":
        if notice.sent:
            return ui(
                "providers.notices.effort_replaced_for_model",
                model=notice.model,
                requested=notice.requested,
                sent=notice.sent,
            )
        return ui(
            "providers.notices.effort_unsupported_by_model",
            model=notice.model,
            requested=notice.requested,
        )
    count = f"{notice.attempt}/{notice.limit}"
    if notice.reason == "tool_parse":
        if notice.limit:
            return ui("providers.notices.output_broke_retry_count", count=count)
        return ui("providers.notices.output_broke_retry")
    delay = f"{notice.delay_s:.0f}"
    if notice.limit:
        return ui("providers.notices.reply_failed_retry_count", delay=delay, count=count)
    return ui("providers.notices.reply_failed_retry", delay=delay)

"""段の予算に、モデルの窓から決めた上限を掛ける。"""

from __future__ import annotations

from pathlib import Path

from satsumaestro.config.models import CompactionConfig, HookConfig, RuntimeConfig
from satsumaestro.engine.compaction import Compactor
from satsumaestro.engine.context_fit import CONTEXT_FIT_RATIO
from satsumaestro.engine.plan_budget import (
    WINDOW_REPLY_SHARE,
    compute_plan_budget,
    window_reply_tokens,
)
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.i18n.text import pinned_ui_language
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from satsumaestro.util import chunking
from tests.fixtures.mock_provider import SequencedMockProvider


def _sample(rate: float, tokens: int = 1000) -> tuple[int, float]:
    return tokens, tokens / rate


def test_the_reply_share_is_what_compaction_leaves() -> None:
    assert WINDOW_REPLY_SHARE == round(1 - CONTEXT_FIT_RATIO, 6)


def test_a_small_window_caps_the_speed_budget() -> None:
    budget = compute_plan_budget(120.0, [_sample(72.0)], window=16_384)
    assert (budget.tokens, budget.lines, budget.window, budget.speed_lines) == (
        1_638,
        130,
        16_384,
        360,
    )
    assert budget.source == "measured"
    assert budget.explain_card() == (
        "1 段の上限 130 行 ＝ 窓 16,384 トークン × 0.2 × 安全率 0.5"
        "（≈ 1,638 トークン。速さからの 360 行より小さい）"
    )
    assert budget.explain_en() == (
        "16,384-token context window × 0.2 (the room left for one reply) × 0.5 ≈ 1,638 tokens "
        "≈ 130 lines at ~2.5 chars/token, ~30 chars/line (smaller than the 360 lines from "
        "72 tokens/s measured in this conversation)"
    )
    with pinned_ui_language("en"):
        assert budget.explain_card() == (
            "step limit 130 lines = context window 16,384 tokens × 0.2 × safety factor 0.5 "
            "(≈ 1,638 tokens; smaller than the 360 lines from the speed)"
        )


def test_an_8k_window_falls_to_the_minimum_lines() -> None:
    budget = compute_plan_budget(120.0, [_sample(34.0)], window=8_192)
    assert budget.window == 8_192
    assert budget.lines == chunking.STEP_BUDGET_MIN_LINES


def test_large_and_unknown_windows_keep_the_speed_budget() -> None:
    plain = compute_plan_budget(120.0, [_sample(57.5)])
    for window in (131_072, 0):
        budget = compute_plan_budget(120.0, [_sample(57.5)], window=window)
        assert budget == plain
        assert budget.window == 0
        assert "1 段の上限 280 行 ＝ この会話で計測した 58 tok/s" in budget.explain_card()
    assert window_reply_tokens(0) is None


def test_the_pin_wins_over_the_window() -> None:
    budget = compute_plan_budget(120.0, [_sample(72.0)], pin=200, window=8_192)
    assert (budget.source, budget.lines, budget.window) == ("pinned", 200, 0)


def test_the_runtime_reads_the_window_when_it_computes_the_budget(tmp_path: Path) -> None:
    rt = ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=SequencedMockProvider([[]]),  # type: ignore[arg-type]
        tools=default_registry(),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=RuntimeConfig(compaction=CompactionConfig(context_window=16_384)),
        compactor=Compactor(),
    )
    rt._rate_samples.append(_sample(72.0))
    assert rt._plan_budget().lines == 130
    rt.config.compaction.context_window = 131_072
    assert rt._plan_budget().lines == 360

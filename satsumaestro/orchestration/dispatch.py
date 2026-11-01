"""単発／並列の振り分け判定。"""

from __future__ import annotations

import math

from satsumaestro.config.models import DecompositionConfig
from satsumaestro.orchestration.plan import (
    DispatchDecision,
    DispatchMode,
    ValidatedPlan,
)


class Dispatcher:
    """振り分けの規則で :class:`DispatchDecision` を返す。"""

    def __init__(self, cfg: DecompositionConfig, *, capacity: int) -> None:
        self._cfg = cfg
        self._capacity = max(1, capacity)

    def decide(self, plan: ValidatedPlan) -> DispatchDecision:
        cfg = self._cfg
        capacity = self._capacity
        subtasks = {s.id: s for s in plan.plan.subtasks}
        serial_cost = sum(s.est_calls for s in subtasks.values())

        width = max(
            (sum(1 for i in level if i in plan.background) for level in plan.levels),
            default=0,
        )
        if capacity < 2:
            return DispatchDecision(
                mode=DispatchMode.SINGLE,
                reason=f"capacity={capacity}",
                serial_cost=serial_cost,
                capacity=capacity,
                width=width,
            )
        if width < cfg.min_width:
            return DispatchDecision(
                mode=DispatchMode.SINGLE,
                reason=f"width={width}",
                serial_cost=serial_cost,
                capacity=capacity,
                width=width,
            )

        parallel_cost = cfg.overhead_calls
        for level in plan.levels:
            waves = math.ceil(len(level) / capacity)
            parallel_cost += waves * max(subtasks[i].est_calls for i in level)
        if parallel_cost * (1.0 + cfg.gain_margin) >= serial_cost:
            return DispatchDecision(
                mode=DispatchMode.SINGLE,
                reason=f"no_gain(serial={serial_cost},parallel={parallel_cost})",
                serial_cost=serial_cost,
                parallel_cost=parallel_cost,
                capacity=capacity,
                width=width,
            )

        return DispatchDecision(
            mode=DispatchMode.PARALLEL,
            reason=f"parallel(serial={serial_cost},parallel={parallel_cost},width={width})",
            serial_cost=serial_cost,
            parallel_cost=parallel_cost,
            capacity=capacity,
            width=width,
        )

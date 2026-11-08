"""自動タスク分解の前段フックの具象 ``DecompositionCoordinator``。"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from pathlib import Path

from satsumaestro.config.models import RuntimeConfig
from satsumaestro.engine.tool_call_repair import ToolCallRepairPolicy
from satsumaestro.orchestration.decompose import Decomposer, static_gate
from satsumaestro.orchestration.dispatch import Dispatcher
from satsumaestro.orchestration.executor import PlanExecutor
from satsumaestro.orchestration.ids import make_id
from satsumaestro.orchestration.plan import (
    DecompositionRecord,
    DispatchDecision,
    DispatchMode,
    TaskPlan,
    ValidatedPlan,
    validate_plan,
)
from satsumaestro.orchestration.runner import Orchestrator
from satsumaestro.providers.base import Provider
from satsumaestro.providers.capacity import estimate_parallelism, probe_parallelism
from satsumaestro.providers.factory import (
    openai_endpoint,
    resolve_role_model,
    resolve_role_sampling,
)
from satsumaestro.providers.local import LlamaCppServer, server_for_endpoint
from satsumaestro.session.models import Session, TodoItem
from satsumaestro.util.errors import OrchestrationError

_log = logging.getLogger(__name__)


class _CircuitBreaker:
    """推定より遅い並列が連続したらセッション内で分解を止める。"""

    def __init__(self, losses_to_open: int) -> None:
        self._limit = losses_to_open
        self._losses = 0

    @property
    def open(self) -> bool:
        return self._losses >= self._limit

    def note(self, *, wall_ms: int, serial_estimate_ms: float) -> None:
        if wall_ms > serial_estimate_ms:
            self._losses += 1
        else:
            self._losses = 0


class DecompositionCoordinator:
    """分解の判定と実行を束ねる前段フック（engine の ``DecompositionHook`` 実装）。"""

    def __init__(
        self,
        *,
        orchestrator: Orchestrator,
        config: RuntimeConfig,
        provider: Provider,
        model: str,
        session: Session,
        cwd: Path,
        env: Mapping[str, str],
        provider_builder: Callable[[str], Provider],
    ) -> None:
        self._orchestrator = orchestrator
        self._config = config
        self._provider = provider
        self._model = model
        self._session = session
        self._cwd = cwd
        self._env = env
        self._provider_builder = provider_builder
        self._breaker = _CircuitBreaker(config.decomposition.circuit_breaker_losses)
        self._capacity_value: int | None = None
        self._record: DecompositionRecord | None = None
        self._counter = 0

    async def before_turn(self, user_text: str) -> str | None:
        dcfg = self._config.decomposition
        gate = static_gate(
            user_text,
            dcfg,
            orchestration_enabled=self._config.orchestration.enabled,
            plan_mode=self._session.mode.main == "plan",
            breaker_open=self._breaker.open,
        )
        if gate.single:
            self._single(gate.reason)
            return None
        capacity = await self._capacity()
        if capacity < 2:
            self._single(f"capacity={capacity}", capacity=capacity)
            return None
        plan = await self._decompose(user_text, max_subtasks=dcfg.max_subtasks)
        if plan is None:
            self._single("decompose:none", capacity=capacity)
            return None
        if plan.single or not plan.subtasks:
            self._single("decompose:single", capacity=capacity, plan=plan)
            return None
        try:
            validated = validate_plan(
                plan, max_subtasks=dcfg.max_subtasks, max_depth=dcfg.max_depth
            )
        except OrchestrationError as exc:
            self._single(f"invalid_plan:{exc}"[:120], capacity=capacity, plan=plan)
            return None
        decision = Dispatcher(dcfg, capacity=capacity).decide(validated)
        if decision.mode is DispatchMode.SINGLE:
            self._record = DecompositionRecord(decision=decision, plan=plan)
            return None

        self._counter += 1
        plan_id = make_id("plan", self._counter)
        self._orchestrator.record_plan(plan_id, {"decision": decision.model_dump(mode="json")})
        self._show_todos(validated)
        executor = PlanExecutor(self._orchestrator, dcfg, capacity=capacity)
        try:
            result = await executor.run(validated, plan_id=plan_id)
        finally:
            self._complete_todos()
        self._breaker.note(
            wall_ms=result.wall_ms,
            serial_estimate_ms=decision.serial_cost * dcfg.per_call_budget_s * 1000 / 4,
        )
        self._record = DecompositionRecord(decision=decision, plan=plan, result=result)
        _log.info(
            "decomposition fired: %s wall=%dms fallback=%s",
            decision.reason,
            result.wall_ms,
            result.fallback_ids,
        )
        return executor.build_context(user_text, validated, result)

    def take_record(self) -> DecompositionRecord | None:
        record, self._record = self._record, None
        return record

    def _single(self, reason: str, *, capacity: int = 1, plan: TaskPlan | None = None) -> None:
        self._record = DecompositionRecord(
            decision=DispatchDecision(mode=DispatchMode.SINGLE, reason=reason, capacity=capacity),
            plan=plan,
        )

    async def _decompose(self, user_text: str, *, max_subtasks: int) -> TaskPlan | None:
        role_model = resolve_role_model(self._config, "decompose", default=self._model)
        provider = (
            self._provider if role_model == self._model else self._provider_builder(role_model)
        )
        decomposer = Decomposer(
            provider=provider,
            model=role_model,
            policy=ToolCallRepairPolicy.from_config(self._config),
            sampling=resolve_role_sampling(self._config, "decompose"),
        )
        return await decomposer.decompose(user_text, max_subtasks=max_subtasks)

    async def _capacity(self) -> int:
        if self._capacity_value is not None:
            return self._capacity_value
        dcfg = self._config.decomposition
        base = openai_endpoint(self._config, self._env) or ""
        server = server_for_endpoint(base, self._env, self._config.provider) if base else None
        is_ollama = server is not None and server.kind == "ollama"
        slots = (
            await server.slots()
            if isinstance(server, LlamaCppServer) and dcfg.provider_parallelism == "auto"
            else None
        )
        calibrate = dcfg.calibrate_parallelism
        provider_side = await estimate_parallelism(
            configured=dcfg.provider_parallelism,
            is_ollama=is_ollama,
            slots=slots,
            env=self._env,
            cache_path=(self._cwd / dcfg.capacity_cache_file) if calibrate else None,
            cache_key=f"{base}+{self._model}",
            probe=(
                lambda: probe_parallelism(
                    self._provider,
                    self._model,
                    limit=self._config.orchestration.max_concurrency,
                )
            )
            if calibrate
            else None,
        )
        ceiling = self._config.orchestration.max_concurrency
        value = ceiling if provider_side is None else min(ceiling, provider_side)
        self._capacity_value = max(1, value)
        return self._capacity_value

    def _show_todos(self, validated: ValidatedPlan) -> None:
        """段ごとのサブタスクを可視化する（TODO は複数 in_progress を許容）。"""
        subtasks = {s.id: s for s in validated.plan.subtasks}
        self._session.todos = [
            TodoItem(
                content=f"{sid}: {subtasks[sid].objective}"[:80],
                active_form=f"Researching {sid}",
                status="in_progress",
            )
            for level in validated.levels
            for sid in level
            if sid in validated.background
        ]

    def _complete_todos(self) -> None:
        for todo in self._session.todos:
            todo.status = "completed"

"""実行をまたいで過去の結果を再利用しない。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from satsumaestro.config.models import DecompositionConfig, RuntimeConfig
from satsumaestro.evals.agents import AgentAnswer, EvalAgent
from satsumaestro.evals.models import EvalTask
from satsumaestro.evals.runner import EvalRunner
from satsumaestro.providers.detect import (
    anthropic_model_info_for,
    capabilities_for,
    model_quirks_for,
    reset_capabilities,
)
from satsumaestro.session.store import new_session


def test_calibration_is_off_by_default() -> None:
    assert DecompositionConfig().calibrate_parallelism is False
    assert RuntimeConfig().decomposition.calibrate_parallelism is False
    assert DecompositionConfig().capacity_cache_file == ".satsuma/capacity.json"


async def _capacity(tmp_path: Path, *, calibrate: bool) -> tuple[int, int, bool]:
    """結線側と同じ組み立てで推定し、(値, プローブ回数, キャッシュの有無) を返す。"""
    from satsumaestro.providers.capacity import estimate_parallelism

    cfg = DecompositionConfig(calibrate_parallelism=calibrate)
    cache = tmp_path / cfg.capacity_cache_file
    probe_calls = 0

    async def probe() -> int:
        nonlocal probe_calls
        probe_calls += 1
        return 4

    value = await estimate_parallelism(
        configured=cfg.provider_parallelism,
        is_ollama=True,
        env={},
        cache_path=cache if calibrate else None,
        cache_key="http://localhost:11434/v1+m",
        probe=probe if calibrate else None,
    )
    return value or 0, probe_calls, cache.exists()


async def test_default_does_not_probe_or_write_the_cache(tmp_path: Path) -> None:
    value, probes, cached = await _capacity(tmp_path, calibrate=False)
    assert (value, probes, cached) == (1, 0, False)


async def test_enabling_it_restores_the_old_behaviour(tmp_path: Path) -> None:
    value, probes, cached = await _capacity(tmp_path, calibrate=True)
    assert (value, probes, cached) == (4, 1, True)
    saved = json.loads((tmp_path / ".satsuma/capacity.json").read_text(encoding="utf-8"))
    assert saved["http://localhost:11434/v1+m"]["parallelism"] == 4


async def test_the_hook_passes_neither_cache_nor_probe_when_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """結線側（``orchestration/hook.py``）が実際に外していることを見る。"""
    import satsumaestro.orchestration.hook as hook_module

    seen: dict[str, Any] = {}

    async def fake_estimate(**kwargs: Any) -> int:
        seen.update(kwargs)
        return 1

    monkeypatch.setattr(hook_module, "estimate_parallelism", fake_estimate)

    cfg = RuntimeConfig()
    assert cfg.decomposition.calibrate_parallelism is False
    hook = hook_module.DecompositionCoordinator(
        orchestrator=object(),  # type: ignore[arg-type]
        config=cfg,
        provider=object(),  # type: ignore[arg-type]
        model="m",
        session=new_session(tmp_path, "m"),
        cwd=tmp_path,
        env={},
        provider_builder=lambda _model: object(),  # type: ignore[arg-type,misc]
    )
    await hook._capacity()
    assert seen["cache_path"] is None and seen["probe"] is None


class _Recording(EvalAgent):
    """タスク実行時に capability の記録を汚すだけのエージェント。"""

    name = "recording"
    model = "mock"

    def __init__(self) -> None:
        self.seen_before: list[bool] = []

    async def run(
        self,
        prompt: str,
        workspace: Path,
        max_iterations: int | None = None,
        *,
        permission_mode: str = "read-only",
    ) -> AgentAnswer:
        caps = capabilities_for("http://localhost:11434/v1")
        self.seen_before.append(caps.strict_tools_unsupported)
        caps.mark_strict_unsupported()
        return AgentAnswer(text="ok")


async def test_eval_clears_the_capability_memory_between_tasks(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    tasks = [
        EvalTask.model_validate({"id": f"t{i}", "prompt": "hi", "fixture": str(fixture)})
        for i in range(3)
    ]
    agent = _Recording()
    await EvalRunner(agent).run(tasks)
    assert agent.seen_before == [False, False, False]
    reset_capabilities()


def test_reset_also_clears_the_model_quirks() -> None:
    model_quirks_for("http://x/v1", "m").tools_unsupported = True
    anthropic_model_info_for("https://api.anthropic.com", "m").notice_shown = True
    reset_capabilities()
    assert model_quirks_for("http://x/v1", "m").tools_unsupported is False
    assert anthropic_model_info_for("https://api.anthropic.com", "m").notice_shown is False

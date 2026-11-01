"""役割別モデルルーティングのテスト。"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from satsumaestro.config.models import (
    HookConfig,
    ModelRoles,
    ProviderConfig,
    RuntimeConfig,
    SamplingConfig,
    SubagentConfig,
)
from satsumaestro.engine.compaction import Compactor
from satsumaestro.engine.events import MessageStop, TextDelta
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.engine.subagent import RuntimeFactory, resolve_subagent_type
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.providers.factory import resolve_role_model, resolve_role_sampling
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from satsumaestro.util.errors import ProviderError
from tests.fixtures.mock_provider import MockProvider, ScriptedFailureProvider


def test_resolve_role_model_falls_back_to_default() -> None:
    cfg = RuntimeConfig()
    assert resolve_role_model(cfg, "summarize", default="main-model") == "main-model"
    assert resolve_role_model(cfg, "repair", default="main-model") == "main-model"
    assert resolve_role_model(cfg, "subagent", default="main-model") == "main-model"


def test_resolve_role_model_prefers_configured_role() -> None:
    cfg = RuntimeConfig(
        provider=ProviderConfig(roles=ModelRoles(summarize="small-model", repair="fix-model"))
    )
    assert resolve_role_model(cfg, "summarize", default="main-model") == "small-model"
    assert resolve_role_model(cfg, "repair", default="main-model") == "fix-model"
    assert resolve_role_model(cfg, "subagent", default="main-model") == "main-model"


def test_resolve_role_sampling_override_replaces_whole_config() -> None:
    base = SamplingConfig(temperature=0.7, top_p=0.9)
    override = SamplingConfig(temperature=0.1)
    cfg = RuntimeConfig(
        provider=ProviderConfig(
            sampling=base, roles=ModelRoles(sampling_overrides={"repair": override})
        )
    )
    assert resolve_role_sampling(cfg, "repair") == override
    assert resolve_role_sampling(cfg, "repair").top_p is None
    assert resolve_role_sampling(cfg, "summarize") == base


def test_model_roles_rejects_unknown_keys() -> None:
    with pytest.raises(ValidationError, match="unknown role"):
        ModelRoles(sampling_overrides={"translate": SamplingConfig()})
    with pytest.raises(ValidationError):
        ModelRoles.model_validate({"summarise": "x"})


def test_compactor_from_config_accepts_role_sampling() -> None:
    cfg = RuntimeConfig(provider=ProviderConfig(sampling=SamplingConfig(temperature=0.8)))
    override = SamplingConfig(temperature=0.2)
    compactor = Compactor.from_config(cfg, sampling=override)
    assert compactor._sampling == override
    assert Compactor.from_config(cfg)._sampling == cfg.provider.sampling


def _make_runtime(provider: object, tmp_path: Path, cfg: RuntimeConfig) -> ConversationRuntime:
    return ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=provider,  # type: ignore[arg-type]
        tools=default_registry(),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=cfg,
    )


def _parse_failure_provider() -> ScriptedFailureProvider:
    return ScriptedFailureProvider(
        [
            ProviderError("error parsing tool call", status_code=500),
            [TextDelta(text="ok"), MessageStop()],
        ]
    )


async def test_repair_retry_uses_repair_role(tmp_path: Path) -> None:
    cfg = RuntimeConfig(
        provider=ProviderConfig(
            roles=ModelRoles(
                repair="repair-model",
                sampling_overrides={"repair": SamplingConfig(temperature=0.0)},
            )
        )
    )
    provider = _parse_failure_provider()
    summary = await _make_runtime(provider, tmp_path, cfg).run_turn("hi")
    assert summary.final_text == "ok" and summary.tool_call_repairs == 1
    assert provider.requests[0].model == "mock-model"
    assert provider.requests[1].model == "repair-model"
    sampling = provider.requests[1].sampling
    assert sampling is not None and sampling.temperature == 0.0


async def test_repair_retry_defaults_to_main_model(tmp_path: Path) -> None:
    provider = _parse_failure_provider()
    await _make_runtime(provider, tmp_path, RuntimeConfig()).run_turn("hi")
    assert [r.model for r in provider.requests] == ["mock-model", "mock-model"]


def _factory(cfg: RuntimeConfig, tmp_path: Path) -> tuple[RuntimeFactory, list[str]]:
    built: list[str] = []

    def builder(model: str) -> MockProvider:
        built.append(model)
        return MockProvider([TextDelta(text="child"), MessageStop()])

    factory = RuntimeFactory(
        provider=MockProvider([TextDelta(text="parent"), MessageStop()]),
        model="parent-model",
        config=cfg,
        tools=default_registry(),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        hooks=HookRunner(HookConfig(), tmp_path),
        cwd=tmp_path,
        provider_builder=builder,
    )
    return factory, built


def test_build_child_uses_subagent_role(tmp_path: Path) -> None:
    cfg = RuntimeConfig(provider=ProviderConfig(roles=ModelRoles(subagent="child-model")))
    factory, built = _factory(cfg, tmp_path)
    child = factory.build_child(preset=resolve_subagent_type("Explore"), depth=1)
    assert child.session.model == "child-model"
    assert built == ["child-model"]


def test_build_child_role_precedence(tmp_path: Path) -> None:
    cfg = RuntimeConfig(
        provider=ProviderConfig(roles=ModelRoles(subagent="role-model")),
        subagent=SubagentConfig(default_model="legacy-model"),
    )
    factory, _ = _factory(cfg, tmp_path)
    assert factory.build_child(preset=resolve_subagent_type("Explore"), depth=1).session.model == (
        "role-model"
    )
    arg = factory.build_child(preset=resolve_subagent_type("Explore"), depth=1, model="arg-model")
    assert arg.session.model == "arg-model"
    legacy_cfg = RuntimeConfig(subagent=SubagentConfig(default_model="legacy-model"))
    legacy_factory, _ = _factory(legacy_cfg, tmp_path)
    child = legacy_factory.build_child(preset=resolve_subagent_type("Explore"), depth=1)
    assert child.session.model == "legacy-model"
    plain_factory, plain_built = _factory(RuntimeConfig(), tmp_path)
    plain = plain_factory.build_child(preset=resolve_subagent_type("Explore"), depth=1)
    assert plain.session.model == "parent-model" and plain_built == []


def test_build_child_applies_subagent_sampling_override(tmp_path: Path) -> None:
    override = SamplingConfig(reasoning_effort="low")
    cfg = RuntimeConfig(
        provider=ProviderConfig(
            sampling=SamplingConfig(temperature=0.5),
            roles=ModelRoles(sampling_overrides={"subagent": override}),
        )
    )
    factory, _ = _factory(cfg, tmp_path)
    child = factory.build_child(preset=resolve_subagent_type("Explore"), depth=1)
    assert child.config.provider.sampling == override
    assert cfg.provider.sampling.temperature == 0.5


def test_build_child_summarize_role_reaches_child_compactor(tmp_path: Path) -> None:
    cfg = RuntimeConfig(provider=ProviderConfig(roles=ModelRoles(summarize="sum-model")))
    factory, built = _factory(cfg, tmp_path)
    child = factory.build_child(preset=resolve_subagent_type("Explore"), depth=1)
    assert child.compactor._model == "sum-model"
    assert built == ["sum-model"]

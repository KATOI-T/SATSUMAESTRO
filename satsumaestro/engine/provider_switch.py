"""プロバイダへの接続と切り替え（画面に出さず、文は値で返す）。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from satsumaestro.config.models import ChosenProvider, ProviderProfile, RuntimeConfig
from satsumaestro.engine.compaction import Compactor
from satsumaestro.engine.context_fit import (
    ContextFitResult,
    MissingModelNotice,
    check_model,
    check_reachable,
    detect_local_server,
    list_models,
    local_server_for,
    missing_model_notice,
)
from satsumaestro.i18n.text import ui
from satsumaestro.orchestration.worker import WorkerState
from satsumaestro.providers.base import Provider
from satsumaestro.providers.choice import (
    CHOICE_LABELS,
    CLOUD_KEYS,
    DEFAULT_PORTS,
    LOCAL_CHOICES,
    PROVIDER_CHOICES,
    Endpoint,
    current_choice,
    is_choice,
    missing_text,
    resolve_endpoint,
    switch_endpoint,
)
from satsumaestro.providers.detect import same_model
from satsumaestro.providers.factory import build_provider, resolve_role_model, resolve_role_sampling
from satsumaestro.providers.local import LlamaCppServer, Unloaded, unload_used
from satsumaestro.providers.registry import expand_model_name
from satsumaestro.util.errors import ConfigError
from satsumaestro.util.urls import host_url, openai_base

if TYPE_CHECKING:
    from satsumaestro.engine.runtime import ConversationRuntime

ProviderBuilder = Callable[[RuntimeConfig, str, Mapping[str, str]], Provider]

_BUSY_WORKERS = frozenset({WorkerState.STARTING, WorkerState.RUNNING})


class MissingModelError(ConfigError):
    """接続先のローカルのサーバに無いモデルで起動した。"""

    def __init__(self, notice: MissingModelNotice) -> None:
        super().__init__(notice.text())
        self.notice = notice


class ModelNeededError(ConfigError):
    """切り替え先で使うモデルが決まらない（ローカルのサーバならモデルの一覧を持つ）。"""

    def __init__(self, choice: str, models: list[str]) -> None:
        name = CHOICE_LABELS[choice]
        if choice in LOCAL_CHOICES:
            text = ui("engine.provider_switch.pick_a_local_model", name=name, choice=choice)
        else:
            text = ui("engine.provider_switch.name_a_cloud_model", name=name, choice=choice)
        super().__init__(text)
        self.choice = choice
        self.models = models


@dataclass(frozen=True)
class Connection:
    """組み立てたプロバイダと、起動時に知らせる注意。"""

    provider: Provider
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class SwitchResult:
    """切り替えた結果（保存は呼び側で行う）。"""

    choice: ChosenProvider
    model: str
    endpoint: Endpoint | None
    previous: str | None
    remember: dict[ChosenProvider, ProviderProfile] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    unloaded: tuple[Unloaded, ...] = ()
    fit: ContextFitResult | None = None

    @property
    def base_url(self) -> str | None:
        """設定の ``base_url`` に固定する接続先（ローカルの種類だけ）。"""
        return self.endpoint.url if self.choice in LOCAL_CHOICES and self.endpoint else None


async def connect(
    cfg: RuntimeConfig,
    model: str,
    env: Mapping[str, str],
    *,
    startup: bool = True,
    build: ProviderBuilder = build_provider,
) -> Connection:
    """サーバの判定 → 組み立て → 届くか → モデルがあるか → ツールのテンプレートを確かめる。"""
    await detect_local_server(cfg, model, env)
    provider = build(cfg, model, env)
    problem = await check_reachable(cfg, model, env)
    if problem is not None:
        raise ConfigError(problem)
    check = await check_model(cfg, model, env)
    if check is not None and check.missing:
        raise MissingModelError(missing_model_notice(model, check, startup=startup))
    note = await tools_note(cfg, model, env)
    return Connection(provider, (note,) if note else ())


async def tools_note(cfg: RuntimeConfig, model: str, env: Mapping[str, str]) -> str | None:
    """llama-server のチャットテンプレートがツールを受け付けないときの注意。"""
    server = local_server_for(cfg, model, env)
    if isinstance(server, LlamaCppServer) and await server.tools_supported() is False:
        return ui("engine.provider_switch.llamacpp_template_has_no_tools", model=model)
    return None


def build_compactor(
    cfg: RuntimeConfig,
    provider: Provider,
    model: str,
    env: Mapping[str, str],
    *,
    build: ProviderBuilder = build_provider,
) -> Compactor:
    """summarize ロールを解決してコンパクタを組む。"""
    sum_model = resolve_role_model(cfg, "summarize", default=model)
    sum_provider = provider if sum_model == model else build(cfg, sum_model, env)
    return Compactor.from_config(
        cfg,
        provider=sum_provider,
        model=sum_model,
        sampling=resolve_role_sampling(cfg, "summarize"),
    )


def unloaded_text(unloaded: Unloaded) -> str:
    """降ろしたモデルの知らせ。"""
    names = ", ".join(unloaded.models)
    if unloaded.server == "lmstudio":
        return ui("engine.provider_switch.unloaded_from_lmstudio", unloaded=names)
    if unloaded.server == "llamacpp":
        return ui("engine.provider_switch.unloaded_from_llamacpp", unloaded=names)
    return ui("engine.provider_switch.unloaded_from_ollama", unloaded=names)


async def switch_provider(
    rt: ConversationRuntime,
    choice: str,
    *,
    env: Mapping[str, str],
    url: str | None = None,
    model: str | None = None,
    build: ProviderBuilder = build_provider,
) -> SwitchResult:
    """プロバイダを切り替える。足りないときは何も変えずに ``ConfigError`` などを投げる。"""
    if not is_choice(choice):
        names = ", ".join(PROVIDER_CHOICES)
        raise ConfigError(ui("engine.provider_switch.unknown_provider", name=choice, names=names))
    chosen = cast(ChosenProvider, choice)
    old = rt.config.provider
    now = rt.session.model
    previous = current_choice(old, now, env)
    if url is not None and chosen not in LOCAL_CHOICES:
        raise ConfigError(ui("engine.provider_switch.url_is_for_local", name=CHOICE_LABELS[chosen]))
    endpoint = (
        Endpoint(openai_base(host_url(url, DEFAULT_PORTS[chosen])), "argument")
        if url is not None
        else switch_endpoint(old, chosen, env, current=previous)
    )
    key = CLOUD_KEYS.get(chosen)
    if (chosen in LOCAL_CHOICES and endpoint is None) or (key and not env.get(key)):
        raise ConfigError(missing_text(chosen))

    remember = _outgoing_profile(rt.config, previous, chosen, now, env)
    trial = _with_choice(rt.config, chosen, endpoint, remember)
    if chosen in LOCAL_CHOICES:
        problem = await check_reachable(trial, now, env)
        if problem is not None:
            raise ConfigError(problem)
    resolved = await _pick_model(trial, chosen, previous, now, model, endpoint, env)
    connection = await connect(trial, resolved, env, startup=False, build=build)
    chosen_settings = trial.provider.model_copy(
        update={
            "default_model": resolved,
            "profiles": {
                **trial.provider.profiles,
                chosen: _profile(chosen, endpoint, resolved),
            },
        }
    )
    staged = rt.config.model_copy(update={"provider": chosen_settings})
    compactor = build_compactor(staged, connection.provider, resolved, env, build=build)

    old_server = local_server_for(rt.config, now, env)
    rt.config.provider = chosen_settings
    rt.use_provider(connection.provider, resolved, compactor=compactor)
    fit = await rt.refit_for_model()

    unloaded: list[Unloaded] = []
    new_server = local_server_for(rt.config, resolved, env)
    moved = old_server is not None and (new_server is None or new_server.root != old_server.root)
    if moved and old_server is not None and not _background_running(rt):
        unloaded = await unload_used(rt.config.provider, root=old_server.root)
    return SwitchResult(
        choice=chosen,
        model=resolved,
        endpoint=endpoint,
        previous=previous,
        remember=remember,
        notes=connection.notes,
        unloaded=tuple(unloaded),
        fit=fit,
    )


def _with_choice(
    cfg: RuntimeConfig,
    choice: ChosenProvider,
    endpoint: Endpoint | None,
    remember: Mapping[ChosenProvider, ProviderProfile],
) -> RuntimeConfig:
    """種類と接続先を替えた設定の写し（入れ子の項目は今の設定と共有する）。"""
    update: dict[str, object] = {
        "kind": choice,
        "profiles": {**remember, **cfg.provider.profiles},
    }
    if choice in LOCAL_CHOICES and endpoint is not None:
        update["base_url"] = endpoint.url
    return cfg.model_copy(update={"provider": cfg.provider.model_copy(update=update)})


def _profile(choice: ChosenProvider, endpoint: Endpoint | None, model: str) -> ProviderProfile:
    url = endpoint.url if choice in LOCAL_CHOICES and endpoint is not None else None
    return ProviderProfile(base_url=url, model=model)


def _outgoing_profile(
    cfg: RuntimeConfig,
    previous: str | None,
    choice: ChosenProvider,
    model: str,
    env: Mapping[str, str],
) -> dict[ChosenProvider, ProviderProfile]:
    """切り替える前のプロバイダに記録が無ければ、戻れるように今の接続先とモデルを残す。"""
    if previous is None or previous == choice or not is_choice(previous):
        return {}
    kind = cast(ChosenProvider, previous)
    if kind in cfg.provider.profiles:
        return {}
    endpoint = resolve_endpoint(cfg.provider, env) if kind in LOCAL_CHOICES else None
    return {kind: _profile(kind, endpoint, model)}


async def _pick_model(
    cfg: RuntimeConfig,
    choice: ChosenProvider,
    previous: str | None,
    now: str,
    given: str | None,
    endpoint: Endpoint | None,
    env: Mapping[str, str],
) -> str:
    """引数 → 覚えたモデル → サーバのただ 1 つのモデル → 今のモデルが一覧にあればそれ。"""
    if given:
        return expand_model_name(given)
    remembered = _remembered(cfg, choice, previous, now, endpoint)
    if remembered:
        return remembered
    if choice not in LOCAL_CHOICES:
        raise ModelNeededError(choice, [])
    names = await list_models(cfg, now, env) or []
    if len(names) == 1:
        return names[0]
    match = next((name for name in names if same_model(name, now)), None)
    if match is not None:
        return match
    raise ModelNeededError(choice, names)


def _remembered(
    cfg: RuntimeConfig,
    choice: ChosenProvider,
    previous: str | None,
    now: str,
    endpoint: Endpoint | None,
) -> str:
    """同じ接続先で前に使ったモデル（URL を替えたときは使わない）。"""
    if endpoint is not None and endpoint.source == "argument":
        if choice == previous:
            return ""
        profile = cfg.provider.profiles.get(choice)
        if profile is None or profile.base_url is None:
            return ""
        return profile.model if openai_base(profile.base_url) == endpoint.url else ""
    if choice == previous:
        return now
    profile = cfg.provider.profiles.get(choice)
    return profile.model if profile is not None else ""


def _background_running(rt: ConversationRuntime) -> bool:
    """背景のタスクかワーカーが、前のモデルを使っているかもしれないか。"""
    running = getattr(rt.orchestrator, "running_count", None)
    if callable(running) and running():
        return True
    ledger = getattr(getattr(rt.orchestrator, "workers", None), "ledger", None)
    return ledger is not None and any(w.state in _BUSY_WORKERS for w in ledger.list())


__all__ = [
    "Connection",
    "MissingModelError",
    "ModelNeededError",
    "ProviderBuilder",
    "SwitchResult",
    "build_compactor",
    "connect",
    "switch_provider",
    "tools_note",
    "unloaded_text",
]

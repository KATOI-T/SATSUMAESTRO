"""モデルコンテキスト長との圧縮閾値の自動整合。"""

from __future__ import annotations

import difflib
import logging
from collections.abc import Mapping
from dataclasses import dataclass

import httpx

from satsumaestro.config.models import RuntimeConfig
from satsumaestro.i18n.text import ui
from satsumaestro.providers.base import ProviderKind
from satsumaestro.providers.detect import (
    ContextSource,
    LocalKind,
    ModelContextInfo,
    choose_provider_kind,
)
from satsumaestro.providers.factory import openai_endpoint
from satsumaestro.providers.local import LocalServer, resolve_server, server_for_endpoint

_log = logging.getLogger(__name__)

CONTEXT_FIT_RATIO = 0.8


@dataclass(frozen=True)
class ContextFitResult:
    """整合の結果（doctor 表示・ログ用）。"""

    model: str
    info: ModelContextInfo
    original_threshold: int
    effective_threshold: int

    @property
    def clamped(self) -> bool:
        return self.effective_threshold != self.original_threshold

    @property
    def context_length(self) -> int | None:
        return self.info.context_length

    def describe(self) -> str:
        """画面に出す 1 行（起動時の ``context fit`` と doctor）。"""
        ctx = self.context_length
        kept = self.original_threshold
        if ctx is None:
            return ui(
                "engine.context_fit.could_not_detect_the_context", model=self.model, kept=kept
            )
        head = f"{self.model}: context_length={ctx} [{source_label(self.info.source)}]"
        if self.clamped:
            return f"{head} → compact_token_threshold {kept} → {self.effective_threshold}"
        return head + ui("engine.context_fit.compact_token_threshold_unchanged", kept=kept)


def source_label(source: str) -> str:
    """コンテキスト長をどこから読んだか。"""
    if source == "ps":
        return ui("engine.context_fit.effective_num_ctx_loaded")
    if source == "show.parameters":
        return ui("engine.context_fit.num_ctx_in_the_modelfile")
    if source == "show.model_info":
        return ui("engine.context_fit.the_model_s_native_window")
    if source == "lmstudio.loaded":
        return ui("engine.context_fit.lmstudio_loaded_window")
    if source == "lmstudio.max":
        return ui("engine.context_fit.lmstudio_largest_window")
    return ui("engine.context_fit.unknown")


def loaded_source(server: LocalKind) -> ContextSource:
    """載っているモデルの窓を読んだときの出どころ。"""
    return "lmstudio.loaded" if server == "lmstudio" else "ps"


def fit_compaction_threshold(
    threshold: int,
    context_length: int | None,
    *,
    enabled: bool = True,
    ratio: float = CONTEXT_FIT_RATIO,
) -> int | None:
    """クランプ後の閾値を返す。"""
    if not enabled or context_length is None or context_length <= 0 or threshold <= 0:
        return None
    limit = int(context_length * ratio)
    return limit if threshold > limit else None


def local_server_for(cfg: RuntimeConfig, model: str, env: Mapping[str, str]) -> LocalServer | None:
    """実際に使う接続先がローカルのサーバならそれを、そうでなければ ``None``。"""
    endpoint = _local_endpoint(cfg, model, env)
    return server_for_endpoint(endpoint, env, cfg.provider) if endpoint else None


async def detect_local_server(
    cfg: RuntimeConfig,
    model: str,
    env: Mapping[str, str],
    *,
    client: httpx.AsyncClient | None = None,
) -> LocalServer | None:
    """起動時に 1 回だけ、ローカルに見える接続先へ問い合わせてサーバの種類を決める。"""
    endpoint = _local_endpoint(cfg, model, env)
    if not endpoint:
        return None
    return await resolve_server(endpoint, env, cfg.provider, client=client)


def _local_endpoint(cfg: RuntimeConfig, model: str, env: Mapping[str, str]) -> str | None:
    if choose_provider_kind(model, env, cfg.provider.base_url) is not ProviderKind.OPENAI:
        return None
    return openai_endpoint(cfg, env)


async def refit_window(
    cfg: RuntimeConfig,
    model: str,
    env: Mapping[str, str],
    *,
    client: httpx.AsyncClient | None = None,
) -> int | None:
    """載っているモデルの実際の窓が、いま使っている窓より小さければ合わせる。"""
    server = local_server_for(cfg, model, env)
    if server is None:
        return None
    info = await server.context_info(model, client=client)
    window = info.context_length
    if not info.confirmed or window is None:
        return None
    current = cfg.compaction.context_window
    if current and window >= current:
        return None
    cfg.compaction.context_window = window
    fitted = fit_compaction_threshold(
        cfg.compact_token_threshold, window, enabled=cfg.compaction.auto_context_fit
    )
    if fitted is not None:
        cfg.compact_token_threshold = fitted
    _log.info("refit the context window: %d (threshold %d)", window, cfg.compact_token_threshold)
    return window


async def check_reachable(
    cfg: RuntimeConfig,
    model: str,
    env: Mapping[str, str],
    *,
    client: httpx.AsyncClient | None = None,
    timeout_s: float = 3.0,
) -> str | None:
    """起動前にローカルのサーバへ届くかを確かめる。"""
    server = local_server_for(cfg, model, env)
    if server is None:
        return None
    cause = await server.reachable(client=client, timeout_s=timeout_s)
    if cause is None:
        return None
    if server.kind == "lmstudio":
        return ui(
            "engine.context_fit.cannot_reach_lmstudio_check_that", root=server.root, cause=cause
        )
    return ui("engine.context_fit.cannot_reach_ollama_check_that", root=server.root, cause=cause)


@dataclass(frozen=True)
class ModelCheck:
    """接続先のローカルのサーバにモデルがあるか。"""

    root: str
    missing: bool
    available: list[str]
    server: LocalKind = "ollama"


async def list_models(
    cfg: RuntimeConfig,
    model: str,
    env: Mapping[str, str],
    *,
    client: httpx.AsyncClient | None = None,
    timeout_s: float = 3.0,
) -> list[str] | None:
    """接続先がローカルのサーバならモデルの名前。"""
    server = local_server_for(cfg, model, env)
    if server is None:
        return None
    return await server.list_models(client=client, timeout_s=timeout_s)


async def check_model(
    cfg: RuntimeConfig,
    model: str,
    env: Mapping[str, str],
    *,
    client: httpx.AsyncClient | None = None,
    timeout_s: float = 3.0,
) -> ModelCheck | None:
    """接続先のローカルのサーバでモデルの有無を確かめる。"""
    server = local_server_for(cfg, model, env)
    if server is None:
        return None
    owned = client is None
    http = client if client is not None else httpx.AsyncClient(timeout=timeout_s)
    try:
        found = await server.has_model(model, client=http)
        if found is None:
            return None
        if found:
            return ModelCheck(server.root, missing=False, available=[], server=server.kind)
        available = await server.list_models(client=http) or []
        return ModelCheck(server.root, missing=True, available=available, server=server.kind)
    finally:
        if owned:
            await http.aclose()


_LISTED_MODELS = 10


@dataclass(frozen=True)
class MissingModelNotice:
    """存在しないモデルの案内の部品。"""

    model: str
    root: str
    close: list[str]
    shown: list[str]
    rest: int
    startup: bool
    server: LocalKind = "ollama"

    @property
    def headline(self) -> str:
        if self.server == "lmstudio":
            return ui(
                "engine.context_fit.model_was_not_found_on_lmstudio",
                model=self.model,
                root=self.root,
            )
        return ui("engine.context_fit.model_was_not_found_on", model=self.model, root=self.root)

    @property
    def close_label(self) -> str:
        return ui("engine.context_fit.similar_names")

    @property
    def shown_label(self) -> str:
        return ui("engine.context_fit.available")

    @property
    def more(self) -> str:
        """一覧で省いた件数の付記（省いていなければ空）。"""
        if self.rest <= 0:
            return ""
        return ui("engine.context_fit.and_more", rest=self.rest)

    @property
    def pull_command(self) -> str:
        if self.server == "lmstudio":
            return f"lms get {self.model}"
        return f"ollama pull {self.model}"

    def pull_hint(self, command: str) -> str:
        """取得方法の 1 文（``command`` は取得コマンドの書き方。CLI は色を付けるために探す）。"""
        if self.startup:
            return ui("engine.context_fit.pull_hint_startup", command=command)
        return ui("engine.context_fit.pull_hint", command=command)

    def text(self) -> str:
        lines = [self.headline]
        if self.close:
            lines.append(f"{self.close_label}: {', '.join(self.close)}")
        if self.shown:
            lines.append(f"{self.shown_label}: {', '.join(self.shown)}{self.more}")
        lines.append(self.pull_hint(f"`{self.pull_command}`"))
        return "\n".join(lines)


def missing_model_notice(
    model: str, check: ModelCheck, *, startup: bool = True
) -> MissingModelNotice:
    """存在しないモデルの案内を組み立てる（近い名前は ``difflib`` で 3 件まで）。"""
    return MissingModelNotice(
        model=model,
        root=check.root,
        close=difflib.get_close_matches(model, check.available, n=3, cutoff=0.5),
        shown=check.available[:_LISTED_MODELS],
        rest=max(len(check.available) - _LISTED_MODELS, 0),
        startup=startup,
        server=check.server,
    )


def describe_missing_model(model: str, check: ModelCheck, *, startup: bool = True) -> str:
    """存在しないモデルを伝える文（近い名前・一覧・取得方法つき）。"""
    return missing_model_notice(model, check, startup=startup).text()


async def apply_context_fit(
    cfg: RuntimeConfig,
    model: str,
    env: Mapping[str, str],
    *,
    client: httpx.AsyncClient | None = None,
) -> ContextFitResult | None:
    """ローカルのサーバの判定 → 検出 → クランプを起動時に一度だけ行う（対象外なら ``None``）。"""
    if not cfg.compaction.auto_context_fit or cfg.compact_token_threshold <= 0:
        return None
    server = local_server_for(cfg, model, env)
    if server is None:
        return None
    info = await server.context_info(model, client=client)
    if cfg.compaction.context_window <= 0 and info.context_length:
        cfg.compaction.context_window = info.context_length
    original = cfg.compact_token_threshold
    fitted = fit_compaction_threshold(original, info.context_length)
    result = ContextFitResult(
        model=model,
        info=info,
        original_threshold=original,
        effective_threshold=fitted if fitted is not None else original,
    )
    if fitted is not None:
        cfg.compact_token_threshold = fitted
    _log.log(
        logging.INFO if fitted is not None else logging.DEBUG,
        "context fit: %s context_length=%s (%s) threshold %d -> %d",
        model,
        info.context_length,
        info.source,
        original,
        result.effective_threshold,
    )
    if info.unconfirmed and info.server == "lmstudio":
        _log.info(
            "LM Studio has not loaded %s yet; it loads with the context length set for the model, "
            "so the window is checked again after the first reply",
            model,
        )
    elif info.unconfirmed:
        _log.warning(
            "could not detect Ollama's effective num_ctx (the model is not loaded). If it is "
            "the server default OLLAMA_CONTEXT_LENGTH (4096 when unset), the context may overflow "
            "before compaction. Check with `satsumaestro doctor --probe`."
        )
    return result

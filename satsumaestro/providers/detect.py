"""リクエストを送るプロバイダの判定と、ローカルのサーバの検出。"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlparse

import httpx

from satsumaestro.providers.base import ProviderKind
from satsumaestro.providers.registry import vendor_for_model
from satsumaestro.util.urls import OLLAMA_DEFAULT_PORT, ollama_host_url

_log = logging.getLogger(__name__)

_NUM_CTX_RE = re.compile(r"^\s*num_ctx\s+(\d+)\s*$", re.MULTILINE)


_KEY_ORDER: tuple[tuple[str, ProviderKind], ...] = (
    ("OPENAI_API_KEY", ProviderKind.OPENAI),
    ("ANTHROPIC_API_KEY", ProviderKind.ANTHROPIC),
    ("XAI_API_KEY", ProviderKind.XAI),
)


def choose_provider_kind(
    model: str, env: Mapping[str, str], base_url: str | None = None, *, choice: str = "auto"
) -> ProviderKind:
    """設定の種類、または接続先の設定・モデル ID・環境変数から、プロバイダの種類を決める。"""
    if choice == "anthropic":
        return ProviderKind.ANTHROPIC
    if choice == "xai":
        return ProviderKind.XAI
    if choice != "auto":
        return ProviderKind.OPENAI
    if base_url or env.get("OLLAMA_HOST"):
        return ProviderKind.OPENAI
    vendor = vendor_for_model(model)
    if vendor is not None:
        return vendor
    if env.get("OPENAI_BASE_URL"):
        return ProviderKind.OPENAI
    return next((kind for var, kind in _KEY_ORDER if env.get(var)), ProviderKind.OPENAI)


OllamaDetectMode = Literal["auto", "on", "off"]


def ollama_root(base_url: str) -> str:
    """OpenAI 互換 URL からネイティブ API のルートを得る。"""
    trimmed = base_url.rstrip("/")
    for suffix in ("/api/v1", "/v1"):
        if trimmed.endswith(suffix):
            return trimmed[: -len(suffix)]
    return trimmed


def _address_of(url: str) -> tuple[str | None, int | None]:
    parsed = urlparse(url if "://" in url else f"http://{url}")
    return parsed.hostname, parsed.port


def is_ollama_endpoint(
    base_url: str, env: Mapping[str, str], mode: OllamaDetectMode = "auto"
) -> bool:
    """OpenAI 互換エンドポイントが Ollama かを判定する。"""
    if mode == "on":
        return True
    if mode == "off":
        return False
    host, port = _address_of(base_url)
    if port == OLLAMA_DEFAULT_PORT:
        return True
    configured = env.get("OLLAMA_HOST")
    if configured and (host, port) == _address_of(ollama_host_url(configured)):
        return True
    return False


LocalKind = Literal["ollama", "lmstudio", "llamacpp"]

ContextSource = Literal[
    "ps",
    "show.parameters",
    "show.model_info",
    "lmstudio.loaded",
    "lmstudio.max",
    "llamacpp.slot",
    "llamacpp.train",
    "none",
]


@dataclass(frozen=True)
class ModelContextInfo:
    """ローカルのサーバから得たモデルのコンテキスト長情報。"""

    effective: int | None = None
    native: int | None = None
    source: ContextSource = "none"
    loaded: bool = False
    expires_at: str | None = None
    ps_ok: bool = False
    server: LocalKind = "ollama"
    slots: int | None = None

    @property
    def context_length(self) -> int | None:
        """整合に使う長さ（実効値を優先し、無ければ本来の窓）。"""
        return self.effective if self.effective is not None else self.native

    @property
    def confirmed(self) -> bool:
        """載っているモデルの実効の窓を読めたか。"""
        return self.source in ("ps", "lmstudio.loaded", "llamacpp.slot")

    @property
    def unconfirmed(self) -> bool:
        """載っておらず本来の窓しか読めなかったか（載ると小さくなりうる）。"""
        return self.source in ("show.model_info", "lmstudio.max", "llamacpp.train")


def same_model(a: str, b: str) -> bool:
    def norm(name: str) -> str:
        name = name.strip().casefold()
        return name[: -len(":latest")] if name.endswith(":latest") else name

    return norm(a) == norm(b)


def _positive_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


async def fetch_model_context(
    root: str,
    model: str,
    *,
    client: httpx.AsyncClient | None = None,
    timeout_s: float = 2.0,
) -> ModelContextInfo:
    """``/api/ps`` → ``/api/show`` の順にコンテキスト長を取得する（失敗は ``source="none"``）。"""
    owned = client is None
    http = client if client is not None else httpx.AsyncClient(timeout=timeout_s)
    try:
        effective: int | None = None
        source: ContextSource = "none"
        loaded = False
        expires_at: str | None = None
        ps_ok = False
        try:
            resp = await http.get(f"{root}/api/ps")
            resp.raise_for_status()
            models = resp.json().get("models") or []
            ps_ok = True
            for entry in models:
                name = entry.get("name") or entry.get("model") or ""
                if not same_model(str(name), model):
                    continue
                loaded = True
                raw_expiry = entry.get("expires_at")
                expires_at = str(raw_expiry) if raw_expiry else None
                effective = _positive_int(entry.get("context_length"))
                if effective is not None:
                    source = "ps"
                break
        except Exception as exc:
            _log.debug("Ollama /api/ps query failed: %s", exc)

        native: int | None = None
        try:
            resp = await http.post(f"{root}/api/show", json={"model": model})
            resp.raise_for_status()
            data = resp.json()
            if effective is None:
                match = _NUM_CTX_RE.search(str(data.get("parameters") or ""))
                if match:
                    effective = _positive_int(int(match.group(1)))
                    if effective is not None:
                        source = "show.parameters"
            info = data.get("model_info") or {}
            for key, value in info.items():
                if str(key).endswith(".context_length"):
                    native = _positive_int(value)
                    if native is not None:
                        break
            if effective is None and native is not None:
                source = "show.model_info"
        except Exception as exc:
            _log.debug("Ollama /api/show query failed: %s", exc)

        return ModelContextInfo(
            effective=effective,
            native=native,
            source=source,
            loaded=loaded,
            expires_at=expires_at,
            ps_ok=ps_ok,
        )
    finally:
        if owned:
            await http.aclose()


async def model_loaded(
    root: str,
    model: str,
    *,
    client: httpx.AsyncClient | None = None,
    timeout_s: float = 1.5,
) -> bool | None:
    """``/api/ps`` に ``model`` が載っているか。"""
    owned = client is None
    http = client if client is not None else httpx.AsyncClient(timeout=timeout_s)
    try:
        resp = await http.get(f"{root}/api/ps")
        resp.raise_for_status()
        models = resp.json().get("models") or []
        return any(
            same_model(str(entry.get("name") or entry.get("model") or ""), model)
            for entry in models
        )
    except Exception as exc:
        _log.debug("Ollama /api/ps query failed: %s", exc)
        return None
    finally:
        if owned:
            await http.aclose()


async def touch_keep_alive(
    root: str,
    model: str,
    keep_alive: str,
    *,
    client: httpx.AsyncClient | None = None,
    timeout_s: float = 5.0,
) -> None:
    """``/api/generate`` に ``keep_alive`` のみを送り、モデルの常駐期限を延長する。"""
    owned = client is None
    http = client if client is not None else httpx.AsyncClient(timeout=timeout_s)
    try:
        resp = await http.post(
            f"{root}/api/generate", json={"model": model, "keep_alive": keep_alive}
        )
        if resp.status_code >= 400:
            _log.debug(
                "Ollama refused the keep_alive update: %s %s", resp.status_code, resp.text[:200]
            )
    except Exception as exc:
        _log.debug("Ollama keep_alive update failed: %s", exc)
    finally:
        if owned:
            await http.aclose()


async def unload_model(
    root: str,
    model: str,
    *,
    client: httpx.AsyncClient | None = None,
    timeout_s: float = 3.0,
) -> bool:
    """``/api/generate`` に ``keep_alive: 0`` を送り、``model`` を降ろす。"""
    owned = client is None
    http = client if client is not None else httpx.AsyncClient(timeout=timeout_s)
    try:
        resp = await http.post(f"{root}/api/generate", json={"model": model, "keep_alive": 0})
        if resp.status_code >= 400:
            _log.debug("Ollama refused the unload: %s %s", resp.status_code, resp.text[:200])
            return False
        return True
    except Exception as exc:
        _log.debug("Ollama unload failed: %s", exc)
        return False
    finally:
        if owned:
            await http.aclose()


_USED_MODELS: dict[tuple[str, str], None] = {}


def note_model_used(root: str, model: str) -> None:
    """このプロセスが Ollama の ``root`` で ``model`` を使った（応答を受け始めた）と記録する。"""
    _USED_MODELS[(root.rstrip("/"), model)] = None


def used_models() -> list[tuple[str, str]]:
    """記録した (ルート, モデル名) を使った順に返す。"""
    return list(_USED_MODELS)


def reset_used_models() -> None:
    """記録を空にする（テスト用）。"""
    _USED_MODELS.clear()


@dataclass
class ProviderCapabilities:
    """エンドポイント単位のプロセス内 capability キャッシュ。"""

    strict_tools_unsupported: bool = False
    strict_fallback_logged: bool = False

    def mark_strict_unsupported(self) -> None:
        self.strict_tools_unsupported = True


_CAPABILITIES: dict[str, ProviderCapabilities] = {}


@dataclass
class ModelQuirks:
    """OpenAI 公式 API と xAI のモデルごとに、400 から学んだ要求の形（プロセス内）。"""

    token_cap: int | None = None
    no_token_cap: bool = False
    rejected_values: set[tuple[str, str]] = field(default_factory=set)
    tools_need_effort_none: bool = False
    tools_unsupported: bool = False
    notice_shown: bool = False


_QUIRKS: dict[tuple[str, str], ModelQuirks] = {}


@dataclass
class AnthropicModelInfo:
    """Anthropic のモデルごとの機能（Models API）と、400 から学んだこと（プロセス内）。"""

    looked_up: bool = False
    known: bool = False
    thinking_types: frozenset[str] = frozenset()
    effort_levels: frozenset[str] = frozenset()
    max_tokens: int | None = None
    drop_unbound_thinking: bool = False
    notice_shown: bool = False


_ANTHROPIC_MODELS: dict[tuple[str, str], AnthropicModelInfo] = {}


def capabilities_for(base_url: str) -> ProviderCapabilities:
    """``base_url`` 単位のキャッシュを返す（無ければ作る）。"""
    return _CAPABILITIES.setdefault(base_url.rstrip("/"), ProviderCapabilities())


def model_quirks_for(base_url: str, model: str) -> ModelQuirks:
    """``(base_url, model)`` 単位の記憶を返す（無ければ作る）。"""
    return _QUIRKS.setdefault((base_url.rstrip("/"), model), ModelQuirks())


def anthropic_model_info_for(base_url: str, model: str) -> AnthropicModelInfo:
    """``(base_url, model)`` 単位の Anthropic のモデルの記録を返す（無ければ作る）。"""
    return _ANTHROPIC_MODELS.setdefault((base_url.rstrip("/"), model), AnthropicModelInfo())


def reset_capabilities() -> None:
    """記録を空にする（テストと、評価の 1 タスクごとの初期化）。"""
    _CAPABILITIES.clear()
    _QUIRKS.clear()
    _ANTHROPIC_MODELS.clear()

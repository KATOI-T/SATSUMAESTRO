"""4 層設定マージと .env ローダ。"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dotenv import dotenv_values, load_dotenv
from pydantic import ValidationError

from satsumaestro.config.models import (
    ChosenProvider,
    DotenvConfig,
    LanguageConfig,
    ProviderConfig,
    ProviderProfile,
    RuntimeConfig,
)
from satsumaestro.i18n.text import ui
from satsumaestro.util.environ import ENVIRON, forget_dotenv, hold_dotenv
from satsumaestro.util.errors import ConfigError
from satsumaestro.util.jsonl import write_text_atomic
from satsumaestro.util.paths import (
    AGENT_DIRNAME,
    default_config_home,
    find_project_file,
    project_dirs,
)
from satsumaestro.util.urls import ollama_host_url


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """後勝ちの再帰マージ。"""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


_ENV_TRUE = frozenset({"1", "true", "on", "yes"})
_ENV_FALSE = frozenset({"0", "false", "off", "no"})

_WEB_SEARCH_PROVIDERS = frozenset({"duckduckgo", "searxng", "generic", "none"})


def _env_bool(name: str) -> bool | None:
    """環境変数を真偽値として読む。"""
    raw = ENVIRON.get(name, "").strip().lower()
    if not raw:
        return None
    if raw in _ENV_TRUE:
        return True
    if raw in _ENV_FALSE:
        return False
    yes, no = "/".join(sorted(_ENV_TRUE)), "/".join(sorted(_ENV_FALSE))
    raise ConfigError(ui("config.loader.invalid_env_flag", name=name, raw=raw, yes=yes, no=no))


def _invalid(exc: ValidationError) -> str:
    return ui("config.loader.invalid_settings", exc=exc)


def _load_json(path: Path) -> dict[str, Any]:
    """JSON ファイルを dict として読む。"""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(
            ui("config.loader.invalid_json_in_the_settings", path=path, exc=exc)
        ) from exc
    if not isinstance(data, dict):
        raise ConfigError(ui("config.loader.the_settings_file_must_be", path=path))
    return data


def save_screen_language(code: str, config_home: Path | None = None) -> Path:
    """ユーザーの設定（①）の ``language.ui`` に ``code`` を書き、書いたファイルを返す。"""
    home = config_home if config_home is not None else default_config_home()
    path = home / "settings.json"
    data = _load_json(path)
    language = data.setdefault("language", {})
    if not isinstance(language, dict):
        raise ConfigError(ui("config.loader.the_language_item_must_be", path=path))
    language["ui"] = code
    write_text_atomic(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    return path


_SELECTION = ("kind", "base_url", "default_model")


def provider_update(
    data: Mapping[str, Any],
    choice: str,
    model: str,
    base_url: str | None,
    *,
    remember: Mapping[ChosenProvider, ProviderProfile] | None = None,
) -> dict[str, Any]:
    """設定の dict に、プロバイダの切り替え（種類・モデル・接続先・覚える分）を足した dict。"""
    provider, profiles = _provider_items(data)
    for kind, profile in (remember or {}).items():
        if kind not in profiles:
            profiles[kind] = profile.model_dump(exclude_none=True)
    profiles[choice] = ProviderProfile(base_url=base_url, model=model).model_dump(exclude_none=True)
    provider["kind"] = choice
    provider["default_model"] = model
    if base_url:
        provider["base_url"] = base_url
    provider["profiles"] = profiles
    return _with_provider(data, provider)


def model_update(
    data: Mapping[str, Any], model: str, *, choice: ChosenProvider | None = None
) -> dict[str, Any]:
    """設定の dict に、選んだモデルを足した dict（``choice`` の記録があればそのモデルも替える）。"""
    provider, profiles = _provider_items(data)
    provider["default_model"] = model
    profile = profiles.get(choice) if choice is not None else None
    if choice is not None and isinstance(profile, dict):
        profiles[choice] = {**profile, "model": model}
        provider["profiles"] = profiles
    return _with_provider(data, provider)


def save_provider(
    choice: str,
    model: str,
    base_url: str | None,
    *,
    remember: Mapping[ChosenProvider, ProviderProfile] | None = None,
    config_home: Path | None = None,
) -> Path:
    """ユーザーの設定（①）にプロバイダの切り替えを書き、書いたファイルを返す。"""
    path = _user_settings(config_home)
    data = provider_update(_load_json(path), choice, model, base_url, remember=remember)
    write_text_atomic(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    return path


def save_model(
    model: str, *, choice: ChosenProvider | None = None, config_home: Path | None = None
) -> Path:
    """ユーザーの設定（①）に選んだモデルを書き、書いたファイルを返す。"""
    path = _user_settings(config_home)
    data = model_update(_load_json(path), model, choice=choice)
    write_text_atomic(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    return path


def _user_settings(config_home: Path | None) -> Path:
    home = config_home if config_home is not None else default_config_home()
    return home / "settings.json"


def _provider_items(data: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """``provider`` と ``provider.profiles`` の写し（オブジェクトでなければ ConfigError）。"""
    provider = data.get("provider", {})
    profiles = provider.get("profiles", {}) if isinstance(provider, dict) else None
    if not isinstance(provider, dict) or not isinstance(profiles, dict):
        raise ConfigError(ui("config.loader.the_provider_item_must_be"))
    return dict(provider), dict(profiles)


def _with_provider(data: Mapping[str, Any], provider: dict[str, Any]) -> dict[str, Any]:
    try:
        ProviderConfig.model_validate(provider)
    except ValidationError as exc:
        raise ConfigError(_invalid(exc)) from exc
    out = dict(data)
    out["provider"] = provider
    return out


def _keep_selection(merged: dict[str, Any], user: Mapping[str, Any]) -> dict[str, Any]:
    """ユーザーの設定の最後の選択（種類・接続先・モデル）を、作業場所の層より優先する。"""
    chosen = user.get("provider")
    picked = {k: chosen[k] for k in _SELECTION if k in chosen} if isinstance(chosen, dict) else {}
    provider = merged.get("provider")
    if not picked or not isinstance(provider, dict):
        return merged
    return {**merged, "provider": {**provider, **picked}}


def prefer_dotenv_ollama(path: Path) -> None:
    """``.env`` の ``OLLAMA_*`` を、シェルの環境変数より優先する（空の値は除く）。"""
    if not path.is_file():
        return
    for key, value in dotenv_values(path).items():
        if key.startswith("OLLAMA_") and value and value.strip():
            os.environ[key] = value.strip()


class ConfigLoader:
    """4 層マージ + .env で ``RuntimeConfig`` を構築する。"""

    def __init__(self, cwd: Path, config_home: Path | None = None, *, trusted: bool) -> None:
        self.cwd = cwd.resolve()
        self.config_home = config_home if config_home is not None else default_config_home()
        self.trusted = trusted

    def load(self) -> RuntimeConfig:
        """4 層をマージ・検証し、.env で環境依存値を補完した設定を返す。"""
        merged = self._merge_layers()
        merged = self._overlay_env(merged)
        try:
            return RuntimeConfig.model_validate(merged)
        except ValidationError as exc:
            raise ConfigError(_invalid(exc)) from exc

    def load_language(self) -> LanguageConfig:
        """設定ファイルの ``language`` の項目だけを読む（ターンの頭で読み直す）。"""
        try:
            return LanguageConfig.model_validate(self._merge_layers().get("language") or {})
        except ValidationError as exc:
            raise ConfigError(_invalid(exc)) from exc

    def load_env(self) -> None:
        """信頼した作業場所の ``.env`` を読む（``OLLAMA_*`` 以外はシェルの値が勝つ）。"""
        self._read_env(self._merge_layers())

    def _read_env(self, merged: dict[str, Any]) -> None:
        """``dotenv.into_environ`` が偽なら、``.env`` を ``os.environ`` に書かずに持つ。"""
        if not self.trusted:
            return
        try:
            section = DotenvConfig.model_validate(merged.get("dotenv") or {})
        except ValidationError as exc:
            raise ConfigError(_invalid(exc)) from exc
        dotenv = find_project_file(self.cwd, ".env")
        if not section.into_environ:
            hold_dotenv(dotenv_values(dotenv) if dotenv is not None else {})
            return
        forget_dotenv()
        if dotenv is not None:
            load_dotenv(dotenv)
            prefer_dotenv_ollama(dotenv)

    def _merge_layers(self) -> dict[str, Any]:
        user = _load_json(self.config_home / "settings.json")
        if not self.trusted:
            return user
        merged = user
        for path in self._project_files():
            merged = deep_merge(merged, _load_json(path))
        return _keep_selection(merged, user)

    def _project_files(self) -> list[Path]:
        files: list[Path] = []
        shared = find_project_file(self.cwd, f"{AGENT_DIRNAME}.json")
        if shared is not None:
            files.append(shared)
        state_dir = self._project_state_dir()
        if state_dir is not None:
            files += [state_dir / "settings.json", state_dir / "settings.local.json"]
        return files

    def _project_state_dir(self) -> Path | None:
        home = self.config_home.resolve()
        for directory in project_dirs(self.cwd):
            state_dir = directory / AGENT_DIRNAME
            if state_dir.resolve() == home:
                continue
            if (state_dir / "settings.json").is_file() or (
                state_dir / "settings.local.json"
            ).is_file():
                return state_dir
        return None

    def _overlay_env(self, merged: dict[str, Any]) -> dict[str, Any]:
        """.env を読み、明示値が無い項目のみ環境依存値で補完する。"""
        self._read_env(merged)
        provider = dict(merged.get("provider") or {})
        rag = dict(merged.get("rag") or {})
        kind = provider.get("kind", "auto")

        host = ENVIRON.get("OLLAMA_HOST")
        if host and kind == "auto" and not provider.get("base_url"):
            provider["base_url"] = ollama_host_url(host)

        model = ENVIRON.get("OLLAMA_MODEL")
        if model and kind in ("auto", "ollama") and not provider.get("default_model"):
            provider["default_model"] = model

        embed_base = ENVIRON.get("SATSUMA_EMBED_BASE_URL")
        if embed_base and not rag.get("embed_base_url"):
            rag["embed_base_url"] = embed_base

        embed_model = ENVIRON.get("SATSUMA_EMBED_MODEL")
        if embed_model and "embed_model" not in rag:
            rag["embed_model"] = embed_model

        orchestration = dict(merged.get("orchestration") or {})
        decomposition = dict(merged.get("decomposition") or {})
        orch_flag = _env_bool("SATSUMA_ORCHESTRATION")
        if orch_flag is not None and "enabled" not in orchestration:
            orchestration["enabled"] = orch_flag
        decomp_flag = _env_bool("SATSUMA_DECOMPOSITION")
        if decomp_flag is not None and "enabled" not in decomposition:
            decomposition["enabled"] = decomp_flag

        web_search = dict(merged.get("web_search") or {})
        ws_provider = ENVIRON.get("SATSUMA_WEB_SEARCH_PROVIDER", "").strip().lower()
        if ws_provider and "provider" not in web_search:
            if ws_provider not in _WEB_SEARCH_PROVIDERS:
                names = "/".join(sorted(_WEB_SEARCH_PROVIDERS))
                raise ConfigError(
                    ui(
                        "config.loader.invalid_web_search_provider",
                        ws_provider=ws_provider,
                        names=names,
                    )
                )
            web_search["provider"] = None if ws_provider == "none" else ws_provider
        ws_base = ENVIRON.get("SATSUMA_WEB_SEARCH_BASE_URL")
        if ws_base and not web_search.get("base_url"):
            web_search["base_url"] = ws_base

        web_fetch = dict(merged.get("web_fetch") or {})
        wf_flag = _env_bool("SATSUMA_WEB_FETCH_ENABLED")
        if wf_flag is not None and "enabled" not in web_fetch:
            web_fetch["enabled"] = wf_flag

        if provider:
            merged["provider"] = provider
        if rag:
            merged["rag"] = rag
        if orchestration:
            merged["orchestration"] = orchestration
        if decomposition:
            merged["decomposition"] = decomposition
        if web_search:
            merged["web_search"] = web_search
        if web_fetch:
            merged["web_fetch"] = web_fetch
        return merged

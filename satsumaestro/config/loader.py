"""4 層設定マージと .env ローダ。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import dotenv_values, load_dotenv
from pydantic import ValidationError

from satsumaestro.config.models import LanguageConfig, RuntimeConfig
from satsumaestro.i18n.text import ui
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
    raw = os.environ.get(name, "").strip().lower()
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

    def _merge_layers(self) -> dict[str, Any]:
        merged = _load_json(self.config_home / "settings.json")
        if not self.trusted:
            return merged
        shared = find_project_file(self.cwd, f"{AGENT_DIRNAME}.json")
        if shared is not None:
            merged = deep_merge(merged, _load_json(shared))
        state_dir = self._project_state_dir()
        if state_dir is not None:
            merged = deep_merge(merged, _load_json(state_dir / "settings.json"))
            merged = deep_merge(merged, _load_json(state_dir / "settings.local.json"))
        return merged

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
        if self.trusted:
            dotenv = find_project_file(self.cwd, ".env")
            if dotenv is not None:
                load_dotenv(dotenv)
                prefer_dotenv_ollama(dotenv)
        provider = dict(merged.get("provider") or {})
        rag = dict(merged.get("rag") or {})

        host = os.environ.get("OLLAMA_HOST")
        if host and not provider.get("base_url"):
            provider["base_url"] = ollama_host_url(host)

        model = os.environ.get("OLLAMA_MODEL")
        if model and not provider.get("default_model"):
            provider["default_model"] = model

        embed_base = os.environ.get("SATSUMA_EMBED_BASE_URL")
        if embed_base and not rag.get("embed_base_url"):
            rag["embed_base_url"] = embed_base

        embed_model = os.environ.get("SATSUMA_EMBED_MODEL")
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
        ws_provider = os.environ.get("SATSUMA_WEB_SEARCH_PROVIDER", "").strip().lower()
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
        ws_base = os.environ.get("SATSUMA_WEB_SEARCH_BASE_URL")
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

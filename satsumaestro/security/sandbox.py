"""bash の子プロセスに掛ける隔離の設定。"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from satsumaestro.config.models import SandboxConfig
from satsumaestro.security.secret_env import strip_secret_env

FS_MODE_VAR = "SATSUMA_FS_MODE"
FS_MOUNTS_VAR = "SATSUMA_FS_MOUNTS"

_DEAD_END_PROXY = "http://127.0.0.1:9"
_PROXY_VARS = ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")
_NO_PROXY_VARS = ("no_proxy", "NO_PROXY")


class FilesystemMode(StrEnum):
    """ファイルシステムの隔離の段階（設定と bash の入力の値）。"""

    OFF = "off"
    WORKSPACE_ONLY = "workspace-only"
    ALLOW_LIST = "allow-list"


@dataclass(frozen=True)
class Isolation:
    """1 回の bash 実行に効く隔離（設定の既定に呼び出しの上書きを重ねた結果）。"""

    filesystem_mode: FilesystemMode = FilesystemMode.WORKSPACE_ONLY
    isolate_network: bool = False
    allowed_mounts: tuple[str, ...] = ()
    env_strip: tuple[str, ...] = ()
    env_keep: tuple[str, ...] = ()


def isolation_for(
    config: SandboxConfig,
    *,
    filesystem_mode: FilesystemMode | None = None,
    isolate_network: bool | None = None,
    allowed_mounts: Sequence[str] | None = None,
) -> Isolation:
    """設定の既定に、明示された上書き（``None`` でないもの）を重ねる。"""
    mode = FilesystemMode(config.filesystem_mode) if filesystem_mode is None else filesystem_mode
    mounts = tuple(config.allowed_mounts if allowed_mounts is None else allowed_mounts)
    if mode is FilesystemMode.ALLOW_LIST and not mounts:
        mode = FilesystemMode.WORKSPACE_ONLY
    return Isolation(
        filesystem_mode=mode,
        isolate_network=config.isolate_network if isolate_network is None else isolate_network,
        allowed_mounts=mounts,
        env_strip=tuple(config.env_strip),
        env_keep=tuple(config.env_keep),
    )


def mount_paths(isolation: Isolation, workspace: Path) -> tuple[Path, ...]:
    """allow-list で許す追加のパスを、jail と同じ規則で正規化する。"""
    if isolation.filesystem_mode is not FilesystemMode.ALLOW_LIST:
        return ()
    return tuple(
        (p if p.is_absolute() else workspace / p).resolve()
        for p in map(Path, isolation.allowed_mounts)
    )


def bash_environment(
    base: Mapping[str, str], *, isolation: Isolation, mounts: Sequence[Path] = ()
) -> dict[str, str]:
    """bash の子プロセスへ渡す環境変数。"""
    env = strip_secret_env(base, extra_strip=isolation.env_strip, keep=isolation.env_keep)
    if isolation.isolate_network:
        env.update(dict.fromkeys(_PROXY_VARS, _DEAD_END_PROXY))
        for name in _NO_PROXY_VARS:
            env.pop(name, None)
    env[FS_MODE_VAR] = isolation.filesystem_mode.value
    env[FS_MOUNTS_VAR] = os.pathsep.join(str(p) for p in mounts)
    return env


def child_environment(
    config: SandboxConfig | None = None, base: Mapping[str, str] | None = None
) -> dict[str, str]:
    """フック・LSP サーバなど bash 以外の子プロセスへ渡す環境（鍵を外した ``os.environ``）。"""
    cfg = config if config is not None else SandboxConfig()
    source = base if base is not None else os.environ
    return strip_secret_env(source, extra_strip=cfg.env_strip, keep=cfg.env_keep)


AUTO_ALLOWED_GIT_CONFIG: tuple[tuple[str, str], ...] = (
    ("safe.bareRepository", "explicit"),
    ("core.fsmonitor", "false"),
)


def with_git_config(env: Mapping[str, str], pairs: Sequence[tuple[str, str]]) -> dict[str, str]:
    """``env`` の写しに、git の設定を渡す環境変数を足す。"""
    out = dict(env)
    try:
        start = max(int(out.get("GIT_CONFIG_COUNT", "0")), 0)
    except ValueError:
        start = 0
    for i, (key, value) in enumerate(pairs, start=start):
        out[f"GIT_CONFIG_KEY_{i}"] = key
        out[f"GIT_CONFIG_VALUE_{i}"] = value
    out["GIT_CONFIG_COUNT"] = str(start + len(pairs))
    return out

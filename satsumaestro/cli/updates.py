"""新しい版の確かめ（PyPI か tap の formula の版を聞き、結果を 1 日持つ）。"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx
from rich.console import Console

from satsumaestro.__about__ import __version__
from satsumaestro.config.models import RuntimeConfig
from satsumaestro.i18n.text import ui
from satsumaestro.orchestration.worker_supervisor import WORKER_CHILD_ENV
from satsumaestro.util.clock import now_ms
from satsumaestro.util.environ import ENVIRON
from satsumaestro.util.install import Install, detect_install
from satsumaestro.util.jsonl import write_text_atomic
from satsumaestro.util.paths import default_config_home

_log = logging.getLogger(__name__)

Source = Literal["pypi", "tap"]

PYPI_URL = "https://pypi.org/pypi/satsumaestro/json"
TAP_FORMULA_URL = (
    "https://raw.githubusercontent.com/KATOI-T/homebrew-tap/main/Formula/satsumaestro.rb"
)
CACHE_FILE = "update_check.json"
CHECK_INTERVAL_MS = 24 * 60 * 60 * 1000
_TIMEOUT_S = 5.0
_VERSION = re.compile(r"\d+(?:\.\d+)*")
# resource の url と取り違えないよう、本体の配布物の名前で絞る
_FORMULA_URL = re.compile(r'^\s*url\s+"[^"]*/satsumaestro-(\d+(?:\.\d+)*)\.tar\.gz"', re.M)


def source_for(install: Install) -> Source:
    """入れ方ごとの最新の版の出どころ（brew は tap が更新されてから上げられる）。"""
    return "tap" if install.kind == "brew" else "pypi"


def parse_version(text: str) -> tuple[int, ...] | None:
    """数字だけの版を比べられる形にする（pre-release などは ``None``）。"""
    text = text.strip()
    if not _VERSION.fullmatch(text):
        return None
    return tuple(int(part) for part in text.split("."))


def is_newer(latest: str, current: str) -> bool:
    """``latest`` が ``current`` より新しいか（どちらかが読めなければ偽）。"""
    new, now = parse_version(latest), parse_version(current)
    if new is None or now is None:
        return False
    width = max(len(new), len(now))
    return new + (0,) * (width - len(new)) > now + (0,) * (width - len(now))


async def latest_version(
    source: Source, *, transport: httpx.AsyncBaseTransport | None = None
) -> str | None:
    """最新の版を聞く（届かない・読めないときは ``None``）。"""
    url = TAP_FORMULA_URL if source == "tap" else PYPI_URL
    headers = {"User-Agent": f"satsumaestro/{__version__}"}
    try:
        async with httpx.AsyncClient(
            timeout=_TIMEOUT_S, transport=transport, headers=headers, follow_redirects=True
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        _log.info("update check against %s did not finish: %s", source, exc)
        return None
    found = _formula_version(response.text) if source == "tap" else _pypi_version(response)
    if found is None:
        _log.info("update check against %s answered without a readable version", source)
    return found


def _pypi_version(response: httpx.Response) -> str | None:
    try:
        body = response.json()
    except ValueError:
        return None
    info = body.get("info") if isinstance(body, dict) else None
    version = info.get("version") if isinstance(info, dict) else None
    if not isinstance(version, str) or parse_version(version) is None:
        return None
    return version.strip()


def _formula_version(text: str) -> str | None:
    match = _FORMULA_URL.search(text)
    return match.group(1) if match else None


@dataclass(frozen=True)
class CachedCheck:
    """前に確かめた結果（``latest`` は分からなければ ``None``）。"""

    checked_at: int
    source: Source
    latest: str | None


def cache_path(config_home: Path | None = None) -> Path:
    """確かめた結果を置くファイル。"""
    return (config_home or default_config_home()) / CACHE_FILE


def load_cached(path: Path) -> CachedCheck | None:
    """前の結果を読む（無い・壊れているときは ``None``）。"""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    checked_at, source, latest = data.get("checked_at"), data.get("source"), data.get("latest")
    if not isinstance(checked_at, int) or source not in ("pypi", "tap"):
        return None
    if latest is not None and (not isinstance(latest, str) or parse_version(latest) is None):
        return None
    return CachedCheck(checked_at, source, latest)


def save_cached(path: Path, check: CachedCheck) -> None:
    """結果を書く（書けなくても止めない）。"""
    body = {"checked_at": check.checked_at, "source": check.source, "latest": check.latest}
    try:
        write_text_atomic(path, json.dumps(body, indent=2) + "\n")
    except OSError:
        _log.debug("update check cache write failed: %s", path, exc_info=True)


def is_due(cached: CachedCheck | None, source: Source, now: int | None = None) -> bool:
    """確かめ直す時か（結果が無い・出どころが違う・1 日を過ぎた）。"""
    if cached is None or cached.source != source:
        return True
    elapsed = (now_ms() if now is None else now) - cached.checked_at
    return not 0 <= elapsed < CHECK_INTERVAL_MS


async def check_now(
    source: Source, path: Path, *, transport: httpx.AsyncBaseTransport | None = None
) -> CachedCheck:
    """最新の版を聞いて書く（聞けなければ、同じ出どころの前の版を残す）。"""
    latest = await latest_version(source, transport=transport)
    if latest is None:
        previous = load_cached(path)
        if previous is not None and previous.source == source:
            latest = previous.latest
    check = CachedCheck(now_ms(), source, latest)
    save_cached(path, check)
    return check


class UpdateCheck:
    """REPL の起動時の確かめ（知らせは前に保存した結果から出し、裏で取り直す）。"""

    def __init__(
        self,
        install: Install,
        path: Path,
        *,
        current: str = __version__,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.install = install
        self.source = source_for(install)
        self.path = path
        self.current = current
        self._transport = transport

    @classmethod
    def for_repl(
        cls, cfg: RuntimeConfig, console: Console, env: Mapping[str, str] = ENVIRON
    ) -> UpdateCheck | None:
        """確かめるときだけ作る（設定で止めた・端末でない・worker・editable では ``None``）。"""
        if not cfg.update_check or not console.is_terminal or env.get(WORKER_CHILD_ENV) == "1":
            return None
        install = detect_install()
        if install.kind == "editable":
            return None
        return cls(install, cache_path())

    def notice(self) -> str | None:
        """保存した最新の版が今の版より新しければ、知らせの 1 行。"""
        cached = load_cached(self.path)
        if cached is None or cached.source != self.source or cached.latest is None:
            return None
        if not is_newer(cached.latest, self.current):
            return None
        return ui(
            "cli.updates.satsumaestro_is_available",
            latest=cached.latest,
            current=self.current,
            command=self.install.shown,
        )

    async def refresh(self) -> None:
        """1 日を過ぎていれば取り直す（失敗しても何も出さない）。"""
        try:
            if is_due(load_cached(self.path), self.source):
                await check_now(self.source, self.path, transport=self._transport)
        except Exception:
            _log.info("update check failed", exc_info=True)


__all__ = [
    "CACHE_FILE",
    "CHECK_INTERVAL_MS",
    "PYPI_URL",
    "TAP_FORMULA_URL",
    "CachedCheck",
    "Source",
    "UpdateCheck",
    "cache_path",
    "check_now",
    "is_due",
    "is_newer",
    "latest_version",
    "load_cached",
    "parse_version",
    "save_cached",
    "source_for",
]

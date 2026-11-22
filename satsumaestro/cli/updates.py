"""新しい版の確かめ（PyPI か tap の formula の版を聞き、結果を 1 日持つ）。"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx
from rich.console import Console
from rich.text import Text

from satsumaestro.__about__ import __version__
from satsumaestro.cli.line_input import ask_line
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
_PROGRAMS = frozenset({"satsumaestro", "satsuma", "stro"})


def program_name(argv0: str | None = None) -> str:
    """利用者が起動に使った名前（知らない名前なら ``satsumaestro``）。"""
    name = Path(sys.argv[0] if argv0 is None else argv0).stem
    return name if name in _PROGRAMS else "satsumaestro"


def update_command() -> str:
    """知らせと ``/update`` が案内する更新のコマンド。"""
    return f"{program_name()} update"


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
            command=update_command(),
        )

    async def refresh(self) -> None:
        """1 日を過ぎていれば取り直す（失敗しても何も出さない）。"""
        try:
            if is_due(load_cached(self.path), self.source):
                await check_now(self.source, self.path, transport=self._transport)
        except Exception:
            _log.info("update check failed", exc_info=True)


def _where(source: Source) -> str:
    return ui("cli.updates.the_homebrew_tap") if source == "tap" else "PyPI"


async def _ask_latest(source: Source) -> str | None:
    """利用者が頼んだ確かめ（前の結果は使わず、取れたときだけ書く）。"""
    latest = await latest_version(source)
    if latest is not None:
        save_cached(cache_path(), CachedCheck(now_ms(), source, latest))
    return latest


async def update_hint(console: Console, *, current: str = __version__) -> None:
    """``/update``: 版と更新のコマンドを出す（REPL の中では回さない）。"""
    install = detect_install()
    if install.kind == "editable":
        console.print(ui("cli.updates.this_is_an_editable_install"))
        return
    source = source_for(install)
    latest = await _ask_latest(source)
    if latest is None:
        unknown = ui("cli.updates.could_not_get_the", where=_where(source))
        console.print(Text(unknown, style="yellow"))
    elif not is_newer(latest, current):
        console.print(ui("cli.updates.is_the_latest_version", current=current))
        return
    else:
        console.print(
            Text(
                ui("cli.updates.satsumaestro_is_newer", latest=latest, current=current),
                style="yellow",
            )
        )
    console.print(ui("cli.updates.leave_the_repl_and", command=update_command()))


def run_update(
    console: Console, *, yes: bool, interactive: bool, current: str = __version__
) -> int:
    """``satsumaestro update``: 入れ方に合ったコマンドを、確かめてから回す（終了コードを返す）。"""
    install = detect_install()
    if install.kind == "editable":
        console.print(ui("cli.updates.this_is_an_editable_install"))
        return 0
    source = source_for(install)
    latest = asyncio.run(_ask_latest(source))
    command = install.shown
    if latest is None:
        console.print(Text(ui("cli.updates.could_not_get_the", where=_where(source)), style="red"))
        console.print(ui("cli.updates.command", command=command))
        return 1
    if not is_newer(latest, current):
        console.print(ui("cli.updates.is_the_latest_version", current=current))
        return 0
    console.print(ui("cli.updates.satsumaestro_is_newer", latest=latest, current=current))
    console.print(ui("cli.updates.command", command=command))
    if not install.runnable:
        console.print(ui("cli.updates.run_it_yourself"))
        return 0
    if not yes:
        if not interactive:
            console.print(ui("cli.updates.not_a_terminal"))
            return 1
        if not _confirmed(console):
            console.print(ui("cli.updates.not_upgraded"))
            return 0
    # 更新は自分のファイルを置き換えるので、後で出す文はここで作る
    finished = ui("cli.updates.the_upgrade_finished", check=f"{program_name()} --version")
    failed = ui("cli.updates.the_upgrade_failed", code="{code}")
    console.print(Text(ui("cli.updates.running", command=command), style="dim"))
    try:
        code = subprocess.run(install.argv, check=False).returncode
    except OSError as exc:
        _log.info("could not run %s: %s", install.argv, exc)
        code = 127
    if code == 0:
        console.print(Text(finished, style="green"))
        return 0
    console.print(Text(failed.replace("{code}", str(code)), style="red"))
    return code


def _confirmed(console: Console) -> bool:
    while True:
        try:
            answer = ask_line(console, ui("cli.updates.upgrade_now_y_n")).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if answer in ("y", "n"):
            return answer == "y"


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
    "program_name",
    "run_update",
    "save_cached",
    "source_for",
    "update_command",
    "update_hint",
]

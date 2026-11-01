"""起動時に、開いたディレクトリの設定を今回読むかを決める。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from rich.text import Text

from satsumaestro.cli.language_gate import HEADING_STYLE, NUMBER_STYLE
from satsumaestro.i18n.text import ui
from satsumaestro.security.trust import TrustStore, trust_root

REPO_SETTINGS = (".satsuma/", ".satsuma.json", ".env", "policy.md", "memory.md")
QUESTION = "(y/n/q): "
_KEY = r"\[[a-z]\]"
CAUTION_STYLE = "yellow"


class TrustBasis(StrEnum):
    FLAG = "flag"
    RECORDED = "recorded"
    ANSWERED = "answered"
    DECLINED = "declined"
    UNTRUSTED = "untrusted"


@dataclass(frozen=True)
class TrustDecision:
    root: Path
    basis: TrustBasis

    @property
    def trusted(self) -> bool:
        return self.basis in (TrustBasis.FLAG, TrustBasis.RECORDED, TrustBasis.ANSWERED)


class TrustQuit(Exception):
    """確認で q（終了）が選ばれた。"""


def _repo_settings() -> str:
    return ui("cli.trust_gate.list_separator").join(REPO_SETTINGS)


def question_lines(root: Path) -> list[Text]:
    """確認の前に出す説明。"""
    files = _repo_settings()
    choices = Text(ui("cli.trust_gate.trust_and_remember_continue_without"))
    choices.highlight_regex(_KEY, NUMBER_STYLE)
    return [
        Text(ui("cli.trust_gate.opening_this_directory_for_the", root=root), style=HEADING_STYLE),
        Text(ui("cli.trust_gate.if_you_trust_it_the", files=files)),
        Text(ui("cli.trust_gate.hooks_mcp_servers_lsp_servers"), style=CAUTION_STYLE),
        choices,
    ]


def untrusted_notice(root: Path) -> str:
    """非対話で信頼していないときに stderr へ出す 1 行。"""
    files = _repo_settings()
    return ui("cli.trust_gate.untrusted_directory_continuing_without_the", root=root, files=files)


def decide_trust(
    cwd: Path,
    *,
    trust_flag: bool,
    ask: Callable[[Path], str] | None,
    store: TrustStore | None = None,
) -> TrustDecision:
    """信頼の判断。"""
    root = trust_root(cwd)
    store = store if store is not None else TrustStore()
    if store.is_trusted(root):
        return TrustDecision(root, TrustBasis.RECORDED)
    if trust_flag:
        return TrustDecision(root, TrustBasis.FLAG)
    if ask is None:
        return TrustDecision(root, TrustBasis.UNTRUSTED)
    answer = ask(root)
    if answer == "y":
        store.trust(root)
        return TrustDecision(root, TrustBasis.ANSWERED)
    if answer == "q":
        raise TrustQuit
    return TrustDecision(root, TrustBasis.DECLINED)

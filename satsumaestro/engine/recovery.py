"""既知の失敗に、モデルへ差し戻す前に 1 回だけ手当てを試みる。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

from satsumaestro.util.errors import ProviderError, SessionError


class Fault(StrEnum):
    """手順書を持つ、既知の失敗の種類。"""

    PROVIDER = "provider"
    MCP_INIT = "mcp_init"
    BRANCH_BEHIND = "branch_behind"
    UNHEALTHY_SESSION = "unhealthy_session"
    TRUST_PENDING = "trust_pending"
    MISDELIVERED_PROMPT = "misdelivered_prompt"


class Remedy(StrEnum):
    """手当ての 1 手。"""

    RETRY_PROVIDER = "retry_provider"
    SWITCH_MODEL = "switch_model"
    RECONNECT_MCP = "reconnect_mcp"
    REBASE = "rebase"
    ASK_FOR_NEW_SESSION = "ask_for_new_session"
    CLEAR_TRUST_GATE = "clear_trust_gate"
    RESEND_PROMPT = "resend_prompt"


class Fallback(StrEnum):
    """手当てが尽きた（または最初の 1 手で失敗した）ときの落とし所。"""

    RETURN_ERROR = "return_error"
    ABORT_TURN = "abort_turn"


class PlaybookOutcome(StrEnum):
    """:func:`run_playbook` の結果。"""

    SUCCEEDED = "succeeded"
    INCOMPLETE = "incomplete"
    HANDED_OFF = "handed_off"


@dataclass(frozen=True)
class Playbook:
    """失敗の種類 1 つに対する手当ての列と、尽きたときの落とし所。"""

    remedies: tuple[Remedy, ...]
    fallback: Fallback = Fallback.RETURN_ERROR
    attempt_limit: int = 1


_PLAYBOOKS: dict[Fault, Playbook] = {
    Fault.PROVIDER: Playbook((Remedy.RETRY_PROVIDER, Remedy.SWITCH_MODEL)),
    Fault.MCP_INIT: Playbook((Remedy.RECONNECT_MCP,), Fallback.ABORT_TURN),
    Fault.BRANCH_BEHIND: Playbook((Remedy.REBASE,)),
    Fault.UNHEALTHY_SESSION: Playbook((Remedy.ASK_FOR_NEW_SESSION,), Fallback.ABORT_TURN),
    Fault.TRUST_PENDING: Playbook((Remedy.CLEAR_TRUST_GATE,)),
    Fault.MISDELIVERED_PROMPT: Playbook((Remedy.RESEND_PROMPT,)),
}


def playbook_for(fault: Fault) -> Playbook:
    """失敗の種類の手順書。"""
    return _PLAYBOOKS[fault]


@dataclass(frozen=True)
class PlaybookStep:
    """台帳の 1 行。"""

    fault: Fault
    remedy: Remedy | None
    outcome: PlaybookOutcome
    attempt: int
    note: str = ""


@runtime_checkable
class RemedyRunner(Protocol):
    """手当ての実処理。"""

    async def apply(self, remedy: Remedy, fault: Fault) -> bool: ...


@dataclass
class RecoveryLedger:
    """失敗の種類ごとの試行回数と、手当ての行。"""

    attempt_limit: int = 1
    _tries: dict[Fault, int] = field(default_factory=dict)
    _steps: list[PlaybookStep] = field(default_factory=list)

    def tries(self, fault: Fault) -> int:
        """この失敗に手順書を使った回数。"""
        return self._tries.get(fault, 0)

    def limit_for(self, fault: Fault) -> int:
        """この失敗に手順書を使える回数（手順書の上限と設定の上限の小さい方）。"""
        return min(playbook_for(fault).attempt_limit, self.attempt_limit)

    def is_spent(self, fault: Fault) -> bool:
        """使った上で上限に達した（一度も使っていない 0 回は含まない）。"""
        used = self.tries(fault)
        return used > 0 and used >= self.limit_for(fault)

    def steps(self) -> list[PlaybookStep]:
        """行の写し（呼び側が足したり消したりしても台帳は変わらない）。"""
        return list(self._steps)

    def begin(self, fault: Fault) -> int | None:
        """試行を 1 つ使い、その番号（1 始まり）を返す。"""
        used = self.tries(fault)
        if used >= self.limit_for(fault):
            return None
        self._tries[fault] = used + 1
        return used + 1

    def add(self, step: PlaybookStep) -> None:
        self._steps.append(step)


async def _apply(runner: RemedyRunner, remedy: Remedy, fault: Fault) -> tuple[bool, str]:
    """1 手を当てる。"""
    try:
        return bool(await runner.apply(remedy, fault)), ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


async def run_playbook(
    fault: Fault, ledger: RecoveryLedger, *, runner: RemedyRunner
) -> PlaybookOutcome:
    """手順書を 1 回使い、各手の結果を台帳に残して全体の結果を返す。"""
    attempt = ledger.begin(fault)
    if attempt is None:
        ledger.add(
            PlaybookStep(
                fault,
                None,
                PlaybookOutcome.HANDED_OFF,
                ledger.tries(fault),
                note=f"attempt limit {ledger.limit_for(fault)} reached",
            )
        )
        return PlaybookOutcome.HANDED_OFF
    for position, remedy in enumerate(playbook_for(fault).remedies, start=1):
        ok, note = await _apply(runner, remedy, fault)
        if ok:
            ledger.add(PlaybookStep(fault, remedy, PlaybookOutcome.SUCCEEDED, attempt, note))
            continue
        outcome = PlaybookOutcome.HANDED_OFF if position == 1 else PlaybookOutcome.INCOMPLETE
        ledger.add(PlaybookStep(fault, remedy, outcome, attempt, note))
        return outcome
    return PlaybookOutcome.SUCCEEDED


def fault_of(exc: BaseException) -> Fault | None:
    """例外から失敗の種類を引く。"""
    if isinstance(exc, SessionError):
        return Fault.UNHEALTHY_SESSION
    if isinstance(exc, ProviderError):
        return Fault.PROVIDER
    return None


class SessionAbortError(SessionError):
    """落とし所が ABORT_TURN のとき投げる: ターンを止め、新しいセッションの開始を促す。"""


__all__ = [
    "Fallback",
    "Fault",
    "Playbook",
    "PlaybookOutcome",
    "PlaybookStep",
    "RecoveryLedger",
    "Remedy",
    "RemedyRunner",
    "SessionAbortError",
    "fault_of",
    "playbook_for",
    "run_playbook",
]

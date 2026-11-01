"""Team の台帳。"""

from __future__ import annotations

import builtins
from enum import StrEnum

from pydantic import BaseModel, Field

from satsumaestro.orchestration.ids import UnknownIdError, make_id
from satsumaestro.util.clock import now_ms


class TeamStatus(StrEnum):
    CREATED = "created"
    DELETED = "deleted"


class Team(BaseModel):
    team_id: str
    name: str
    members: list[str] = Field(default_factory=list)
    status: TeamStatus = TeamStatus.CREATED
    created_at: int
    updated_at: int


class TeamLedger:
    """Team の台帳。"""

    def __init__(self) -> None:
        self._teams: dict[str, Team] = {}
        self._serial = 0

    def __len__(self) -> int:
        return len(self._teams)

    def get(self, team_id: str) -> Team | None:
        team = self._teams.get(team_id)
        return None if team is None else team.model_copy(deep=True)

    def list(self) -> builtins.list[Team]:
        return [t.model_copy(deep=True) for t in self._teams.values()]

    def create(self, name: str, members: builtins.list[str]) -> Team:
        self._serial += 1
        now = now_ms()
        team = Team(
            team_id=make_id("team", self._serial),
            name=name,
            members=list(members),
            created_at=now,
            updated_at=now,
        )
        self._teams[team.team_id] = team
        return team.model_copy(deep=True)

    def delete(self, team_id: str) -> Team:
        """論理削除（``status=deleted``）。"""
        team = self._teams.get(team_id)
        if team is None:
            raise UnknownIdError("team", team_id)
        team.status = TeamStatus.DELETED
        team.updated_at = now_ms()
        return team.model_copy(deep=True)

    def remove(self, team_id: str) -> Team | None:
        return self._teams.pop(team_id, None)

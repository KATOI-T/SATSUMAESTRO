"""Cron の台帳。"""

from __future__ import annotations

import builtins

from pydantic import BaseModel

from satsumaestro.orchestration.ids import UnknownIdError, make_id
from satsumaestro.util.clock import now_ms


class CronJob(BaseModel):
    cron_id: str
    schedule: str
    prompt: str
    description: str | None = None
    enabled: bool = True
    created_at: int
    updated_at: int
    runs: int = 0
    last_ran_at: int | None = None


class CronLedger:
    """Cron の台帳。"""

    def __init__(self) -> None:
        self._jobs: dict[str, CronJob] = {}
        self._serial = 0

    def __len__(self) -> int:
        return len(self._jobs)

    def get(self, cron_id: str) -> CronJob | None:
        job = self._jobs.get(cron_id)
        return None if job is None else job.model_copy(deep=True)

    def list(self, only_enabled: bool = False) -> builtins.list[CronJob]:
        return [
            j.model_copy(deep=True) for j in self._jobs.values() if j.enabled or not only_enabled
        ]

    def create(self, schedule: str, prompt: str, description: str | None = None) -> CronJob:
        self._serial += 1
        now = now_ms()
        job = CronJob(
            cron_id=make_id("cron", self._serial),
            schedule=schedule,
            prompt=prompt,
            description=description,
            created_at=now,
            updated_at=now,
        )
        self._jobs[job.cron_id] = job
        return job.model_copy(deep=True)

    def disable(self, cron_id: str) -> None:
        job = self._find(cron_id)
        job.enabled = False
        job.updated_at = now_ms()

    def mark_ran(self, cron_id: str) -> None:
        """1 回走ったことを記録する（回数・最後に走った時刻・更新時刻）。"""
        job = self._find(cron_id)
        now = now_ms()
        job.runs += 1
        job.last_ran_at = now
        job.updated_at = now

    def delete(self, cron_id: str) -> CronJob:
        """物理削除。"""
        job = self._jobs.pop(cron_id, None)
        if job is None:
            raise UnknownIdError("cron", cron_id)
        return job

    def _find(self, cron_id: str) -> CronJob:
        job = self._jobs.get(cron_id)
        if job is None:
            raise UnknownIdError("cron", cron_id)
        return job

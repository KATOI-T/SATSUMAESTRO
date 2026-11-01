"""スイート manifest。"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from satsumaestro.evals.loader import load_tasks
from satsumaestro.evals.models import EvalTask
from satsumaestro.i18n.text import ui
from satsumaestro.util.errors import ConfigError

PLAN_SUFFIX = "+plan"


class SuiteCase(BaseModel):
    """manifest の 1 行（課題ファイルかディレクトリへの参照）。"""

    model_config = ConfigDict(extra="forbid")

    path: str
    plan: bool = False
    capability: str = ""


class SuiteManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    name: str
    target_minutes: float | None = Field(default=None, gt=0)
    cases: list[SuiteCase] = Field(min_length=1)


@dataclass(frozen=True)
class SuiteTasks:
    """manifest を展開した課題列（通常経路と plan 経路に分けて持つ）。"""

    name: str
    target_minutes: float | None
    plain: list[EvalTask]
    plan: list[EvalTask]
    order: list[str]

    @property
    def ids(self) -> list[str]:
        return list(self.order)

    def select(self, only: set[str] | None) -> tuple[list[EvalTask], list[EvalTask]]:
        """``--only`` の課題 ID で絞る（None = 全部）。"""
        if only is None:
            return list(self.plain), list(self.plan)
        unknown = sorted(only - set(self.order))
        if unknown:
            ids = ", ".join(unknown)
            raise ConfigError(ui("evals.suite.task_ids_not_in_the", ids=ids))
        return [t for t in self.plain if t.id in only], [t for t in self.plan if t.id in only]


def is_suite_manifest(path: Path) -> bool:
    """``path`` が課題ファイルではなく manifest（``cases`` を持つ TOML）か。"""
    if not path.is_file() or path.suffix != ".toml":
        return False
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return False
    return isinstance(raw, dict) and "cases" in raw


def load_suite(path: Path) -> SuiteTasks:
    """manifest を読み、参照先の課題を展開する（ID 重複・参照不在は読込時に拒否）。"""
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise ConfigError(ui("evals.suite.cannot_read_the_manifest", path=path, error=e)) from e
    try:
        manifest = SuiteManifest.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(ui("evals.suite.invalid_manifest", path=path, error=e)) from e

    plain: list[EvalTask] = []
    plan: list[EvalTask] = []
    order: list[str] = []
    seen: set[str] = set()
    for case in manifest.cases:
        target = Path(case.path)
        if not target.is_absolute():
            target = (path.parent / target).resolve()
        for task in load_tasks(target):
            task_id = f"{task.id}{PLAN_SUFFIX}" if case.plan else task.id
            if task_id in seen:
                raise ConfigError(
                    ui("evals.suite.duplicate_task_id_in_the", task_id=task_id, path=case.path)
                )
            seen.add(task_id)
            order.append(task_id)
            if case.plan:
                plan.append(task.model_copy(update={"id": task_id}))
            else:
                plain.append(task)
    return SuiteTasks(
        name=manifest.name,
        target_minutes=manifest.target_minutes,
        plain=plain,
        plan=plan,
        order=order,
    )


def select_tasks(tasks: list[EvalTask], only: set[str] | None) -> list[EvalTask]:
    """課題列を ``--only`` の ID で絞る（None = 全部）。"""
    if only is None:
        return tasks
    unknown = sorted(only - {t.id for t in tasks})
    if unknown:
        ids = ", ".join(unknown)
        raise ConfigError(ui("evals.suite.task_ids_not_found", ids=ids))
    return [t for t in tasks if t.id in only]

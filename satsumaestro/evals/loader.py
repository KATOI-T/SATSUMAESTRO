"""評価タスクの TOML 読込。"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from pydantic import ValidationError

from satsumaestro.evals.models import EvalTask, is_valid_category
from satsumaestro.i18n.text import ui
from satsumaestro.util.errors import ConfigError


def load_tasks(target: Path) -> list[EvalTask]:
    """タスクファイル（.toml）またはディレクトリからタスク列を読み込む。"""
    if target.is_dir():
        files = sorted(target.glob("*.toml"))
        if not files:
            raise ConfigError(ui("evals.loader.no_eval_tasks_toml_found", target=target))
    elif target.is_file():
        files = [target]
    else:
        raise ConfigError(ui("evals.loader.eval_tasks_not_found", target=target))

    tasks: list[EvalTask] = []
    seen: set[str] = set()
    for path in files:
        task = _load_one(path)
        if task.id in seen:
            raise ConfigError(ui("evals.loader.duplicate_eval_task_id", id=task.id, path=path))
        seen.add(task.id)
        tasks.append(task)
    return tasks


def localize_tasks(tasks: list[EvalTask], lang: str | None) -> list[EvalTask]:
    """依頼を ``lang`` の版に替える（``None`` は題材の ``prompt`` のまま）。"""
    if lang is None:
        return tasks
    if lang != "en":
        raise ConfigError(ui("evals.loader.prompt_lang_accepts_only_en", lang=lang))
    missing = ", ".join(t.id for t in tasks if not t.prompt_en)
    if missing:
        raise ConfigError(ui("evals.loader.some_tasks_have_no_english", missing=missing))
    return [t.model_copy(update={"prompt": t.prompt_en}) for t in tasks]


def _load_one(path: Path) -> EvalTask:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise ConfigError(ui("evals.loader.cannot_read_the_eval_task", path=path, error=e)) from e
    try:
        task = EvalTask.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(ui("evals.loader.invalid_eval_task", path=path, error=e)) from e

    fixture = Path(task.fixture)
    if not fixture.is_absolute():
        fixture = (path.parent / fixture).resolve()
    if not fixture.is_dir():
        raise ConfigError(ui("evals.loader.eval_fixture_not_found", path=path, fixture=fixture))

    for pattern in task.checks.regex:
        try:
            re.compile(pattern)
        except re.error as e:
            raise ConfigError(
                ui("evals.loader.invalid_eval_regex", path=path, pattern=pattern, error=e)
            ) from e

    if task.category and not is_valid_category(task.category):
        example = '"WHERE/identifier-location"'
        raise ConfigError(
            ui(
                "evals.loader.invalid_eval_category_use_a",
                path=path,
                category=task.category,
                example=example,
            )
        )

    return task.model_copy(update={"fixture": str(fixture)})

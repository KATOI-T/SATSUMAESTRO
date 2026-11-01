"""手順書（スキル）の置き場所・一覧・名前での引き当て。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from satsumaestro.util.paths import AGENT_DIRNAME, default_config_home, find_project_file

SKILL_FILE = "SKILL.md"
PROJECT_SKILL_DIRS: tuple[str, ...] = (f"{AGENT_DIRNAME}/skills", ".claude/skills")


@dataclass(frozen=True)
class SkillMeta:
    """一覧の 1 件（system の手順書の一覧と、引き当ての結果に使う）。"""

    name: str
    description: str
    path: Path


class SkillLookupError(LookupError):
    """名前から手順書を引き当てられなかった。"""


def skill_dirs(cwd: Path, *, state_dir: bool = True) -> list[Path]:
    """実在する置き場所を、優先する順に返す。"""
    project = [
        find_project_file(cwd, rel)
        for rel in PROJECT_SKILL_DIRS
        if state_dir or Path(rel).parts[0] != AGENT_DIRNAME
    ]
    candidates = [*project, default_config_home() / "skills"]
    return [d for d in candidates if d is not None and d.is_dir()]


def read_skill_header(text: str) -> dict[str, str]:
    """先頭の ``---`` で囲んだ見出しを ``{キー: 値}`` にする。"""
    lines = text.removeprefix("﻿").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    header: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        key, sep, value = line.partition(":")
        if not sep:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1].strip()
        if value:
            header[key.strip()] = value
    return header


def list_skills(cwd: Path, *, state_dir: bool = True) -> list[SkillMeta]:
    """置き場所を優先する順に見て、手順書の一覧を返す（置き場所の中はフォルダ名の順）。"""
    found: dict[str, SkillMeta] = {}
    for directory in skill_dirs(cwd, state_dir=state_dir):
        try:
            folders = sorted(entry for entry in directory.iterdir() if entry.is_dir())
        except OSError:
            continue
        for folder in folders:
            meta = _read_skill(folder)
            if meta is not None:
                found.setdefault(meta.name.lower(), meta)
    return list(found.values())


def _read_skill(folder: Path) -> SkillMeta | None:
    path = folder / SKILL_FILE
    if not path.is_file():
        return None
    try:
        header = read_skill_header(path.read_text(encoding="utf-8"))
    except OSError:
        return None
    return SkillMeta(
        name=header.get("name", folder.name),
        description=header.get("description", ""),
        path=path,
    )


def resolve_skill(cwd: Path, skill: str, *, state_dir: bool = True) -> SkillMeta:
    """名前（大小を無視）で手順書を引き当てる。"""
    wanted = skill.strip().lstrip("/$").strip()
    if not wanted:
        raise SkillLookupError("no skill name was given")
    if any(part in wanted for part in ("/", "\\", "..")):
        raise SkillLookupError(f"a skill name cannot contain a path: {skill}")
    for meta in list_skills(cwd, state_dir=state_dir):
        if meta.name.lower() == wanted.lower():
            return meta
    raise SkillLookupError(f"there is no skill named {wanted}")

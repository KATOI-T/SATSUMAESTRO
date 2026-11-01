"""公式の評価の ai.toml を作るスクリプトと、リポジトリに入れた ai.toml。"""

from __future__ import annotations

import importlib.util
import sys
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

_REPO = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO / "scripts" / "official" / "catalog.py"
_OFFICIAL = _REPO / "evals" / "official"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("official_catalog", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["official_catalog"] = mod
    spec.loader.exec_module(mod)
    return mod


cat = _load()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.mark.parametrize(
    "text",
    [
        "one line",
        "two\nlines\n",
        "ends with a quote'\n'",
        "has ''' inside\nx",
        "cr\r\nlf",
        "del\x7f\n",
    ],
)
def test_text_survives_a_toml_round_trip(text: str) -> None:
    assert tomllib.loads(f"v = {cat.toml_text(text)}")["v"] == text


def test_polyglot_reads_the_docs_and_the_test_names_per_language(tmp_path: Path) -> None:
    for lang, test_file, body in [
        ("python", "two_fer_test.py", "def test_no_name():\n"),
        ("javascript", "two-fer.spec.js", "  xtest('a name given', () => {})\n"),
    ]:
        d = tmp_path / lang / "exercises" / "practice" / "two-fer"
        _write(d / ".meta" / "config.json", f'{{"files": {{"test": ["{test_file}"]}}}}')
        _write(d / ".docs" / "instructions.md", "Say hi.\n")
        _write(d / ".docs" / "instructions.append.md", "Extra.\n")
        _write(d / test_file, body)
    for lang in cat.POLYGLOT_LANGS:
        (tmp_path / lang / "exercises" / "practice").mkdir(parents=True, exist_ok=True)
    tasks = {str(t["id"]): t for t in cat.build_polyglot(tmp_path)}
    assert tasks["python/two-fer"]["prompt"] == "Say hi.\nExtra.\n"
    assert tasks["python/two-fer"]["expected"] == ["test_no_name"]
    assert tasks["javascript/two-fer"]["expected"] == ["a name given"]


def test_write_source_writes_one_ai_toml_and_is_stable(tmp_path: Path) -> None:
    tasks = [
        {"id": "a", "priority": "high", "prompt": "Do a.\nThen b.\n", "expected": ["t1", "t2"]},
        {"id": "b", "priority": "low", "prompt": "Do c.", "expected": []},
    ]
    source = cat.Source(
        "fake", "Fake", "https://example.invalid", "0" * 40, fetch=lambda: None,
        build=lambda: tasks, expected="test names", smoke=("a",),
    )  # fmt: skip
    cat.write_source(source, tmp_path)
    text = (tmp_path / "fake" / "ai.toml").read_text(encoding="utf-8")
    data = tomllib.loads(text)
    assert data["smoke"] == ["a"] and data["tasks"] == tasks
    cat.write_source(source, tmp_path)
    assert (tmp_path / "fake" / "ai.toml").read_text(encoding="utf-8") == text
    with pytest.raises(SystemExit, match="smoke"):
        cat.render_ai(cat.Source(**{**source.__dict__, "smoke": ("gone",)}), tasks)


def test_every_source_folder_has_only_readme_and_ai() -> None:
    for name in cat.SOURCES:
        files = {p.name for p in (_OFFICIAL / name).iterdir()}
        assert files == {"README.md", "ai.toml"}, name


def test_the_checked_in_ai_toml_match_the_script() -> None:
    for name, source in cat.SOURCES.items():
        data = tomllib.loads((_OFFICIAL / name / "ai.toml").read_text(encoding="utf-8"))
        assert data["source"] == name and data["revision"] == source.revision
        ids = [t["id"] for t in data["tasks"]]
        assert len(ids) == len(set(ids)), name
        assert {t["priority"] for t in data["tasks"]} <= {"high", "low"}
        high = {t["id"] for t in data["tasks"] if t["priority"] == "high"}
        assert set(data.get("smoke", [])) <= high, name

"""画面の文（``ui()``）と、言語ごとの文のファイル。"""

from __future__ import annotations

import ast
import re
import string
from pathlib import Path

import pytest
from typer.testing import CliRunner

from satsumaestro.cli import app as cli_app
from satsumaestro.cli.trust_gate import question_lines, untrusted_notice
from satsumaestro.config.models import LanguageConfig
from satsumaestro.i18n.schema import catalog
from satsumaestro.i18n.state import LanguageState
from satsumaestro.i18n.text import (
    PLURAL_FORMS,
    Message,
    messages,
    pinned_ui_language,
    ui,
    ui_language,
    use_language,
)

REPO = Path(__file__).resolve().parents[1]
_JAPANESE = re.compile(r"[\u3000-\u30ff\u4e00-\u9fff\uff00-\uffef]")

INPUT_VOCABULARY: dict[str, frozenset[str]] = {
    "satsumaestro/orchestration/executor.py": frozenset({".,、。"}),
    "satsumaestro/cli/repl.py": frozenset({"_CARD_QUESTIONS"}),
    "satsumaestro/security/output_guard.py": frozenset({"_REQUEST_JA"}),
    "satsumaestro/engine/plan_approval.py": frozenset(
        {"_APPROVE_JA", "_BARE_YES", "_NEGATE", "？"}
    ),
    "satsumaestro/engine/plan_request.py": frozenset({"_WANTS_PLAN"}),
    "satsumaestro/engine/plan_steps.py": frozenset(
        {"_ESTIMATE_RE", "_EXACT_TITLES", "_STEP_WORD_RE", "_RESUME_RE", "？"}
    ),
    "satsumaestro/engine/reply_checks.py": frozenset({"_COMPLETION_RE"}),
    "satsumaestro/orchestration/decompose.py": frozenset(
        {
            "_SENTENCE_SPLIT",
            "_PARALLEL_MARKERS",
            "_ENUM_LINE",
            "_WRITE_VERBS",
            "_TRAILING_NOISE",
            "、",
        }
    ),
    "satsumaestro/tools/builtins/todo_write.py": frozenset({"_CHECK_WORDS"}),
}


def _is_ui_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (isinstance(func, ast.Name) and func.id == "ui") or (
        isinstance(func, ast.Attribute) and func.attr == "ui"
    )


def _has_japanese(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str) and bool(_JAPANESE.search(node.value))
    if isinstance(node, ast.JoinedStr):
        return any(_has_japanese(v) for v in node.values)
    return False


def screen_literals(path: Path) -> list[tuple[int, str]]:
    """docstring と入力の照合の語彙を除いた、日本語を含む文字列リテラル（f-string は 1 つ）。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    rel = path.relative_to(REPO).as_posix() if path.is_relative_to(REPO) else ""
    vocabulary = INPUT_VOCABULARY.get(rel, frozenset())
    skipped: set[int] = set()
    for statement in tree.body:
        targets = (
            statement.targets
            if isinstance(statement, ast.Assign)
            else [statement.target]
            if isinstance(statement, ast.AnnAssign)
            else []
        )
        names = {n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)}
        if names & vocabulary:
            skipped.update(id(n) for n in ast.walk(statement))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value in vocabulary:
            skipped.add(id(node))
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            first = node.body[0] if node.body else None
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                skipped.add(id(first.value))
        if isinstance(node, ast.JoinedStr):
            skipped.update(id(v) for v in node.values)
    found = [
        (getattr(node, "lineno", 0), ast.unparse(node)[:60])
        for node in ast.walk(tree)
        if id(node) not in skipped
        and isinstance(node, ast.Constant | ast.JoinedStr)
        and _has_japanese(node)
    ]
    return sorted(found)


def placeholders(message: Message) -> set[str]:
    """文が埋める値の名前（数で変わる文は、どの形の分も合わせる）。"""
    forms = [message] if isinstance(message, str) else list(message.values())
    return {
        field.split(".")[0].split("[")[0]
        for text in forms
        for _literal, field, _spec, _conversion in string.Formatter().parse(text)
        if field is not None
    }


def ui_call_problems(path: Path) -> list[str]:
    """``ui()`` の呼び方の誤り（キーが無い・渡す値の名前が文の埋める値と違う）。"""
    english = messages("en")
    problems: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not _is_ui_call(node):
            continue
        assert isinstance(node, ast.Call)
        where = f"{path.name}:{node.lineno}"
        key = node.args[0] if node.args else None
        if not (isinstance(key, ast.Constant) and isinstance(key.value, str)) or len(node.args) > 1:
            problems.append(f"{where}: キーを 1 つ文字列で渡す")
            continue
        if key.value not in english:
            problems.append(f"{where}: キーが無い: {key.value}")
            continue
        names = {kw.arg for kw in node.keywords}
        if None in names:
            problems.append(f"{where}: 値は名前で 1 つずつ渡す")
            continue
        wanted = placeholders(english[key.value])
        if names != wanted:
            problems.append(
                f"{where}: {key.value} の値の名前が違う: {sorted(names)} / {sorted(wanted)}"
            )
    return problems


def _sources() -> list[Path]:
    return sorted((REPO / "satsumaestro").rglob("*.py"))


def test_no_japanese_strings_in_the_code() -> None:
    found = {
        str(path.relative_to(REPO)): lines
        for path in _sources()
        if (lines := screen_literals(path))
    }
    assert found == {}


def test_ui_calls_use_existing_keys_and_values() -> None:
    assert [p for path in _sources() for p in ui_call_problems(path)] == []


def test_every_screen_language_has_every_key() -> None:
    english = messages("en")
    assert english
    for spec in catalog().languages:
        if not spec.ui:
            continue
        texts = messages(spec.code)
        assert set(texts) == set(english), spec.code
        for key, text in texts.items():
            assert placeholders(text) == placeholders(english[key]), (spec.code, key)
            if not isinstance(text, str):
                assert set(text) <= PLURAL_FORMS, (spec.code, key)


def test_english_texts_have_no_japanese() -> None:
    for key, text in messages("en").items():
        forms = [text] if isinstance(text, str) else list(text.values())
        assert not any(_JAPANESE.search(form) for form in forms), key


def test_every_key_is_used() -> None:
    used = {
        node.value
        for path in _sources()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert sorted(set(messages("en")) - used) == []


def test_the_check_finds_japanese_strings(tmp_path: Path) -> None:
    sample = tmp_path / "sample.py"
    sample.write_text(
        '"""日本語の docstring は数えない。"""\n'
        'A = ui("cli.render.done")\n'
        'C = "日本語だけの文"\n'
        'D = f"{n} 件"\n'
        'F = f"（{n}）"\n',
        encoding="utf-8",
    )
    assert [line for line, _text in screen_literals(sample)] == [3, 4, 5]


def test_the_check_finds_wrong_keys_and_values(tmp_path: Path) -> None:
    sample = tmp_path / "sample.py"
    sample.write_text(
        'A = ui("no.such.key")\n'
        'B = ui("cli.render.thinking", extra=1)\n'
        'C = ui("cli.app.trusted_recorded_in", root=1)\n'
        "D = ui(key)\n",
        encoding="utf-8",
    )
    problems = ui_call_problems(sample)
    assert any("キーが無い: no.such.key" in p for p in problems)
    assert any("cli.render.thinking の値の名前が違う: ['extra'] / []" in p for p in problems)
    assert any("['root'] / ['record', 'root']" in p for p in problems)
    assert any("キーを 1 つ文字列で渡す" in p for p in problems)


def test_ui_returns_the_text_of_the_screen_language() -> None:
    assert ui("cli.render.done") == "✓ 完了"
    with pinned_ui_language("en"):
        assert ui("cli.render.done") == "✓ Done"
    with pinned_ui_language("ko"):
        assert ui("cli.render.done") == "✓ Done"
    with pytest.raises(KeyError, match="no screen text"):
        ui("no.such.key")


def test_texts_that_change_with_the_count() -> None:
    key = "cli.render.lines_omitted_d_for_full"
    with pinned_ui_language("en"):
        assert ui(key, n=1) == "… (1 line omitted; d for full text)"
        assert ui(key, n=3) == "… (3 lines omitted; d for full text)"
    assert ui(key, n=1) == "… (1 行省略、d で全文)"


def test_ui_follows_the_conversation_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANG", "ja_JP.UTF-8")
    monkeypatch.delenv("LC_ALL", raising=False)
    monkeypatch.delenv("LC_MESSAGES", raising=False)
    with pinned_ui_language(None):
        assert ui_language() == "ja"
        state = LanguageState.from_config(LanguageConfig(), {"LANG": "ja_JP.UTF-8"})
        use_language(state)
        assert ui_language() == "ja"
        state.begin_turn("Where is with_tax called from in this project?")
        assert ui("cli.render.done") == "✓ Done"
        state.begin_turn("with_tax はどこで呼ばれていますか")
        assert ui("cli.render.done") == "✓ 完了"
        use_language(None)
        monkeypatch.setenv("LANG", "C.UTF-8")
        assert ui_language() == "en"


def test_the_trust_question_in_english(tmp_path: Path) -> None:
    with pinned_ui_language("en"):
        lines = [line.plain for line in question_lines(tmp_path)]
        notice = untrusted_notice(tmp_path)
    assert lines[0] == f"Opening this directory for the first time: {tmp_path}"
    assert ".satsuma/, .satsuma.json, .env, policy.md, memory.md" in lines[1]
    assert lines[3].startswith("[y] trust and remember")
    assert notice.startswith(f"Untrusted directory: {tmp_path}.")
    assert all(not _JAPANESE.search(text) for text in [*lines, notice])
    assert question_lines(tmp_path)[0].plain == f"このディレクトリを初めて開きます: {tmp_path}"


def test_the_trust_command_in_english(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SATSUMA_CONFIG_HOME", str(tmp_path / "home"))
    runner = CliRunner()
    with pinned_ui_language("en"):
        listed = runner.invoke(cli_app.app, ["trust", "list"])
        added = runner.invoke(cli_app.app, ["trust", "add", "--cwd", str(tmp_path)])
        removed = runner.invoke(cli_app.app, ["trust", "remove", "--cwd", str(tmp_path)])
        missing = runner.invoke(cli_app.app, ["trust", "remove", "--cwd", str(tmp_path)])
    assert listed.stdout.strip() == "(no trusted directories)"
    assert added.stdout.startswith(f"Trusted: {tmp_path.resolve()} (recorded in ")
    assert removed.stdout.strip() == f"Removed: {tmp_path.resolve()}"
    assert missing.exit_code == 1 and f"Not recorded: {tmp_path.resolve()}" in missing.output

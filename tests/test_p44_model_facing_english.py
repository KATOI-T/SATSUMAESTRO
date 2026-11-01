"""モデルが読む文は英語だけにする（画面の言語に依らない）。"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from satsumaestro.config.models import (
    HookConfig,
    LanguageConfig,
    RuntimeConfig,
    SubagentConfig,
    ToolsConfig,
)
from satsumaestro.engine.subagent import RuntimeFactory, resolve_subagent_type
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.tools import default_registry
from tests.fixtures.mock_provider import MockProvider

REPO = Path(__file__).resolve().parents[1]
_JAPANESE = re.compile(r"[\u3000-\u30ff\u4e00-\u9fff\uff00-\uffef]")

MODEL_FACING: dict[str, frozenset[str]] = {
    "satsumaestro/security/deletion_guard.py": frozenset(),
    "satsumaestro/security/output_guard.py": frozenset(
        {"card_note", "_names", "_label", "_REQUEST_JA"}
    ),
    "satsumaestro/security/self_invocation_guard.py": frozenset(),
    "satsumaestro/security/state_write_guard.py": frozenset(),
    "satsumaestro/orchestration/executor.py": frozenset({"extract_evidence"}),
    "satsumaestro/tools/builtins/web_search.py": frozenset(),
    "satsumaestro/mcp/manager.py": frozenset(),
    "satsumaestro/tools/builtins/lsp.py": frozenset(),
}


def _japanese_literals(path: Path, allowed: frozenset[str]) -> list[tuple[int, str]]:
    """docstring・ログ・``allowed`` の名前の中を除いた、日本語を含む文字列リテラル。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    skipped: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body:
            first = body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                skipped.add(id(first.value))
        inside: ast.AST | None = None
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name in allowed:
            inside = node
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in allowed for t in node.targets
        ):
            inside = node
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"debug", "info", "warning", "error", "exception"}
        ):
            inside = node
        if inside is not None:
            skipped.update(id(n) for n in ast.walk(inside))
    return sorted(
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in skipped
        and _JAPANESE.search(node.value)
    )


@pytest.mark.parametrize("relpath", sorted(MODEL_FACING))
def test_text_the_model_reads_is_english(relpath: str) -> None:
    assert _japanese_literals(REPO / relpath, MODEL_FACING[relpath]) == []


def test_the_check_finds_japanese_literals(tmp_path: Path) -> None:
    sample = tmp_path / "sample.py"
    sample.write_text(
        '"""日本語の docstring は対象外。"""\n'
        "import logging\n_log = logging.getLogger(__name__)\n"
        'MESSAGE = "削除できません"\n'
        'def screen():\n    return "画面の文"\n'
        'def model():\n    _log.info("ログは対象外")\n    return "モデルの文"\n',
        encoding="utf-8",
    )
    found = [text for _line, text in _japanese_literals(sample, frozenset({"screen"}))]
    assert found == ["削除できません", "モデルの文"]


def _factory(tmp_path: Path, language: LanguageConfig | None = None) -> RuntimeFactory:
    config = RuntimeConfig(
        subagent=SubagentConfig(default_model="mock-model"),
        tools=ToolsConfig(deferred_tool_loading=False),
        language=language or LanguageConfig(),
    )
    provider = MockProvider([])
    return RuntimeFactory(
        provider=provider,
        model="mock-model",
        config=config,
        tools=default_registry(),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        hooks=HookRunner(HookConfig(), tmp_path),
        cwd=tmp_path,
        provider_builder=lambda model: provider,
    )


_PARENT_JA = "with_tax はどこで呼ばれていますか"
_CHILD_PROMPT = 'Find the callers of with_tax.\n\n# Report format\nWrite "unknown" if unsure.'


def test_a_child_answers_in_the_language_of_the_parent_turn(tmp_path: Path) -> None:
    factory = _factory(tmp_path)
    factory.language.begin_turn(_PARENT_JA)
    child = factory.build_child(preset=resolve_subagent_type("Explore"), depth=1)
    assert (child.language.mode, child.language.fixed) == ("fixed", "ja")
    assert child.session.language == "ja"
    assert child.language.begin_turn(_CHILD_PROMPT) == "ja"
    assert factory.language.current == "ja"
    assert factory.config.language.mode == "auto"


def test_a_child_keeps_detecting_when_the_parent_language_is_unknown(tmp_path: Path) -> None:
    factory = _factory(tmp_path)
    child = factory.build_child(preset=resolve_subagent_type("Explore"), depth=1)
    assert child.language.mode == "auto"
    assert child.language.begin_turn(_CHILD_PROMPT) == "en"
    assert factory.language.current == ""


def test_a_child_keeps_the_language_control_off(tmp_path: Path) -> None:
    factory = _factory(tmp_path, LanguageConfig(mode="off"))
    factory.language.begin_turn(_PARENT_JA)
    child = factory.build_child(preset=resolve_subagent_type("Explore"), depth=1)
    assert child.language.mode == "off"

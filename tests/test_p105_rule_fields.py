"""権限の規則の ``path`` は、notebook_edit では入力の ``notebook_path`` に当たる。"""

from __future__ import annotations

from pathlib import Path

from satsumaestro.security.allowlist import MemoryLayer, PermissionResolver, signature_for
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import PermissionPolicy, RuleMatch, field_value
from satsumaestro.security.policy_markdown import parse_policy_markdown

WW = PermissionMode.WORKSPACE_WRITE
NB = {"notebook_path": "secret/report.ipynb", "new_source": "x = 1"}


def _policy(body: str) -> PermissionPolicy:
    parsed = parse_policy_markdown("---\nmode: workspace-write\n---\n" + body)
    assert parsed.warnings == []
    return parsed.policy


def test_path_reads_the_notebook_path() -> None:
    assert field_value(NB, "path") == "secret/report.ipynb"
    assert field_value({"path": "a.py", "notebook_path": "b.ipynb"}, "path") == "a.py"
    assert field_value({"command": "ls"}, "path") is None


def test_a_path_deny_stops_notebook_edit() -> None:
    policy = _policy("## Deny\n- notebook_edit when path glob: secret/*\n")
    outcome = policy.authorize("notebook_edit", NB, WW)
    assert outcome.decision == "deny" and "deny rule" in outcome.reason
    assert policy.authorize("notebook_edit", {"notebook_path": "nb/a.ipynb"}, WW).decision != (
        "deny"
    )


def test_a_deny_for_every_tool_covers_notebook_edit() -> None:
    policy = _policy("## Deny\n- * when path glob: secret/**\n")
    assert policy.authorize("notebook_edit", NB, WW).decision == "deny"
    assert policy.authorize("write_file", {"path": "secret/a.txt"}, WW).decision == "deny"


def test_a_path_allow_for_a_notebook_is_scoped() -> None:
    policy = _policy("## Allow\n- notebook_edit when path glob: secret/*.ipynb\n")
    outcome = policy.authorize("notebook_edit", NB, WW)
    assert outcome.decision == "allow" and outcome.scoped


def test_remembering_a_notebook_edit_names_the_notebook(tmp_path: Path) -> None:
    rule = signature_for("notebook_edit", NB)
    assert rule == RuleMatch(tool="notebook_edit", field="path", regex=r"\Asecret/report\.ipynb\Z")
    memory = tmp_path / "memory.md"
    resolver = PermissionResolver(
        PermissionPolicy(mode=PermissionMode.READ_ONLY), [MemoryLayer(memory, writable=True)]
    )
    resolver.record("notebook_edit", NB)
    assert resolver.session.matches("notebook_edit", NB)
    assert not resolver.session.matches("notebook_edit", {"notebook_path": "other.ipynb"})
    assert "notebook_edit when path re: " in memory.read_text(encoding="utf-8")


def test_web_search_is_remembered_per_tool() -> None:
    assert signature_for("web_search", {"query": "weather"}) == RuleMatch(tool="web_search")

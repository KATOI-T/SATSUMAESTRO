"""RuleMatch のマッチングテスト。"""

from __future__ import annotations

from satsumaestro.security import RuleMatch


def test_command_glob() -> None:
    m = RuleMatch(tool="bash", field="command", glob="npm test*")
    assert m.matches("bash", {"command": "npm test"})
    assert m.matches("bash", {"command": "npm test -v"})
    assert not m.matches("bash", {"command": "rm -rf /"})
    assert not m.matches("read_file", {"command": "npm test"})


def test_path_glob_recursive() -> None:
    m = RuleMatch(tool="write_file", field="path", glob="**/.git/**")
    assert m.matches("write_file", {"path": "a/.git/config"})
    assert m.matches("write_file", {"path": ".git/config"})
    assert not m.matches("write_file", {"path": "src/app.py"})


def test_path_glob_prefix() -> None:
    m = RuleMatch(tool="write_file", field="path", glob="src/**")
    assert m.matches("write_file", {"path": "src/app.py"})
    assert m.matches("write_file", {"path": "src/pkg/mod.py"})
    assert not m.matches("write_file", {"path": "lib/app.py"})


def test_star_wildcard_no_slash() -> None:
    m = RuleMatch(tool="read_file", field="path", glob="src/*.py")
    assert m.matches("read_file", {"path": "src/app.py"})
    assert not m.matches("read_file", {"path": "src/pkg/app.py"})


def test_command_regex_anchored() -> None:
    m = RuleMatch(tool="bash", field="command", regex=r"^\s*rm\s+-rf\s+/")
    assert m.matches("bash", {"command": "rm -rf /"})
    assert m.matches("bash", {"command": "  rm  -rf  /tmp"})
    assert not m.matches("bash", {"command": "echo rm -rf /"})


def test_url_field() -> None:
    m = RuleMatch(tool="web_fetch", field="url", glob="https://internal/**")
    assert m.matches("web_fetch", {"url": "https://internal/secret"})
    assert not m.matches("web_fetch", {"url": "https://example.com/x"})


def test_star_tool_matches_any() -> None:
    m = RuleMatch(tool="*", field="path", glob="**/secrets/**")
    assert m.matches("edit_file", {"path": "a/secrets/key"})
    assert m.matches("read_file", {"path": "secrets/key"})


def test_tool_only_rule_matches_any_input() -> None:
    m = RuleMatch(tool="read_file")
    assert m.matches("read_file", {"path": "anything"})
    assert m.matches("read_file", {})


def test_missing_field_value_is_no_match() -> None:
    m = RuleMatch(tool="bash", field="command", glob="*")
    assert not m.matches("bash", {})
    assert not m.matches("bash", {"command": 123})


def test_invalid_regex_is_no_match() -> None:
    m = RuleMatch(tool="bash", field="command", regex="[unclosed")
    assert not m.matches("bash", {"command": "anything"})


def test_regex_timeout_fails_safe_to_no_match() -> None:
    from satsumaestro.security.policy import _regex_search

    class _SlowPattern:
        pattern = "(evil)"

        def search(self, text: str, timeout: float) -> object:
            raise TimeoutError("simulated ReDoS")

    assert _regex_search(_SlowPattern(), "anything") is False  # type: ignore[arg-type]


def test_field_without_matcher_is_no_match() -> None:
    m = RuleMatch(tool="bash", field="command")
    assert not m.matches("bash", {"command": "anything"})


def test_matcher_without_field_is_no_match() -> None:
    m = RuleMatch(tool="bash", glob="*")
    assert not m.matches("bash", {"command": "anything"})

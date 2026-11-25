"""``redaction.live`` の画面: 変更のプレビューは元の値で作り、出す行だけを伏せる。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console, RenderableType

from satsumaestro.cli.render import RichRenderer, change_preview
from satsumaestro.engine.events import ToolUse, ToolUseDelta
from satsumaestro.session.models import ToolResultBlock
from satsumaestro.session.redact import redact_lines

_KEY = "plain-looking-provider-key-0123456789"
_BEARER = "bearer-value-0123456789"
_BODY = ("MIIBVwIBADANBgkqhkiG9w0BAQEFAASCAUEwggE9AgEAAkEA", "q1w2e3r4t5y6u7i8o9p0a1s2d3f4g5h6")
_PEM = "-----BEGIN PRIVATE KEY-----\n" + "\n".join(_BODY) + "\n-----END PRIVATE KEY-----\n"
_KEY_LINES = "\n".join(_BODY)
_CLASS = 'class A:\n    def f(self):\n        API_KEY = "abcd12345678"\n        return API_KEY\n'


@dataclass(frozen=True)
class _Case:
    id: str
    tool: str
    inp: dict[str, object]
    before: str | bytes | None
    shown: tuple[str, ...]
    hidden: tuple[str, ...]
    extra: dict[str, object] = field(default_factory=dict)


_CASES = [
    _Case(
        "edit-rotation",
        "edit_file",
        {"old": "API_KEY=oldsecret123", "new": "API_KEY=newsecret456"},
        "API_KEY=oldsecret123\nname=demo\n",
        ("1 -API_KEY=[redacted]", "1 +API_KEY=[redacted]"),
        ("oldsecret123", "newsecret456"),
    ),
    _Case(
        "edit-same-looking-lines",
        "edit_file",
        {"old": "API_KEY=bbbb33334444", "new": "API_KEY=cccc55556666"},
        "API_KEY=aaaa11112222\nname=demo\nAPI_KEY=bbbb33334444\n",
        (" 1  API_KEY=[redacted]", "3 -API_KEY=[redacted]", "3 +API_KEY=[redacted]"),
        ("aaaa11112222", "bbbb33334444", "cccc55556666", " 1 -"),
    ),
    _Case(
        "edit-replace-all",
        "edit_file",
        {"old": "API_KEY=k1k1k1k1k1", "new": "API_KEY=k2k2k2k2k2", "replace_all": True},
        "API_KEY=k1k1k1k1k1\nname=demo\nAPI_KEY=k1k1k1k1k1\n",
        ("1 -API_KEY=[redacted]", "1 +API_KEY=[redacted]", "3 -API_KEY=[redacted]"),
        ("k1k1k1k1k1", "k2k2k2k2k2"),
    ),
    _Case(
        "edit-reindent",
        "edit_file",
        {
            "old": 'def f(self):\n    API_KEY = "abcd12345678"',
            "new": 'def f(self):\n    API_KEY = "wxyz98765432"',
        },
        _CLASS,
        ('3 -        API_KEY = "[redacted]"', '3 +        API_KEY = "[redacted]"'),
        ("abcd12345678", "wxyz98765432"),
    ),
    _Case(
        "edit-old-not-found",
        "edit_file",
        {"old": "API_KEY=zzzz99998888", "new": "API_KEY=yyyy77776666"},
        "name=demo\n",
        ("-API_KEY=[redacted]", "+API_KEY=[redacted]"),
        ("zzzz99998888", "yyyy77776666"),
    ),
    _Case(
        "write-rotation",
        "write_file",
        {"content": "API_KEY=newsecret456\nname=demo\n"},
        "API_KEY=oldsecret123\nname=demo\n",
        ("1 -API_KEY=[redacted]", "1 +API_KEY=[redacted]"),
        ("oldsecret123", "newsecret456"),
    ),
    _Case(
        "write-new-file",
        "write_file",
        {"content": "API_KEY=newsecret456\nname=demo\n"},
        None,
        (" 1 API_KEY=[redacted]", " 2 name=demo"),
        ("newsecret456",),
    ),
    _Case(
        "write-unreadable-file",
        "write_file",
        {"content": "API_KEY=newsecret456\nname=demo\n"},
        b"\xff\xfeAPI_KEY=oldsecret123",
        ("API_KEY=[redacted]",),
        ("newsecret456", "oldsecret123"),
    ),
    _Case(
        "append-existing-file",
        "append_file",
        {"content": "API_KEY=newsecret456\n"},
        "name=demo\n",
        (" 1  name=demo", " 2 +API_KEY=[redacted]"),
        ("newsecret456",),
    ),
    _Case(
        "append-new-file",
        "append_file",
        {"content": "API_KEY=newsecret456\n"},
        None,
        (" 1 API_KEY=[redacted]",),
        ("newsecret456",),
    ),
    _Case(
        "pem-in-context",
        "edit_file",
        {"old": "tail=1", "new": "tail=2"},
        "name=demo\n" + _PEM + "tail=1\n",
        (" 3  [redacted]", " 4  [redacted]", " 5  [redacted]", " 6 -tail=1", " 6 +tail=2"),
        (*_BODY, "PRIVATE KEY"),
    ),
    _Case(
        "pem-in-new-file",
        "write_file",
        {"content": "name=demo\n" + _PEM + "tail=1\n"},
        None,
        (" 2 [redacted]", " 5 [redacted]", " 6 tail=1"),
        (*_BODY, "PRIVATE KEY"),
    ),
    _Case(
        "pem-wrapped-by-the-new-content",
        "write_file",
        {"content": "name=demo\n" + _PEM + "tail=1\n"},
        f"name=demo\n{_KEY_LINES}\ntail=1\n",
        (" 2 +[redacted]", " 3  [redacted]", " 4  [redacted]", " 5 +[redacted]"),
        (*_BODY, "PRIVATE KEY"),
    ),
    _Case(
        "pem-unwrapped-by-the-new-content",
        "write_file",
        {"content": f"name=demo\n{_KEY_LINES}\ntail=1\n"},
        "name=demo\n" + _PEM + "tail=1\n",
        (" 2 -[redacted]", " 2  [redacted]", " 3  [redacted]", " 5 -[redacted]"),
        (*_BODY, "PRIVATE KEY"),
    ),
    _Case(
        "edit-old-not-found-wraps-pem",
        "edit_file",
        {"old": _KEY_LINES, "new": _PEM.rstrip("\n")},
        "name=demo\n",
        ("+[redacted]", " [redacted]"),
        (*_BODY, "PRIVATE KEY"),
    ),
    _Case(
        "edit-old-not-found-unwraps-pem",
        "edit_file",
        {"old": _PEM.rstrip("\n"), "new": _KEY_LINES},
        "name=demo\n",
        ("-[redacted]", " [redacted]"),
        (*_BODY, "PRIVATE KEY"),
    ),
    _Case(
        "pem-with-crlf",
        "edit_file",
        {"old": "tail=1", "new": "tail=2"},
        ("name=demo\n" + _PEM + "tail=1\n").replace("\n", "\r\n"),
        (" 3  [redacted]", " 4  [redacted]", " 5  [redacted]", " 6 -tail=1", " 6 +tail=2"),
        (*_BODY, "PRIVATE KEY"),
    ),
    _Case(
        "authorization-split-over-lines",
        "edit_file",
        {"old": "name=demo", "new": "name=demo2"},
        "Authorization: Bearer\n    abcdefgh12345678\nname=demo\n",
        (" 2      [redacted]", " 3 -name=demo", " 3 +name=demo2"),
        ("abcdefgh12345678",),
    ),
    _Case(
        "key-value-without-a-name",
        "edit_file",
        {"old": "name=demo", "new": "name=demo2"},
        f"provider {_KEY}\nname=demo\n",
        (" 1  provider [redacted]", " 2 -name=demo", " 2 +name=demo2"),
        (_KEY,),
    ),
    _Case(
        "value-with-shell-characters",
        "edit_file",
        {"old": "name=demo", "new": "name=demo2"},
        "DB_PASSWORD=abc$example-secret\nname=demo\n",
        (" 1  DB_PASSWORD=[redacted]",),
        ("example-secret",),
    ),
]
_IDS = [case.id for case in _CASES]


@pytest.fixture
def ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("ANTHROPIC_API_KEY", _KEY)
    return tmp_path


def _call(ws: Path, case: _Case) -> ToolUse:
    target = ws / "conf.env"
    if isinstance(case.before, str):
        target.write_text(case.before, encoding="utf-8")
    elif isinstance(case.before, bytes):
        target.write_bytes(case.before)
    return ToolUse(id="p", name=case.tool, input={"path": "conf.env", **case.inp})


def _text(renderable: RenderableType | None) -> str:
    assert renderable is not None
    out = StringIO()
    Console(file=out, width=200, color_system=None).print(renderable)
    return out.getvalue()


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_previews_when_on(ws: Path, case: _Case) -> None:
    text = _text(change_preview(_call(ws, case), ws, redact=True))
    assert [s for s in case.shown if s not in text] == []
    assert [s for s in case.hidden if s in text] == []


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_colored_previews_hide_the_values_too(ws: Path, case: _Case) -> None:
    out = StringIO()
    console = Console(file=out, width=200, force_terminal=True, color_system="truecolor")
    console.print(change_preview(_call(ws, case), ws, redact=True))
    text = re.sub(r"\x1b\[[0-9;]*m", "", out.getvalue())
    assert [s for s in case.hidden if s in text] == []


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_previews_when_off_show_the_values(ws: Path, case: _Case) -> None:
    text = _text(change_preview(_call(ws, case), ws))
    assert "[redacted]" not in text
    assert any(s in text for s in case.hidden)


def _shape(text: str, *, marked: bool) -> list[str]:
    """行番号と記号（新規ファイルの行には記号が無い）。"""
    pattern = r"\s*\d*\s?[-+@]?" if marked else r"\s*\d*\s?"
    return [m.group(0) for line in text.splitlines() if (m := re.match(pattern, line))]


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_on_and_off_differ_only_in_values(ws: Path, case: _Case) -> None:
    call = _call(ws, case)
    on = _text(change_preview(call, ws, redact=True))
    off = _text(change_preview(call, ws))
    marked = case.before is not None
    assert _shape(on, marked=marked) == _shape(off, marked=marked)


@pytest.mark.parametrize(
    ("text", "values", "expected"),
    [
        ("", (), []),
        ("a\nb", (), ["a", "b"]),
        ("API_KEY=abcd1234\r\nb\r\n", (), ["API_KEY=[redacted]", "b"]),
        (_PEM, (), ["[redacted]"] * 4),
        (_PEM.replace("\n", "\u2028") + "tail", (), ["[redacted]"] * 4 + ["tail"]),
        (f"x\n{_PEM}\n", (), ["x", *["[redacted]"] * 4, ""]),
        (
            "x\nmulti-line\nsecret-value\ny",
            ("multi-line\nsecret-value",),
            ["x", *["[redacted]"] * 2, "y"],
        ),
        ("Authorization: Bearer\n  abcd1234", (), ["Authorization: Bearer", "  [redacted]"]),
        (f"k {_KEY}\nplain", (_KEY,), ["k [redacted]", "plain"]),
    ],
)
def test_redact_lines_keeps_the_line_count(
    text: str, values: tuple[str, ...], expected: list[str]
) -> None:
    assert redact_lines(text, values=values) == expected
    assert len(expected) == len(text.splitlines())


def _renderer(ws: Path, *, redact: bool) -> tuple[RichRenderer, StringIO]:
    buf = StringIO()
    console = Console(file=buf, force_terminal=True, color_system=None, width=200)
    r = RichRenderer(console, cwd=ws, redact=redact)
    r.start_turn()
    return r, buf


_CURL = ToolUse(
    id="b1",
    name="bash",
    input={"command": f"curl -H 'Authorization: Bearer {_BEARER}' https://example.invalid"},
)


@pytest.mark.parametrize("redact", [True, False])
def test_the_tool_line_is_redacted_by_the_renderer(ws: Path, redact: bool) -> None:
    r, buf = _renderer(ws, redact=redact)
    r.on_tool_start(_CURL)
    r.on_tool_result(ToolResultBlock(tool_use_id="b1", tool_name="bash", output="ok"))
    assert (_BEARER in buf.getvalue()) is not redact
    assert ("Bearer [redacted]" in buf.getvalue()) is redact


@pytest.mark.parametrize("redact", [True, False])
def test_the_renderer_previews_from_the_real_call(ws: Path, redact: bool) -> None:
    r, buf = _renderer(ws, redact=redact)
    r.on_tool_start(_call(ws, _CASES[0]))
    out = buf.getvalue()
    if redact:
        assert "1 -API_KEY=[redacted]" in out and "1 +API_KEY=[redacted]" in out
        assert "oldsecret123" not in out and "newsecret456" not in out
    else:
        assert "1 -API_KEY=oldsecret123" in out and "1 +API_KEY=newsecret456" in out


def test_the_renderer_reads_the_keys_when_it_draws(
    ws: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    r, buf = _renderer(ws, redact=True)
    late = "later-provider-key-0123456789"
    monkeypatch.setenv("ANTHROPIC_API_KEY", late)
    r.on_tool_start(ToolUse(id="b2", name="bash", input={"command": f"echo {late}"}))
    r.on_tool_result(ToolResultBlock(tool_use_id="b2", tool_name="bash", output="ok"))
    assert late not in buf.getvalue() and "echo [redacted]" in buf.getvalue()


def test_the_renderer_does_not_stream_tool_arguments_when_it_redacts() -> None:
    console = Console(file=StringIO(), force_terminal=True, width=70)
    r = RichRenderer(console, stream_tool_args=True, redact=True)
    r.start_turn()
    args = json.dumps({"path": "a.txt", "content": "streamed-content"})
    r.on_assistant_event(ToolUseDelta(index=0, name="write_file", args_delta=args))
    out = StringIO()
    Console(file=out, force_terminal=False, width=70).print(r._live_renderable())
    assert "streamed-content" not in out.getvalue()

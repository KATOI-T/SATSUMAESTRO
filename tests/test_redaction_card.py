"""``redaction.live`` の確認のカード: 差分と行数は元の値で数え、出す文字だけを伏せる。"""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO

import pytest
from rich.console import Console, RenderableType

from satsumaestro.cli.render import action_preview
from satsumaestro.cli.repl import InteractivePrompter
from satsumaestro.engine.events import ToolUse
from satsumaestro.engine.turn import PermissionDetail
from satsumaestro.i18n.text import ui
from satsumaestro.security.enforcer import PermissionMode

_KEY = "plain-looking-provider-key-0123456789"
_TOKEN = "ghp_" + "a1B2c3D4e5" * 3 + "f6G7h8"
_BEARER = "bearer-value-0123456789"
_BODY = ("MIIBVwIBADANBgkqhkiG9w0BAQEFAASCAUEwggE9AgEAAkEA", "q1w2e3r4t5y6u7i8o9p0a1s2d3f4g5h6")
_PEM = "-----BEGIN PRIVATE KEY-----\n" + "\n".join(_BODY) + "\n-----END PRIVATE KEY-----\n"
_KEY_LINES = "\n".join(_BODY)
_PEM_FILE = "name=demo\n" + _PEM + "tail=1\n"


@dataclass(frozen=True)
class _Card:
    id: str
    tool: str
    inp: dict[str, object]
    short: tuple[str, ...]
    full: tuple[str, ...]
    hidden: tuple[str, ...]


_CARDS = [
    _Card(
        "bash-quoted-password",
        "bash",
        {"command": "PASSWORD='ab$private12345' ./login.sh"},
        ("PASSWORD='[redacted]' ./login.sh",),
        ("PASSWORD='[redacted]' ./login.sh",),
        ("private12345",),
    ),
    _Card(
        "bash-after-an-expansion",
        "bash",
        {"command": 'PASSWORD="${PREFIX}private12345" ./login.sh'},
        ('PASSWORD="${PREFIX}[redacted]" ./login.sh',),
        ('PASSWORD="${PREFIX}[redacted]" ./login.sh',),
        ("private12345",),
    ),
    _Card(
        "bash-what-runs",
        "bash",
        {"command": "API_KEY=$(./payload.sh) make test"},
        ("API_KEY=$(./payload.sh) make test",),
        ("API_KEY=$(./payload.sh) make test",),
        (),
    ),
    _Card(
        "bash-bearer",
        "bash",
        {"command": f"curl -H 'Authorization: Bearer {_BEARER}' https://example.invalid"},
        ("Bearer [redacted]",),
        ("Bearer [redacted]",),
        (_BEARER,),
    ),
    _Card(
        "edit-rotation",
        "edit_file",
        {"path": "conf.env", "old": "API_KEY=oldsecret123", "new": "API_KEY=newsecret456"},
        ("conf.env  +1 -1",),
        ("-API_KEY=[redacted]", "+API_KEY=[redacted]"),
        ("oldsecret123", "newsecret456"),
    ),
    _Card(
        "edit-next-to-a-key",
        "edit_file",
        {
            "path": "conf.env",
            "old": "API_KEY=aaaa11112222\nname=demo",
            "new": "API_KEY=aaaa11112222\nname=demo2",
        },
        ("conf.env  +1 -1",),
        (" API_KEY=[redacted]", "-name=demo", "+name=demo2"),
        ("aaaa11112222",),
    ),
    _Card(
        "edit-wraps-pem",
        "edit_file",
        {"path": "k.txt", "old": _KEY_LINES, "new": _PEM.rstrip("\n")},
        ("k.txt  +2 -0",),
        ("+[redacted]", " [redacted]"),
        (*_BODY, "PRIVATE KEY"),
    ),
    _Card(
        "write-pem",
        "write_file",
        {"path": "k.txt", "content": _PEM_FILE},
        ("k.txt",),
        ("[redacted]", "tail=1"),
        (*_BODY, "PRIVATE KEY"),
    ),
    _Card(
        "write-key-in-path",
        "write_file",
        {"path": f"keys/{_TOKEN}.txt", "content": "x\n"},
        ("keys/[redacted].txt",),
        ("keys/[redacted].txt",),
        (_TOKEN,),
    ),
    _Card(
        "append-key",
        "append_file",
        {"path": "conf.env", "content": "API_KEY=newsecret456\n"},
        ("conf.env",),
        ("API_KEY=[redacted]",),
        ("newsecret456",),
    ),
    _Card(
        "delete-key-in-path",
        "delete_path",
        {"path": f"tmp/{_TOKEN}"},
        ("tmp/[redacted]",),
        ("tmp/[redacted]",),
        (_TOKEN,),
    ),
    _Card(
        "web-fetch-query",
        "web_fetch",
        {"url": "https://example.invalid/?token=abcd12345678"},
        ("?token=[redacted]",),
        ("?token=[redacted]",),
        ("abcd12345678",),
    ),
    _Card(
        "mcp-any-key",
        "mcp__ops__create",
        {"name": _TOKEN, "id": _KEY, "count": 3},
        ('"name": "[redacted]"', '"id": "[redacted]"', '"count": 3'),
        ('"name": "[redacted]"',),
        (_TOKEN, _KEY),
    ),
]
_IDS = [card.id for card in _CARDS]
_FILE_TOOLS = ("edit_file", "write_file", "append_file")


@pytest.fixture(autouse=True)
def _key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", _KEY)


def _call(card: _Card) -> ToolUse:
    return ToolUse(id="c", name=card.tool, input=dict(card.inp))


def _text(renderable: RenderableType) -> str:
    out = StringIO()
    Console(file=out, width=200, color_system=None).print(renderable)
    return out.getvalue()


def _card(card: _Card, *, detailed: bool, redact: bool) -> str:
    return _text(action_preview(_call(card), "workspace-write", detailed=detailed, redact=redact))


@pytest.mark.parametrize("detailed", [False, True], ids=["short", "full"])
@pytest.mark.parametrize("card", _CARDS, ids=_IDS)
def test_cards_when_on(card: _Card, detailed: bool) -> None:
    text = _card(card, detailed=detailed, redact=True)
    expected = card.full if detailed else card.short
    assert [s for s in expected if s not in text] == []
    assert [s for s in card.hidden if s in text] == []


@pytest.mark.parametrize("detailed", [False, True], ids=["short", "full"])
@pytest.mark.parametrize("card", _CARDS, ids=_IDS)
def test_cards_when_off_show_the_values(card: _Card, detailed: bool) -> None:
    text = _card(card, detailed=detailed, redact=False)
    assert "[redacted]" not in text
    if detailed or card.tool not in _FILE_TOOLS:
        assert [s for s in card.hidden if s not in text] == []


@pytest.mark.parametrize("detailed", [False, True], ids=["short", "full"])
@pytest.mark.parametrize("card", _CARDS, ids=_IDS)
def test_on_and_off_cards_have_the_same_number_of_lines(card: _Card, detailed: bool) -> None:
    on = _card(card, detailed=detailed, redact=True)
    off = _card(card, detailed=detailed, redact=False)
    assert len(on.splitlines()) == len(off.splitlines())


@pytest.mark.parametrize("redact", [True, False])
def test_the_short_cards_count_the_real_lines(redact: bool) -> None:
    write, append = (next(c for c in _CARDS if c.id == i) for i in ("write-pem", "append-key"))
    lines = ui("cli.render.card_lines_bytes", n=6, size=len(_PEM_FILE))
    assert f"k.txt{lines}" in _card(write, detailed=False, redact=redact)
    added = ui("cli.render.append_lines_bytes", n=1, size=21)
    assert f"conf.env{added}" in _card(append, detailed=False, redact=redact)


def test_the_full_write_card_counts_the_real_lines() -> None:
    [card] = [c for c in _CARDS if c.id == "write-pem"]
    text = _card(card, detailed=True, redact=True)
    header = ui("cli.render.preview_lines_bytes", n=6, path="k.txt", size=len(_PEM_FILE))
    assert header.rstrip("\n") in text
    assert text.count("[redacted]") == 4


def test_the_card_reads_the_keys_when_it_draws(monkeypatch: pytest.MonkeyPatch) -> None:
    late = "later-provider-key-0123456789"
    monkeypatch.setenv("ANTHROPIC_API_KEY", late)
    call = ToolUse(id="c", name="bash", input={"command": f"echo {late}"})
    text = _text(action_preview(call, "workspace-write", redact=True))
    assert late not in text and "echo [redacted]" in text


def _prompter(*, redact: bool) -> tuple[InteractivePrompter, StringIO]:
    buf = StringIO()
    return InteractivePrompter(Console(file=buf, width=200), redact=redact), buf


_REQUIRED = PermissionMode.DANGER_FULL_ACCESS


@pytest.mark.parametrize("redact", [True, False])
def test_the_prompter_draws_the_card_from_the_real_call(
    monkeypatch: pytest.MonkeyPatch, redact: bool
) -> None:
    prompter, buf = _prompter(redact=redact)
    monkeypatch.setattr(prompter.console, "input", lambda *a, **k: "n")
    [card] = [c for c in _CARDS if c.id == "edit-rotation"]
    outcome = prompter.decide(_call(card), _REQUIRED)
    assert outcome.decision == "deny"
    out = buf.getvalue()
    assert "conf.env  +1 -1" in out
    assert "oldsecret123" not in out and "[redacted]" not in out


@pytest.mark.parametrize("redact", [True, False])
def test_the_full_card_after_d_is_drawn_from_the_real_call(
    monkeypatch: pytest.MonkeyPatch, redact: bool
) -> None:
    prompter, buf = _prompter(redact=redact)
    monkeypatch.setattr(prompter.console, "input", lambda *a, **k: "n")
    [card] = [c for c in _CARDS if c.id == "edit-rotation"]
    detail = PermissionDetail(explanation="rotate the key", failure=None)
    prompter.decide_with_detail(_call(card), _REQUIRED, None, detail)
    out = buf.getvalue()
    if redact:
        assert "-API_KEY=[redacted]" in out and "+API_KEY=[redacted]" in out
        assert "oldsecret123" not in out and "newsecret456" not in out
    else:
        assert "-API_KEY=oldsecret123" in out and "+API_KEY=newsecret456" in out


def _prompts(monkeypatch: pytest.MonkeyPatch, call: ToolUse, *, redact: bool) -> list[str]:
    prompter, _ = _prompter(redact=redact)
    asked: list[str] = []

    def answer(prompt: str, **_: object) -> str:
        asked.append(prompt)
        return "n"

    monkeypatch.setattr(prompter.console, "input", answer)
    prompter.decide(call, _REQUIRED)
    return asked


@pytest.mark.parametrize("card", [c for c in _CARDS if c.tool == "bash"], ids=lambda c: c.id)
def test_the_always_answer_is_offered_as_without_redaction(
    monkeypatch: pytest.MonkeyPatch, card: _Card
) -> None:
    on = _prompts(monkeypatch, _call(card), redact=True)
    off = _prompts(monkeypatch, _call(card), redact=False)
    assert on == off


def _secret_parts(card: _Card) -> list[str]:
    return [s for s in card.hidden if s != "PRIVATE KEY"]


def _explained(
    monkeypatch: pytest.MonkeyPatch, call: ToolUse, detail: PermissionDetail, *, redact: bool
) -> str:
    prompter, buf = _prompter(redact=redact)
    monkeypatch.setattr(prompter.console, "input", lambda *a, **k: "n")
    prompter.decide_with_detail(call, _REQUIRED, None, detail)
    return buf.getvalue()


@pytest.mark.parametrize("redact", [True, False])
@pytest.mark.parametrize("card", [c for c in _CARDS if _secret_parts(c)], ids=lambda c: c.id)
def test_the_explanation_after_d_hides_what_the_card_hides(
    monkeypatch: pytest.MonkeyPatch, card: _Card, redact: bool
) -> None:
    said = "It uses " + ", ".join(f"`{s}`" for s in _secret_parts(card)) + " and stops."
    out = _explained(monkeypatch, _call(card), PermissionDetail(said, None), redact=redact)
    shown = [s for s in _secret_parts(card) if s in out]
    assert shown == ([] if redact else _secret_parts(card))
    assert ("It uses" in out and "and stops." in out) is True


def test_the_explanation_keeps_the_expansion_it_names(monkeypatch: pytest.MonkeyPatch) -> None:
    [card] = [c for c in _CARDS if c.id == "bash-after-an-expansion"]
    said = "It sets PASSWORD to `${PREFIX}private12345` and runs login.sh."
    out = _explained(monkeypatch, _call(card), PermissionDetail(said, None), redact=True)
    assert "`${PREFIX}[redacted]`" in out and "private12345" not in out


def test_the_explanation_hides_a_value_after_a_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    call = ToolUse(id="c", name="bash", input={"command": "PASSWORD=$PREFIX-private12345 ./x"})
    said = "It sets PASSWORD to $PREFIX-private12345."
    out = _explained(monkeypatch, call, PermissionDetail(said, None), redact=True)
    assert "$PREFIX[redacted]." in out and "private12345" not in out


def test_short_hidden_pieces_do_not_cut_words(monkeypatch: pytest.MonkeyPatch) -> None:
    call = ToolUse(id="c", name="bash", input={"command": "PASSWORD=ab$(x) ./run"})
    said = "It runs x about the abc file."
    out = _explained(monkeypatch, call, PermissionDetail(said, None), redact=True)
    assert said in out


@pytest.mark.parametrize("redact", [True, False])
def test_the_facts_without_an_explanation_hide_it_too(
    monkeypatch: pytest.MonkeyPatch, redact: bool
) -> None:
    [card] = [c for c in _CARDS if c.id == "bash-quoted-password"]
    detail = PermissionDetail(
        explanation=None,
        failure="timeout",
        model_text="I will log in with ab$private12345 now.",
        step=(1, 2, f"use {_KEY}"),
        user_request="the password is ab$private12345",
    )
    out = _explained(monkeypatch, _call(card), detail, redact=redact)
    assert ("private12345" in out) is not redact
    assert (_KEY in out) is not redact

"""応答言語制御のテスト。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError
from rich.console import Console

from satsumaestro.cli.commands import SlashDispatcher, parse_slash
from satsumaestro.config.models import HookConfig, LanguageConfig, RuntimeConfig, ToolsConfig
from satsumaestro.engine.events import MessageStop, TextDelta
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.engine.subagent import RuntimeFactory, resolve_subagent_type
from satsumaestro.evals.models import EvalReport, TaskResult
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.i18n.state import LanguageState
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.models import Session
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from satsumaestro.util.lang import (
    UNDETERMINED,
    detect_language,
    language_name,
    requests_other_language,
    resolve_language,
    resolve_turn_language,
    strip_non_prose,
    with_language_directive,
)
from tests.fixtures.mock_provider import SequencedMockProvider

JA = "ja"
EN = "en"


def _detect(text: str) -> str:
    """製品経路と同じく「非散文を落としてから判定する」を 1 つにまとめる。"""
    return detect_language(strip_non_prose(text))


def test_japanese_question_detected() -> None:
    assert _detect("with_tax 関数を呼び出しているのはどのファイルのどの関数ですか。") == JA


def test_english_paragraph_detected() -> None:
    assert _detect("The helper is defined in the cart module and called from checkout.") == EN


def test_short_or_empty_text_is_undetermined() -> None:
    assert _detect("") == UNDETERMINED
    assert _detect("OK") == UNDETERMINED
    assert _detect("123 456") == UNDETERMINED


def test_mixed_text_follows_japanese() -> None:
    text = "The function is defined in the cart module. これは合計金額を計算する関数です。"
    assert _detect(text) == JA


def test_code_fence_is_not_counted() -> None:
    fence = "```python\n" + "value = compute(total)\n" * 20 + "```\n"
    assert _detect("この関数の役割を教えてください。\n\n" + fence) == JA


def test_answer_that_is_almost_only_code_is_undetermined() -> None:
    text = "Here:\n\n```python\n" + "value = compute(total)\n" * 20 + "```\n"
    assert _detect(text) == UNDETERMINED


def test_identifiers_and_paths_do_not_make_english_japanese() -> None:
    text = "The caller is `with_tax`, documented in docs/設計/概要.md near the top of the file."
    assert _detect(text) == EN


def test_quoted_english_block_is_ignored() -> None:
    text = "エラーの原因は次の行です。\n\n> Traceback (most recent call last): ValueError raised\n"
    assert _detect(text) == JA


def test_unsupported_scripts_are_undetermined() -> None:
    assert _detect("이 함수는 어디에서 호출됩니까?") == UNDETERMINED
    assert _detect("Где определена эта функция?") == UNDETERMINED


def test_kanji_only_short_text_is_japanese() -> None:
    assert _detect("実装計画") == JA


def test_resolve_language_keeps_previous_when_undetermined() -> None:
    assert resolve_language(JA, UNDETERMINED) == JA
    assert resolve_language("", UNDETERMINED) == ""
    assert resolve_language(JA, EN) == EN


def test_requests_other_language_guard() -> None:
    assert requests_other_language("この説明を英語で書いて") is True
    assert requests_other_language("Answer in Japanese please") is True
    assert requests_other_language("with_tax はどこで呼ばれていますか") is False


def test_language_name_and_directive() -> None:
    assert language_name(JA) == "Japanese"
    assert language_name(EN) == "English"
    assert with_language_directive("質問です", "") == "質問です"
    assert with_language_directive("質問です", JA).startswith("質問です\n\n(Reply in Japanese.")


def test_resolve_turn_language_auto_follows_user() -> None:
    assert resolve_turn_language("auto", JA, "", "この関数はどこで呼ばれていますか") == JA
    assert resolve_turn_language("auto", JA, JA, "Where is this function called?") == EN


def test_resolve_turn_language_auto_keeps_previous_on_undetermined() -> None:
    assert resolve_turn_language("auto", JA, EN, "OK") == EN


def test_resolve_turn_language_fixed_overrides_user() -> None:
    assert resolve_turn_language("fixed", JA, "", "Where is this function called?") == JA


def test_resolve_turn_language_off_disables_control() -> None:
    assert resolve_turn_language("off", JA, JA, "この関数はどこで呼ばれていますか") == ""


def test_language_config_defaults() -> None:
    cfg = RuntimeConfig()
    assert cfg.language.mode == "auto"
    assert cfg.language.fixed == JA
    assert cfg.language.repair is True


def test_language_config_rejects_unsupported_language() -> None:
    with pytest.raises(ValidationError):
        LanguageConfig(fixed="zh")
    with pytest.raises(ValidationError):
        LanguageConfig.model_validate({"mode": "auto", "unknown": 1})


def test_session_language_round_trips(tmp_path: Path) -> None:
    store = JSONLSessionStore()
    s = new_session(tmp_path, "m")
    s.language = JA
    store.save(s)
    assert store.load(store.path_for(tmp_path, s.session_id)).language == JA


def test_session_without_language_key_loads() -> None:
    meta = {
        "version": 1,
        "session_id": "s1",
        "created_at_ms": 0,
        "updated_at_ms": 0,
        "workspace_root": "/tmp",
        "model": "m",
    }
    assert Session(**meta).language == ""  # type: ignore[arg-type]


def _runtime(
    provider: object, tmp_path: Path, cfg: RuntimeConfig | None = None
) -> ConversationRuntime:
    cfg = cfg if cfg is not None else RuntimeConfig()
    return ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=provider,  # type: ignore[arg-type]
        tools=default_registry(),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=cfg,
    )


async def test_language_notice_is_dynamic_only(tmp_path: Path) -> None:
    b = SystemPromptBuilder()
    language = LanguageState(LanguageConfig())
    baseline = (await b.build(tmp_path, "m", RuntimeConfig(), language)).static
    language.begin_turn("この関数はどこで呼ばれていますか")
    prompt = await b.build(tmp_path, "m", RuntimeConfig(), language)
    assert prompt.static == baseline
    assert "# Language" in prompt.dynamic
    assert "Japanese" in prompt.joined()


async def test_language_notice_absent_when_unset(tmp_path: Path) -> None:
    prompt = await SystemPromptBuilder().build(
        tmp_path, "m", RuntimeConfig(), LanguageState(LanguageConfig())
    )
    assert "# Language" not in prompt.joined()


async def test_language_notice_fixed_mode_overrides_convention(tmp_path: Path) -> None:
    language = LanguageState(LanguageConfig())
    language.set_fixed(JA)
    prompt = await SystemPromptBuilder().build(tmp_path, "m", RuntimeConfig(), language)
    assert "overrides the general convention" in prompt.joined()


async def test_directive_appended_to_user_message(tmp_path: Path) -> None:
    provider = SequencedMockProvider([[TextDelta(text="cart.py の total です"), MessageStop()]])
    rt = _runtime(provider, tmp_path, RuntimeConfig())
    await rt.run_turn("with_tax はどこで呼ばれていますか")
    sent = rt.session.messages[0].blocks[0].text  # type: ignore[union-attr]
    assert sent.endswith(
        "(Reply in Japanese. Code, paths, identifiers and command output stay as-is.)"
    )
    assert rt.session.prompt_history == ["with_tax はどこで呼ばれていますか"]
    assert rt.session.language == JA


async def test_language_off_keeps_message_byte_identical(tmp_path: Path) -> None:
    cfg = RuntimeConfig(language=LanguageConfig(mode="off"))
    provider = SequencedMockProvider([[TextDelta(text="done"), MessageStop()]])
    rt = _runtime(provider, tmp_path, cfg)
    await rt.run_turn("with_tax はどこで呼ばれていますか")
    assert rt.session.messages[0].blocks[0].text == "with_tax はどこで呼ばれていますか"  # type: ignore[union-attr]
    assert rt.session.language == ""


async def test_language_follows_switch_mid_conversation(tmp_path: Path) -> None:
    provider = SequencedMockProvider(
        [
            [TextDelta(text="cart.py の total です"), MessageStop()],
            [TextDelta(text="It is total in cart.py"), MessageStop()],
            [TextDelta(text="It is total in cart.py"), MessageStop()],
        ]
    )
    rt = _runtime(provider, tmp_path, RuntimeConfig())
    await rt.run_turn("with_tax はどこで呼ばれていますか")
    assert rt.session.language == JA
    await rt.run_turn("Where is with_tax called from?")
    assert rt.session.language == EN
    await rt.run_turn("OK")
    assert rt.session.language == EN


async def test_fixed_mode_ignores_user_language(tmp_path: Path) -> None:
    cfg = RuntimeConfig(language=LanguageConfig(mode="fixed", fixed=JA, repair=False))
    provider = SequencedMockProvider([[TextDelta(text="cart.py の total です"), MessageStop()]])
    rt = _runtime(provider, tmp_path, cfg)
    await rt.run_turn("Where is with_tax called from?")
    assert rt.session.language == JA
    assert "(Reply in Japanese." in rt.session.messages[0].blocks[0].text  # type: ignore[union-attr]


async def test_child_runtime_inherits_language(tmp_path: Path) -> None:
    tools = default_registry()
    pol = default_policy(PermissionMode.WORKSPACE_WRITE)
    hooks = HookRunner(HookConfig(), tmp_path)
    cfg = RuntimeConfig(tools=ToolsConfig(deferred_tool_loading=False))
    provider = SequencedMockProvider([[TextDelta(text="cart.py の total です"), MessageStop()]])
    factory = RuntimeFactory(
        provider=provider,  # type: ignore[arg-type]
        model="mock-model",
        config=cfg,
        tools=tools,
        policy=pol,
        hooks=hooks,
        cwd=tmp_path,
        provider_builder=lambda _m: provider,  # type: ignore[arg-type,return-value]
    )
    rt = ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=provider,  # type: ignore[arg-type]
        tools=tools,
        policy=pol,
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=hooks,
        config=cfg,
        runtime_factory=factory,
    )
    assert factory.build_child(preset=resolve_subagent_type(""), depth=1).session.language == ""
    await rt.run_turn("with_tax はどこで呼ばれていますか")
    assert factory.language.current == JA
    child = factory.build_child(preset=resolve_subagent_type(""), depth=1)
    assert child.session.language == JA


_EN_ANSWER = "The caller is the total function in the cart module of this project."
_JA_ANSWER = "呼び出しているのは cart モジュールの total 関数です。"


async def test_language_mismatch_repairs_once_and_succeeds(tmp_path: Path) -> None:
    provider = SequencedMockProvider(
        [
            [TextDelta(text=_EN_ANSWER), MessageStop()],
            [TextDelta(text=_JA_ANSWER), MessageStop()],
        ]
    )
    rt = _runtime(provider, tmp_path, RuntimeConfig())
    summary = await rt.run_turn("with_tax はどこで呼ばれていますか")
    assert summary.final_text == _JA_ANSWER
    assert summary.iterations == 2
    assert summary.language_expected == JA
    assert summary.language_detected == JA
    assert summary.language_repaired is True
    assert summary.reminders == 1
    injected = [
        m
        for m in rt.session.messages
        if m.role == "user" and "<system-reminder>" in (m.blocks[0].text or "")  # type: ignore[union-attr]
    ]
    assert len(injected) == 1
    assert "Rewrite the same answer in Japanese" in injected[0].blocks[0].text  # type: ignore[union-attr]


async def test_language_mismatch_gives_up_after_one_repair(tmp_path: Path) -> None:
    provider = SequencedMockProvider(
        [[TextDelta(text=_EN_ANSWER), MessageStop()], [TextDelta(text=_EN_ANSWER), MessageStop()]]
    )
    rt = _runtime(provider, tmp_path, RuntimeConfig())
    summary = await rt.run_turn("with_tax はどこで呼ばれていますか")
    assert summary.final_text == _EN_ANSWER
    assert summary.iterations == 2
    assert summary.language_detected == EN
    assert summary.language_repaired is True


async def test_language_repair_disabled_only_observes(tmp_path: Path) -> None:
    cfg = RuntimeConfig(language=LanguageConfig(repair=False))
    provider = SequencedMockProvider([[TextDelta(text=_EN_ANSWER), MessageStop()]])
    rt = _runtime(provider, tmp_path, cfg)
    summary = await rt.run_turn("with_tax はどこで呼ばれていますか")
    assert summary.iterations == 1
    assert summary.language_expected == JA
    assert summary.language_detected == EN
    assert summary.language_repaired is False
    assert summary.reminders == 0


async def test_language_mismatch_not_fired_when_matching(tmp_path: Path) -> None:
    provider = SequencedMockProvider([[TextDelta(text=_JA_ANSWER), MessageStop()]])
    rt = _runtime(provider, tmp_path, RuntimeConfig())
    summary = await rt.run_turn("with_tax はどこで呼ばれていますか")
    assert summary.iterations == 1
    assert summary.language_repaired is False
    assert summary.reminders == 0


async def test_language_mismatch_skipped_when_user_asked_for_other_language(
    tmp_path: Path,
) -> None:
    provider = SequencedMockProvider([[TextDelta(text=_EN_ANSWER), MessageStop()]])
    rt = _runtime(provider, tmp_path, RuntimeConfig())
    summary = await rt.run_turn("with_tax の呼び出し元を英語で説明してください")
    assert summary.iterations == 1
    assert summary.language_repaired is False


async def test_language_mismatch_skipped_when_answer_is_undetermined(tmp_path: Path) -> None:
    code = "```python\n" + "value = compute(total)\n" * 20 + "```"
    provider = SequencedMockProvider([[TextDelta(text=code), MessageStop()]])
    rt = _runtime(provider, tmp_path, RuntimeConfig())
    summary = await rt.run_turn("with_tax はどこで呼ばれていますか")
    assert summary.iterations == 1
    assert summary.language_detected == UNDETERMINED
    assert summary.language_repaired is False


async def test_language_detected_not_stale_when_final_body_empty(tmp_path: Path) -> None:
    provider = SequencedMockProvider(
        [
            [TextDelta(text=_EN_ANSWER), MessageStop()],
            [MessageStop()],
            [MessageStop()],
        ]
    )
    rt = _runtime(provider, tmp_path, RuntimeConfig())
    summary = await rt.run_turn("with_tax はどこで呼ばれていますか")
    assert summary.final_text == ""
    assert summary.language_repaired is True
    assert summary.language_detected == ""


async def test_language_mismatch_respects_max_iterations(tmp_path: Path) -> None:
    cfg = RuntimeConfig(max_iterations=1)
    provider = SequencedMockProvider([[TextDelta(text=_EN_ANSWER), MessageStop()]])
    rt = _runtime(provider, tmp_path, cfg)
    summary = await rt.run_turn("with_tax はどこで呼ばれていますか")
    assert summary.iterations == 1
    assert summary.language_repaired is False


async def test_language_off_disables_detection(tmp_path: Path) -> None:
    cfg = RuntimeConfig(language=LanguageConfig(mode="off"))
    provider = SequencedMockProvider([[TextDelta(text=_EN_ANSWER), MessageStop()]])
    rt = _runtime(provider, tmp_path, cfg)
    summary = await rt.run_turn("with_tax はどこで呼ばれていますか")
    assert summary.language_expected == ""
    assert summary.language_detected == ""
    assert summary.language_repaired is False


def test_task_result_round_trips_language_fields() -> None:
    r = TaskResult(
        task_id="t",
        passed=True,
        language_expected=JA,
        language_detected=EN,
        language_repaired=True,
    )
    again = TaskResult.model_validate(r.model_dump())
    assert (again.language_expected, again.language_detected, again.language_repaired) == (
        JA,
        EN,
        True,
    )


def test_task_result_reads_legacy_report_without_language_keys() -> None:
    r = TaskResult.model_validate({"task_id": "t", "passed": True})
    assert r.language_expected == "" and r.language_detected == ""
    assert r.language_repaired is False


_AB_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "eval_ab_summary.py"


def _report(path: Path, rows: list[dict[str, object]]) -> Path:
    """本題は言語指標なので、``passed`` は指定が無ければ True で埋める。"""
    report = EvalReport(agent="satsuma", model="m", started_at_ms=0)
    report.results = [TaskResult.model_validate({"passed": True, **r}) for r in rows]
    path.write_text(report.model_dump_json(), encoding="utf-8")
    return path


def _ab_summary(before: Path, after: Path) -> str:
    proc = subprocess.run(
        [
            sys.executable,
            str(_AB_SCRIPT),
            "--title",
            "s",
            "--before",
            str(before),
            "--after",
            str(after),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout


def test_ab_summary_counts_drift_and_repairs(tmp_path: Path) -> None:
    before = _report(
        tmp_path / "b.json",
        [
            {
                "task_id": "a",
                "language_expected": JA,
                "language_detected": JA,
                "language_repaired": True,
            },
            {
                "task_id": "b",
                "language_expected": JA,
                "language_detected": EN,
                "language_repaired": True,
            },
            {"task_id": "c", "passed": True, "language_expected": JA, "language_detected": JA},
            {
                "task_id": "d",
                "passed": True,
                "language_expected": JA,
                "language_detected": UNDETERMINED,
            },
        ],
    )
    after = _report(
        tmp_path / "a.json",
        [
            {"task_id": "a", "passed": True, "language_expected": JA, "language_detected": JA},
            {"task_id": "b", "passed": True, "language_expected": JA, "language_detected": JA},
            {"task_id": "c", "passed": True, "language_expected": JA, "language_detected": JA},
            {
                "task_id": "d",
                "passed": True,
                "language_expected": JA,
                "language_detected": UNDETERMINED,
            },
        ],
    )
    out = _ab_summary(before, after)
    assert "| 言語ドリフト率 (修復前) | 75.0% | 0.0% | -75.0% |" in out
    assert "| 言語不一致率 (修復後に残存) | 25.0% | 0.0% | -25.0% |" in out
    assert "| 言語修復の発火タスク数 | 2.0 | 0.0 | -2.0 |" in out


def test_ab_summary_reads_legacy_reports_as_zero(tmp_path: Path) -> None:
    legacy = json.loads(
        _report(tmp_path / "b.json", [{"task_id": "x", "passed": True}]).read_text()
    )
    for r in legacy["results"]:
        for key in ("language_expected", "language_detected", "language_repaired"):
            r.pop(key, None)
    (tmp_path / "b.json").write_text(json.dumps(legacy), encoding="utf-8")
    out = _ab_summary(
        tmp_path / "b.json", _report(tmp_path / "a.json", [{"task_id": "x", "passed": True}])
    )
    assert "| 言語ドリフト率 (修復前) | 0.0% | 0.0% | +0.0% |" in out


async def test_the_reply_language_is_not_fixed_from_the_repl(tmp_path: Path) -> None:
    rt = _runtime(SequencedMockProvider([[MessageStop()]]), tmp_path)
    console = Console(record=True, width=160)
    d = SlashDispatcher(rt, console)
    await d.dispatch(parse_slash("/lang en"))  # type: ignore[arg-type]
    assert "/lang" in console.export_text(clear=True)
    assert rt.language.mode == "auto"

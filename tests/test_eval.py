"""Tests for cairn.core.eval and the `cairn eval` command.

Every Anthropic client here is a mock, for extraction and judging alike, so
nothing makes a network call.
"""

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from anthropic.types import Message, ToolUseBlock, Usage
from typer.testing import CliRunner

from cairn import cli
from cairn.core.eval import (
    JUDGE_TOOL_NAME,
    JudgeError,
    build_judge_prompt,
    check_atomic,
    check_evidence_backed,
    check_non_redundant,
    evaluate_fixture,
    judge_candidate,
    match_candidates,
    score_entry,
    summarize,
)
from cairn.core.models import (
    Confidence,
    Entry,
    EntryStatus,
    EntryType,
    Evidence,
    SessionTrace,
)
from cairn.core.salience import find_salient_spans, is_reflectable
from cairn.providers.anthropic import TOOL_NAME as EXTRACTION_TOOL_NAME

runner = CliRunner()

GOLD_DIR = Path(__file__).parent / "fixtures" / "gold"
SPEC_PATH = Path(__file__).resolve().parent.parent / "SPEC.md"
PROXY_NOTE = (
    "precision/non-redundant are judge- and match-scored, not human-verified — see README's "
    "Evaluation section for why this is a proxy, not ground truth."
)

GOTCHA_BODY = "## What happens\n\nIt breaks.\n\n## What to do\n\nFix it.\n"
PASSING_VERDICT: dict[str, object] = {
    "actionable": True,
    "actionable_reason": "It says exactly what to change.",
    "project_specific": True,
    "project_specific_reason": "It names this repository's files.",
    "stable": True,
    "stable_reason": "It describes how the repository is built, not a branch.",
    "future_useful": True,
    "future_useful_reason": "Any agent touching these tests would hit it.",
}


def _entry(
    title: str,
    *,
    entry_type: EntryType = EntryType.GOTCHA,
    session_id: str | None = "session-1",
    artifacts: list[str] | None = None,
    excerpt_sha256: str | None = None,
) -> Entry:
    when = datetime(2026, 9, 12, 11, 0, 0, tzinfo=UTC)
    return Entry(
        id=f"{entry_type.value}-000000",
        type=entry_type,
        title=title,
        status=EntryStatus.STAGED,
        spec_version="0.1.0",
        scope=["tests/**"],
        confidence=Confidence.HIGH,
        evidence=Evidence(
            harness="claude-code",
            session_id=session_id,
            captured_at=when,
            artifacts=["tests/conftest.py"] if artifacts is None else artifacts,
            excerpt_sha256=excerpt_sha256,
        ),
        created=when,
        updated=when,
    )


def _tool_response(name: str, tool_input: dict[str, object]) -> Message:
    return Message(
        id="msg_test",
        type="message",
        role="assistant",
        model="claude-sonnet-4-6",
        content=[ToolUseBlock(id="toolu_test", type="tool_use", name=name, input=tool_input)],
        stop_reason="tool_use",
        usage=Usage(input_tokens=1, output_tokens=1),
    )


def _judge_client(verdict: dict[str, object] = PASSING_VERDICT) -> MagicMock:
    client = MagicMock()
    client.messages.create.return_value = _tool_response(JUDGE_TOOL_NAME, verdict)
    return client


def _load_gold(n: int) -> list[dict[str, Any]]:
    gold: list[dict[str, Any]] = json.loads((GOLD_DIR / f"gold_{n}.json").read_text("utf-8"))
    return gold


def _load_trace(n: int) -> SessionTrace:
    raw = (GOLD_DIR / f"session_trace_{n}.json").read_text("utf-8")
    return SessionTrace.model_validate(json.loads(raw))


# --- Gold fixtures ------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "expected_type"),
    [(1, EntryType.GOTCHA), (2, EntryType.STRATEGY), (3, EntryType.FACT)],
)
def test_gold_fixture_is_well_formed(n: int, expected_type: EntryType) -> None:
    trace = _load_trace(n)
    gold = _load_gold(n)

    assert is_reflectable(trace, min_turns=4)
    assert gold
    for gold_entry in gold:
        assert set(gold_entry) == {"title", "type", "scope", "tags", "confidence", "body"}
        assert EntryType(gold_entry["type"]) is expected_type
        Confidence(gold_entry["confidence"])
        assert gold_entry["scope"] and gold_entry["tags"]
        # Gold entries meet the code-checked bar they are used to grade against.
        assert check_atomic(_entry(gold_entry["title"]), gold_entry["body"])


def test_gold_titles_do_not_match_each_other_across_fixtures() -> None:
    titles = [{"title": g["title"]} for n in (1, 2, 3) for g in _load_gold(n)]

    for i, gold_entry in enumerate(titles):
        others = [_entry(other["title"]) for j, other in enumerate(titles) if j != i]
        assert check_non_redundant(_entry(gold_entry["title"]), others)


def test_fixture_3_has_fewer_gold_entries_than_salient_spans() -> None:
    spans = find_salient_spans(_load_trace(3))

    assert len(spans) == 2
    assert len(_load_gold(3)) < len(spans)


# --- match_candidates ---------------------------------------------------------


def test_match_pairs_candidates_and_gold_one_to_one() -> None:
    stale = _entry("Run alembic upgrade head before the test suite")
    unrelated = _entry("Frontend bundle exceeds the size budget")
    port = _entry("Integration Redis listens on port 6380")
    gold = [
        {"title": "Integration-test Redis listens on host port 6380"},
        {"title": "Run `alembic upgrade head` before the test suite"},
        {"title": "Nightly exports are gzip-compressed"},
    ]

    result = match_candidates([stale, unrelated, port], gold)

    assert result.matched == [(stale, gold[1]), (port, gold[0])]
    assert result.unmatched_candidates == [unrelated]
    assert result.unmatched_gold == [gold[2]]


def test_match_gives_a_gold_entry_to_only_its_best_candidate() -> None:
    weaker = _entry("Redis port differs for tests")
    stronger = _entry("Integration-test Redis is on port 6380")
    gold = [{"title": "The integration-test Redis is on host port 6380, not 6379"}]

    result = match_candidates([weaker, stronger], gold, threshold=0.3)

    assert result.matched == [(stronger, gold[0])]
    assert result.unmatched_candidates == [weaker]
    assert result.unmatched_gold == []


# --- Code checks --------------------------------------------------------------


def test_evidence_backed_needs_session_id_and_an_artifact_or_excerpt_hash() -> None:
    assert check_evidence_backed(_entry("t"))
    assert check_evidence_backed(_entry("t", artifacts=[], excerpt_sha256="ab" * 32))
    assert not check_evidence_backed(_entry("t", session_id=None))
    assert not check_evidence_backed(_entry("t", artifacts=[]))
    assert not check_evidence_backed(_entry("t", artifacts=["  "]))


@pytest.mark.parametrize(
    ("body", "atomic"),
    [
        (GOTCHA_BODY, True),
        (
            "## What happens\n\nx\n\n"
            "## Why the obvious fix doesn't work:\n\ny\n\n"
            "## WHAT TO DO\n\nz",
            True,
        ),
        ("## The fact\n\nx\n\n### Details\n\ny\n", True),
        ("## The fact\n\n```md\n## Not a heading\n```\n\n## What to do\n\nz\n", True),
        ("## What happens\n\nx\n\n## Also\n\ny\n", False),
        ("Just a paragraph with no headings.\n", False),
        ("## What to do\n\nOnly scaffolding.\n", False),
        ("# Title\n\n###Not level two\n", False),
    ],
)
def test_check_atomic_counts_level_two_claim_headings(body: str, atomic: bool) -> None:
    assert check_atomic(_entry("t"), body) is atomic


# --- Judge --------------------------------------------------------------------


def _spec_rows(*criteria: str) -> list[str]:
    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec.split("\n## Entry quality criteria", 1)[1].split("\n## ", 1)[0]
    rows = re.findall(r"^\| \*\*([^*]+)\*\* \| .+ \|$", section, re.MULTILINE)
    lines = re.findall(r"^\| \*\*[^*]+\*\* \| .+ \|$", section, re.MULTILINE)
    by_name = dict(zip(rows, lines, strict=True))
    return [by_name[name] for name in criteria]


def test_judge_prompt_has_the_four_judged_spec_rows_verbatim_and_not_the_others() -> None:
    entry = _entry("Run alembic upgrade head before the test suite")

    prompt = build_judge_prompt(entry, GOTCHA_BODY)

    for row in _spec_rows("Actionable", "Project-specific", "Stable", "Future-useful"):
        assert row in prompt
    for row in _spec_rows("Evidence-backed", "Atomic", "Non-redundant"):
        assert row not in prompt
    assert "title: Run alembic upgrade head before the test suite" in prompt
    assert "## What happens\n\nIt breaks." in prompt


def test_judge_candidate_forces_the_verdict_tool_and_parses_it() -> None:
    client = _judge_client()

    verdict = judge_candidate(_entry("t"), GOTCHA_BODY, client)

    request = client.messages.create.call_args.kwargs
    assert request["tool_choice"] == {"type": "tool", "name": JUDGE_TOOL_NAME}
    assert request["tools"][0]["name"] == JUDGE_TOOL_NAME
    assert verdict.actionable is True
    assert verdict.future_useful_reason == "Any agent touching these tests would hit it."


def test_judge_candidate_raises_when_the_verdict_is_missing_or_invalid() -> None:
    client = MagicMock()
    client.messages.create.return_value = _tool_response("some_other_tool", {})
    with pytest.raises(JudgeError):
        judge_candidate(_entry("t"), GOTCHA_BODY, client)

    client.messages.create.return_value = _tool_response(JUDGE_TOOL_NAME, {"actionable": True})
    with pytest.raises(JudgeError):
        judge_candidate(_entry("t"), GOTCHA_BODY, client)


def test_score_entry_separates_code_match_and_judge_criteria() -> None:
    entry = _entry("Run alembic upgrade head before the test suite")
    known = [_entry("Run alembic upgrade head before the tests")]

    scores = score_entry(entry, GOTCHA_BODY, _judge_client(), known=known)

    assert scores["code_checked"] == {"evidence_backed": True, "atomic": True}
    assert scores["match_checked"] == {"non_redundant": False}
    assert set(scores["judge_scored"]) == {
        "actionable",
        "project_specific",
        "stable",
        "future_useful",
    }
    assert scores["judge_scored"]["stable"]["pass"] is True
    assert scores["passed_all"] is False


# --- Precision and recall -----------------------------------------------------


def test_matched_candidate_passing_all_seven_counts_toward_precision() -> None:
    gold = _load_gold(1)
    candidate = _entry(gold[0]["title"])

    fixture = evaluate_fixture([(candidate, gold[0]["body"])], gold, _judge_client())
    summary = summarize([fixture])

    record = fixture["candidates"][0]
    assert record["scores"]["passed_all"] is True
    assert record["counts_toward_precision"] is True
    assert summary["precision"] == {"rate": 1.0, "numerator": 1, "denominator": 1}
    assert summary["recall"] == {"rate": 1.0, "numerator": 1, "denominator": 1}


def test_unmatched_candidate_counts_against_precision_not_recall() -> None:
    gold = _load_gold(1)
    matched = _entry(gold[0]["title"])
    unmatched = _entry("Invoice PDFs render with the wrong currency symbol")

    fixture = evaluate_fixture(
        [(matched, gold[0]["body"]), (unmatched, GOTCHA_BODY)], gold, _judge_client()
    )
    summary = summarize([fixture])

    unmatched_record = fixture["candidates"][1]
    # It passes every criterion on its own, but has no gold entry.
    assert unmatched_record["scores"]["passed_all"] is True
    assert unmatched_record["matched_gold_title"] is None
    assert unmatched_record["counts_toward_precision"] is False
    assert summary["precision"] == {"rate": 0.5, "numerator": 1, "denominator": 2}
    assert summary["recall"] == {"rate": 1.0, "numerator": 1, "denominator": 1}


def test_fixture_3_red_herring_does_not_count_against_recall() -> None:
    gold = _load_gold(3)
    fact = _entry(
        "Integration-test Redis runs on host port 6380, not 6379", entry_type=EntryType.FACT
    )
    red_herring = _entry("Redis connection refused on port 6380 during integration tests")

    with_herring = evaluate_fixture(
        [(fact, gold[0]["body"]), (red_herring, GOTCHA_BODY)], gold, _judge_client()
    )
    without_herring = evaluate_fixture([(fact, gold[0]["body"])], gold, _judge_client())

    # The span with no gold entry adds nothing to recall's denominator, whether
    # or not the reflector proposes something for it.
    for fixture in (with_herring, without_herring):
        assert summarize([fixture])["recall"] == {"rate": 1.0, "numerator": 1, "denominator": 1}
    herring_record = with_herring["candidates"][1]
    assert herring_record["matched_gold_title"] is None
    assert herring_record["counts_toward_precision"] is False
    assert summarize([with_herring])["precision"]["rate"] == 0.5


def test_duplicate_of_known_or_earlier_candidate_is_counted() -> None:
    known = [_entry("Run alembic upgrade head before the test suite")]
    candidates = [
        (_entry("Run alembic upgrade head before tests"), GOTCHA_BODY),
        (_entry("Frontend bundle exceeds the size budget"), GOTCHA_BODY),
        (_entry("Frontend bundle exceeds its size budget"), GOTCHA_BODY),
    ]

    fixture = evaluate_fixture(candidates, [], _judge_client(), known=known)

    assert [r["duplicate"] for r in fixture["candidates"]] == [True, False, True]
    assert summarize([fixture])["duplicate_rate"]["numerator"] == 2


def test_judge_failure_is_recorded_and_fails_the_candidate() -> None:
    client = MagicMock()
    client.messages.create.return_value = _tool_response("some_other_tool", {})
    gold = _load_gold(1)

    fixture = evaluate_fixture([(_entry(gold[0]["title"]), gold[0]["body"])], gold, client)

    record = fixture["candidates"][0]
    assert record["judge_error"]
    assert record["scores"]["judge_scored"] is None
    assert record["scores"]["code_checked"] == {"evidence_backed": True, "atomic": True}
    assert record["counts_toward_precision"] is False
    assert summarize([fixture])["judge_errors"] == 1


# --- cairn eval ---------------------------------------------------------------


def _fake_anthropic(extraction_batches: list[list[dict[str, object]]]) -> MagicMock:
    """A client that answers extraction calls from `extraction_batches`, in
    order, and every judge call with a passing verdict."""

    batches = list(extraction_batches)

    def create(**kwargs: Any) -> Message:
        tool = kwargs["tool_choice"]["name"]
        if tool == JUDGE_TOOL_NAME:
            return _tool_response(JUDGE_TOOL_NAME, PASSING_VERDICT)
        if tool == EXTRACTION_TOOL_NAME:
            return _tool_response(EXTRACTION_TOOL_NAME, {"candidates": batches.pop(0)})
        raise AssertionError(f"unexpected tool {tool}")

    client = MagicMock()
    client.messages.create.side_effect = create
    return client


def test_eval_cli_with_anthropic_extraction_scores_gold_like_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batches = [[{"id": "slug", **g} for g in _load_gold(n)] for n in (1, 2, 3)]
    client = _fake_anthropic(batches)
    monkeypatch.setattr(cli, "_anthropic_client", lambda: client)
    report = tmp_path / "eval-report.json"

    result = runner.invoke(cli.app, ["eval", "--suite", str(GOLD_DIR), "--report", str(report)])

    assert result.exit_code == 0, result.output
    assert "both call the Anthropic API" in result.output
    assert "schema validity  100.0% (3/3)" in result.output
    assert "precision        100.0% (3/3)" in result.output
    assert "recall           100.0% (3/3)" in result.output
    assert "duplicate rate   0.0% (0/3)" in result.output
    assert PROXY_NOTE in result.output

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["modes"]["extraction"] == "anthropic"
    assert [f["trace"] for f in payload["fixtures"]] == [
        "session_trace_1.json",
        "session_trace_2.json",
        "session_trace_3.json",
    ]
    record = payload["fixtures"][0]["candidates"][0]
    assert set(record["scores"]) == {"code_checked", "match_checked", "judge_scored", "passed_all"}


def test_eval_cli_mock_mode_flags_the_real_judge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _fake_anthropic([])
    monkeypatch.setattr(cli, "_anthropic_client", lambda: client)
    report = tmp_path / "eval-report.json"

    result = runner.invoke(
        cli.app, ["eval", "--suite", str(GOLD_DIR), "--report", str(report), "--mock"]
    )

    assert result.exit_code == 0, result.output
    assert "mixed modes" in result.output
    assert "judge still calls the Anthropic API" in result.output
    assert PROXY_NOTE in result.output

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["modes"] == {
        "extraction": "mock",
        "judge": "anthropic",
        "judge_model": "claude-sonnet-4-6",
    }
    candidates = [r for f in payload["fixtures"] for r in f["candidates"]]
    assert candidates
    assert payload["summary"]["schema_validity"]["rate"] == 1.0
    # One judge call per mock-extracted candidate, and nothing else.
    assert client.messages.create.call_count == len(candidates)


def test_eval_cli_errors_when_a_trace_has_no_gold_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "_anthropic_client", lambda: _fake_anthropic([]))
    (tmp_path / "session_trace_1.json").write_text(
        (GOLD_DIR / "session_trace_1.json").read_text("utf-8"), encoding="utf-8"
    )

    result = runner.invoke(cli.app, ["eval", "--suite", str(tmp_path), "--mock"])

    assert result.exit_code == 1
    assert "gold_1.json" in result.output

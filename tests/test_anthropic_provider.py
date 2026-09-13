"""Tests for cairn.providers.anthropic.AnthropicProvider.

The Anthropic client is always a mock injected through the constructor, so
nothing here makes a network call.
"""

import logging
import re
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from anthropic.types import Message, ToolUseBlock, Usage

from cairn.core.models import (
    Confidence,
    Diff,
    Entry,
    EntryStatus,
    EntryType,
    SessionTrace,
    ToolResult,
    Turn,
    TurnRole,
)
from cairn.providers.anthropic import TOOL_NAME, AnthropicProvider

ENDED_AT = datetime(2026, 9, 12, 11, 30, 0, tzinfo=UTC)
SECRET = "sk-" + "A1b2C3d4E5f6G7h8I9j0K1l2"


def _turn(content: str, *, tool_results: list[ToolResult] | None = None) -> Turn:
    return Turn(role=TurnRole.ASSISTANT, content=content, tool_results=tool_results or [])


def _trace(turns: list[Turn]) -> SessionTrace:
    return SessionTrace(
        session_id="session-1",
        harness="claude-code",
        started_at=datetime(2026, 9, 12, 11, 0, 0, tzinfo=UTC),
        ended_at=ENDED_AT,
        turns=turns,
        diffs=[
            Diff(file="alembic/versions/0042_last_seen.py", patch="@@ +last_seen_at"),
            Diff(file=".env.local", patch="DATABASE_URL=postgres://localhost/app"),
        ],
        outcome="success",
    )


def _reflectable_trace() -> SessionTrace:
    return _trace(
        [
            _turn("running the migration tests"),
            _turn(
                "pytest tests/test_migrations.py",
                tool_results=[
                    ToolResult(
                        name="bash",
                        output=f"UndefinedColumn: users.last_seen_at (key {SECRET})",
                        is_error=True,
                    )
                ],
            ),
            _turn("the test database was stale, running alembic upgrade head first"),
            _turn("PASSED tests/test_migrations.py::test_last_seen"),
        ]
    )


def _candidate(title: str, **overrides: object) -> dict[str, object]:
    candidate: dict[str, object] = {
        "id": "stale-test-db",
        "type": "gotcha",
        "title": title,
        "scope": ["alembic/**", "tests/**"],
        "tags": ["database"],
        "confidence": "high",
        "body": f"## What happens\n\n{title}\n\n## What to do\n\nRun `alembic upgrade head`.\n",
    }
    candidate.update(overrides)
    return candidate


def _response(candidates: list[dict[str, object]]) -> Message:
    return Message(
        id="msg_test",
        type="message",
        role="assistant",
        model="claude-sonnet-4-6",
        content=[
            ToolUseBlock(
                id="toolu_test",
                type="tool_use",
                name=TOOL_NAME,
                input={"candidates": candidates},
            )
        ],
        stop_reason="tool_use",
        usage=Usage(input_tokens=1, output_tokens=1),
    )


def _provider(client: MagicMock) -> AnthropicProvider:
    return AnthropicProvider(
        deny_globs=[".env*"],
        patterns=["sk-[A-Za-z0-9]{20,}"],
        min_session_turns=4,
        client=client,
    )


def test_unreflectable_trace_short_circuits_without_api_call() -> None:
    client = MagicMock()
    trace = _trace([_turn(f"routine turn {i}") for i in range(4)])

    candidates = _provider(client).extract(trace, known=[], max_candidates=3)

    assert candidates == []
    client.messages.create.assert_not_called()


def test_two_candidates_convert_to_staged_entries() -> None:
    client = MagicMock()
    client.messages.create.return_value = _response(
        [
            _candidate("Run alembic upgrade head before the migration tests"),
            _candidate(
                "The test database is shared across runs",
                type="fact",
                confidence="medium",
                scope=[],
                tags=[],
            ),
        ]
    )

    candidates = _provider(client).extract(_reflectable_trace(), known=[], max_candidates=3)

    client.messages.create.assert_called_once()
    request = client.messages.create.call_args.kwargs
    assert request["model"] == "claude-sonnet-4-6"
    assert request["max_tokens"] == 2000
    assert request["tool_choice"] == {"type": "tool", "name": TOOL_NAME}

    assert len(candidates) == 2
    (gotcha, gotcha_body), (fact, fact_body) = candidates

    assert re.fullmatch(r"gotcha-[0-9a-f]{6}", gotcha.id)
    assert gotcha.type is EntryType.GOTCHA
    assert gotcha.title == "Run alembic upgrade head before the migration tests"
    assert gotcha.scope == ["alembic/**", "tests/**"]
    assert gotcha.tags == ["database"]
    assert gotcha.confidence is Confidence.HIGH
    assert "Run `alembic upgrade head`." in gotcha_body

    assert re.fullmatch(r"fact-[0-9a-f]{6}", fact.id)
    assert fact.type is EntryType.FACT
    assert fact.confidence is Confidence.MEDIUM
    assert "The test database is shared across runs" in fact_body

    for entry, _ in candidates:
        assert entry.status is EntryStatus.STAGED
        assert entry.spec_version == "0.1.0"
        assert entry.created == ENDED_AT
        assert entry.updated == ENDED_AT
        assert entry.evidence.harness == "claude-code"
        assert entry.evidence.session_id == "session-1"
        assert entry.evidence.captured_at == ENDED_AT
        # the deny-globbed `.env.local` diff never becomes an artifact
        assert entry.evidence.artifacts == ["alembic/versions/0042_last_seen.py"]
        assert re.fullmatch(r"[0-9a-f]{64}", entry.evidence.excerpt_sha256 or "")
        assert Entry.model_validate(entry.model_dump(mode="json")) == entry


def test_trace_is_redacted_before_it_reaches_the_api() -> None:
    client = MagicMock()
    client.messages.create.return_value = _response([])

    _provider(client).extract(_reflectable_trace(), known=[], max_candidates=3)

    prompt = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "UndefinedColumn: users.last_seen_at" in prompt
    assert SECRET not in prompt
    assert "[REDACTED:sk]" in prompt


def test_response_exceeding_max_candidates_is_truncated_in_code() -> None:
    client = MagicMock()
    titles = [f"Distinct lesson number {i}" for i in range(5)]
    client.messages.create.return_value = _response([_candidate(title) for title in titles])

    candidates = _provider(client).extract(_reflectable_trace(), known=[], max_candidates=2)

    assert [entry.title for entry, _ in candidates] == titles[:2]
    tool = client.messages.create.call_args.kwargs["tools"][0]
    assert "at most 2" in tool["description"]


def test_invalid_candidate_is_dropped_with_warning_not_raised(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = MagicMock()
    client.messages.create.return_value = _response(
        [
            _candidate("Run alembic upgrade head before the migration tests"),
            _candidate("A lesson with a confidence Entry does not accept", confidence="certain"),
        ]
    )

    with caplog.at_level(logging.WARNING, logger="cairn.providers.anthropic"):
        candidates = _provider(client).extract(_reflectable_trace(), known=[], max_candidates=3)

    assert [entry.title for entry, _ in candidates] == [
        "Run alembic upgrade head before the migration tests"
    ]
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "dropping invalid candidate" in warnings[0].getMessage()
    assert "confidence" in warnings[0].getMessage()

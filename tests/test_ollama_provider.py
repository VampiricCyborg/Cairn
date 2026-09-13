"""Tests for cairn.providers.ollama.OllamaProvider.

The httpx client is always a fake injected through the constructor, so
nothing here makes a real network call.
"""

import json
import logging
from datetime import UTC, datetime
from unittest.mock import MagicMock

import httpx
import pytest

from cairn.core.models import (
    Confidence,
    Diff,
    EntryStatus,
    EntryType,
    SessionTrace,
    ToolResult,
    Turn,
    TurnRole,
)
from cairn.providers.base import ProviderUnavailableError
from cairn.providers.ollama import OllamaProvider

ENDED_AT = datetime(2026, 9, 12, 11, 30, 0, tzinfo=UTC)


def _turn(content: str, *, tool_results: list[ToolResult] | None = None) -> Turn:
    return Turn(role=TurnRole.ASSISTANT, content=content, tool_results=tool_results or [])


def _reflectable_trace() -> SessionTrace:
    return SessionTrace(
        session_id="session-1",
        harness="claude-code",
        started_at=datetime(2026, 9, 12, 11, 0, 0, tzinfo=UTC),
        ended_at=ENDED_AT,
        turns=[
            _turn("running the migration tests"),
            _turn(
                "pytest tests/test_migrations.py",
                tool_results=[
                    ToolResult(
                        name="bash",
                        output="UndefinedColumn: users.last_seen_at",
                        is_error=True,
                    )
                ],
            ),
            _turn("the test database was stale, running alembic upgrade head first"),
            _turn("PASSED tests/test_migrations.py::test_last_seen"),
        ],
        diffs=[Diff(file="alembic/versions/0042_last_seen.py", patch="@@ +last_seen_at")],
        outcome="success",
    )


def _candidate(title: str, **overrides: object) -> dict[str, object]:
    candidate: dict[str, object] = {
        "id": "stale-test-db",
        "type": "gotcha",
        "title": title,
        "scope": ["alembic/**"],
        "tags": ["database"],
        "confidence": "high",
        "body": "## What happens\n\nStale db.\n\n## What to do\n\nRun `alembic upgrade head`.\n",
    }
    candidate.update(overrides)
    return candidate


def _chat_response(content: str, *, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code,
        json={"message": {"role": "assistant", "content": content}},
        request=httpx.Request("POST", "http://localhost:11434/api/chat"),
    )


def _client(response: httpx.Response) -> MagicMock:
    client = MagicMock()
    client.post.return_value = response
    return client


def _provider(client: MagicMock) -> OllamaProvider:
    return OllamaProvider(min_session_turns=4, client=client)


def test_unreflectable_trace_short_circuits_without_http_call() -> None:
    client = MagicMock()
    trace = _reflectable_trace().model_copy(
        update={"turns": [_turn(f"routine turn {i}") for i in range(4)], "errors": []}
    )

    candidates = _provider(client).extract(trace, known=[], max_candidates=3)

    assert candidates == []
    client.post.assert_not_called()


def test_valid_json_response_converts_to_staged_entries() -> None:
    body = json.dumps({"candidates": [_candidate("Run alembic upgrade head first")]})
    client = _client(_chat_response(body))

    candidates = _provider(client).extract(_reflectable_trace(), known=[], max_candidates=3)

    client.post.assert_called_once()
    request = client.post.call_args
    assert request.args[0] == "/api/chat"
    assert request.kwargs["json"]["model"] == OllamaProvider().model

    assert len(candidates) == 1
    entry, entry_body = candidates[0]
    assert entry.type is EntryType.GOTCHA
    assert entry.title == "Run alembic upgrade head first"
    assert entry.status is EntryStatus.STAGED
    assert entry.confidence is Confidence.HIGH
    assert "alembic upgrade head" in entry_body


def test_markdown_fenced_json_is_repaired() -> None:
    payload = json.dumps({"candidates": [_candidate("Fenced candidate")]})
    fenced = f"```json\n{payload}\n```"
    client = _client(_chat_response(fenced))

    candidates = _provider(client).extract(_reflectable_trace(), known=[], max_candidates=3)

    assert [entry.title for entry, _ in candidates] == ["Fenced candidate"]


def test_unrecoverable_malformed_json_returns_empty_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = _client(_chat_response("not json at all, even after stripping fences ```"))

    with caplog.at_level(logging.WARNING, logger="cairn.providers._json_schema_common"):
        candidates = _provider(client).extract(_reflectable_trace(), known=[], max_candidates=3)

    assert candidates == []
    assert any("treating as zero candidates" in record.getMessage() for record in caplog.records)


def test_connection_refused_raises_provider_unavailable() -> None:
    client = MagicMock()
    client.post.side_effect = httpx.ConnectError("Connection refused")

    with pytest.raises(ProviderUnavailableError, match="ollama serve"):
        _provider(client).extract(_reflectable_trace(), known=[], max_candidates=3)

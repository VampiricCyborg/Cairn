"""Tests for cairn.providers.mock.MockProvider."""

from datetime import UTC, datetime

from cairn.core.models import (
    Confidence,
    Entry,
    EntryStatus,
    EntryType,
    Evidence,
    SessionTrace,
)
from cairn.providers.mock import MockProvider


def _trace(errors: list[str], **overrides: object) -> SessionTrace:
    defaults: dict[str, object] = dict(
        session_id="0f3c1a9e-6b2d-4f77-9a10-2d4c8e51b3aa",
        harness="claude-code",
        started_at=datetime(2026, 9, 12, 11, 0, 0, tzinfo=UTC),
        ended_at=datetime(2026, 9, 12, 11, 30, 0, tzinfo=UTC),
        errors=errors,
        outcome="success",
    )
    defaults.update(overrides)
    return SessionTrace(**defaults)  # type: ignore[arg-type]


def _known_entry(title: str) -> Entry:
    return Entry(
        id="gotcha-000000",
        type=EntryType.GOTCHA,
        title=title,
        status=EntryStatus.APPROVED,
        spec_version="0.1.0",
        confidence=Confidence.HIGH,
        evidence=Evidence(
            harness="manual",
            captured_at=datetime(2026, 9, 1, 0, 0, 0, tzinfo=UTC),
        ),
        created=datetime(2026, 9, 1, 0, 0, 0, tzinfo=UTC),
        updated=datetime(2026, 9, 1, 0, 0, 0, tzinfo=UTC),
    )


def test_extract_is_deterministic() -> None:
    trace = _trace(["UndefinedColumn: users.last_seen_at does not exist"])

    first = MockProvider().extract(trace, known=[], max_candidates=5)
    second = MockProvider().extract(trace, known=[], max_candidates=5)

    assert first == second
    assert len(first) == 1
    assert first[0].id.startswith("gotcha-")


def test_extract_respects_max_candidates() -> None:
    errors = [f"Error number {i}: something distinct broke" for i in range(5)]
    trace = _trace(errors)

    candidates = MockProvider().extract(trace, known=[], max_candidates=3)

    assert len(candidates) == 3
    # distinct errors produce distinct ids
    assert len({entry.id for entry in candidates}) == 3


def test_extract_skips_near_duplicate_of_known_entry() -> None:
    error_text = "Config file missing required key DB_URL"
    trace = _trace([error_text])
    known = [_known_entry(error_text)]

    candidates = MockProvider().extract(trace, known=known, max_candidates=5)

    assert candidates == []


def test_extract_entries_validate_against_entry_model() -> None:
    errors = [f"Error number {i}: something distinct broke" for i in range(3)]
    trace = _trace(errors)

    candidates = MockProvider().extract(trace, known=[], max_candidates=5)

    assert len(candidates) == 3
    for entry in candidates:
        assert Entry.model_validate(entry.model_dump(mode="json")) == entry
        assert entry.status is EntryStatus.STAGED

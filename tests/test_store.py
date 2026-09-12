"""Tests for cairn.core.store.

P0 exit gate: "a hand-written entry validates." `tests/fixtures/entry_valid.md`
is the Alembic gotcha example from documentation/readme.md's "Entry format"
section, hand-copied verbatim; `entry_malformed.md` is the same file with the
required `confidence` field removed.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from cairn.core.models import Confidence, EntryStatus, EntryType
from cairn.core.store import load_entry

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def test_load_entry_valid() -> None:
    entry = load_entry(FIXTURES_DIR / "entry_valid.md")

    assert entry.id == "gotcha-7b21c4"
    assert entry.type is EntryType.GOTCHA
    assert entry.title == "Alembic migrations must run before the test fixtures import"
    assert entry.status is EntryStatus.APPROVED
    assert entry.scope == ["tests/**", "alembic/**"]
    assert entry.tags == ["testing", "database", "ci"]
    assert entry.confidence is Confidence.HIGH
    assert entry.evidence.harness == "claude-code"
    assert entry.evidence.commit == "9d3f1ab"
    assert entry.review is not None
    assert entry.review.approved_by == "madhav"
    assert entry.usage.injected == 12


def test_load_entry_malformed_missing_required_field() -> None:
    with pytest.raises(ValidationError) as exc_info:
        load_entry(FIXTURES_DIR / "entry_malformed.md")

    errors = exc_info.value.errors()
    assert any(error["loc"] == ("confidence",) and error["type"] == "missing" for error in errors)

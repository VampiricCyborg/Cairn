"""Tests for cairn.core.store.

P0 exit gate: "a hand-written entry validates." `tests/fixtures/entry_valid.md`
is the Alembic gotcha example from documentation/readme.md's "Entry format"
section, hand-copied verbatim; `entry_malformed.md` is the same file with the
required `confidence` field removed.
"""

import logging
import os
from datetime import UTC, datetime
from pathlib import Path

import frontmatter
import pytest
from pydantic import ValidationError

from cairn.core.models import Confidence, Entry, EntryStatus, EntryType, Evidence
from cairn.core.store import Store, StoreNotFoundError, load_entry

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


def _make_entry(**overrides: object) -> Entry:
    defaults: dict[str, object] = dict(
        id="fact-abcdef",
        type=EntryType.FACT,
        title="Config values load from .env before defaults",
        status=EntryStatus.APPROVED,
        spec_version="0.1.0",
        confidence=Confidence.MEDIUM,
        evidence=Evidence(
            harness="manual",
            captured_at=datetime(2026, 9, 12, 11, 0, 0, tzinfo=UTC),
        ),
        created=datetime(2026, 9, 12, 11, 0, 0, tzinfo=UTC),
        updated=datetime(2026, 9, 12, 11, 0, 0, tzinfo=UTC),
    )
    defaults.update(overrides)
    return Entry(**defaults)  # type: ignore[arg-type]


def _write_fixture(directory: Path, entry: Entry, filename: str, body: str = "content") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    post = frontmatter.Post(body, **entry.model_dump(mode="json"))
    path = directory / filename
    path.write_text(frontmatter.dumps(post) + "\n", encoding="utf-8")
    return path


def test_store_requires_existing_cairn_dir(tmp_path: Path) -> None:
    with pytest.raises(StoreNotFoundError):
        Store(tmp_path / "does-not-exist")


def test_load_all_returns_all_valid_entries(tmp_path: Path) -> None:
    store = Store(_new_store_root(tmp_path))

    for index, entry_type in enumerate([EntryType.STRATEGY, EntryType.GOTCHA, EntryType.FACT]):
        entry = _make_entry(
            id=f"{entry_type.value}-{index:06x}", type=entry_type, title=f"Entry {index}"
        )
        _write_fixture(store.entries_dir / entry_type.value, entry, f"entry-{index}.md")

    entries = store.load_all()

    assert len(entries) == 3


def test_load_all_logs_type_directory_mismatch(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = Store(_new_store_root(tmp_path))
    mismatched = _make_entry(id="fact-000001", type=EntryType.FACT, title="Mismatched")
    _write_fixture(store.entries_dir / "gotcha", mismatched, "mismatched.md")

    with caplog.at_level(logging.WARNING):
        entries = store.load_all()

    assert entries == []
    assert any(
        "does not match" in record.getMessage() and "gotcha" in record.getMessage()
        for record in caplog.records
    )


def test_write_entry_round_trips_through_load_entry(tmp_path: Path) -> None:
    store = Store(_new_store_root(tmp_path))
    entry = _make_entry()
    body = "## What to know\n\nSomething true and worth remembering.\n"

    path = store.write_entry(entry, body)
    loaded = load_entry(path)

    assert loaded == entry
    assert frontmatter.load(path).content.strip() == body.strip()


def test_write_entry_kill_mid_write_preserves_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = Store(_new_store_root(tmp_path))
    entry = _make_entry()
    original_body = "## Original\n\nThe original body.\n"

    path = store.write_entry(entry, original_body)
    original_bytes = path.read_bytes()

    def _kill(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated kill mid-write")

    monkeypatch.setattr(os, "replace", _kill)

    with pytest.raises(OSError):
        store.write_entry(entry, "## Changed\n\nA different body.\n")

    assert path.read_bytes() == original_bytes
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []


def _new_store_root(tmp_path: Path) -> Path:
    root = tmp_path / ".cairn"
    root.mkdir()
    return root

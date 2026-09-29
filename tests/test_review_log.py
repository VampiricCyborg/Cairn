"""Tests for cairn.core.review_log and `cairn stats --review`.

The two populations in the log must stay separate: human decisions feed
approval rate, extraction-side events feed gate-drop rate, and neither
number may absorb the other's records.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cairn.cli import app
from cairn.core import review_log
from cairn.core.models import Confidence, Entry, EntryStatus, EntryType, Evidence
from cairn.core.review_log import (
    APPROVING_ACTIONS,
    EXTRACTION_ACTIONS,
    HUMAN_ACTIONS,
    ReviewAction,
    ReviewLogRecord,
)

runner = CliRunner()

_TS = datetime(2026, 9, 29, 12, 0, 0, tzinfo=UTC)


def _make_entry(**overrides: object) -> Entry:
    defaults: dict[str, object] = dict(
        id="gotcha-abcdef",
        type=EntryType.GOTCHA,
        title="Migrations must run before fixtures import",
        status=EntryStatus.STAGED,
        spec_version="0.1.0",
        confidence=Confidence.MEDIUM,
        evidence=Evidence(
            harness="claude-code", session_id="sess-1", captured_at=_TS, excerpt_sha256="a" * 64
        ),
        created=_TS,
        updated=_TS,
    )
    defaults.update(overrides)
    return Entry(**defaults)  # type: ignore[arg-type]


def _read_lines(cairn_root: Path) -> list[dict[str, object]]:
    raw = review_log.log_path(cairn_root).read_text(encoding="utf-8")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


# -- the two populations are disjoint -------------------------------------------


def test_human_and_extraction_actions_partition_every_action() -> None:
    assert HUMAN_ACTIONS.isdisjoint(EXTRACTION_ACTIONS)
    assert HUMAN_ACTIONS | EXTRACTION_ACTIONS == set(ReviewAction)


def test_approving_actions_are_all_human() -> None:
    assert APPROVING_ACTIONS <= HUMAN_ACTIONS
    assert ReviewAction.REJECT not in APPROVING_ACTIONS
    assert ReviewAction.SKIP not in APPROVING_ACTIONS


# -- append / read --------------------------------------------------------------


def test_append_writes_one_json_line_per_record(tmp_path: Path) -> None:
    record = review_log.make_record(_make_entry(), ReviewAction.APPROVE, now=_TS)

    assert review_log.append(tmp_path, record) is True
    assert review_log.append(tmp_path, record) is True

    lines = _read_lines(tmp_path)
    assert len(lines) == 2
    assert lines[0] == {
        "ts": "2026-09-29T12:00:00Z",
        "candidate_id": "gotcha-abcdef",
        "type": "gotcha",
        "title": "Migrations must run before fixtures import",
        "action": "approve",
        "reason": None,
        "session_id": "sess-1",
        "harness": "claude-code",
        "model": None,
        "edited_fields": [],
    }


def test_append_never_raises_when_the_log_cannot_be_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed log write must cost the record, never the decision that has
    already been applied to the store."""

    def _explode(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "open", _explode)
    record = review_log.make_record(_make_entry(), ReviewAction.APPROVE, now=_TS)

    assert review_log.append(tmp_path, record) is False


def test_read_round_trips_and_preserves_order(tmp_path: Path) -> None:
    for action in (ReviewAction.STAGED, ReviewAction.APPROVE, ReviewAction.REJECT):
        review_log.append(
            tmp_path, review_log.make_record(_make_entry(), action, reason="why", now=_TS)
        )

    records = review_log.read(tmp_path)

    assert [record.action for record in records] == [
        ReviewAction.STAGED,
        ReviewAction.APPROVE,
        ReviewAction.REJECT,
    ]
    assert all(isinstance(record, ReviewLogRecord) for record in records)


def test_read_of_absent_log_is_empty(tmp_path: Path) -> None:
    assert review_log.read(tmp_path) == []


def test_read_skips_a_truncated_final_line(tmp_path: Path) -> None:
    review_log.append(
        tmp_path, review_log.make_record(_make_entry(), ReviewAction.APPROVE, now=_TS)
    )
    with review_log.log_path(tmp_path).open("a", encoding="utf-8") as handle:
        handle.write('{"ts": "2026-09-29T12:00:00Z", "candida')

    records = review_log.read(tmp_path)

    assert len(records) == 1
    assert records[0].action is ReviewAction.APPROVE


# -- the log is not committed ---------------------------------------------------


def test_init_gitignores_the_review_log(tmp_path: Path) -> None:
    result = runner.invoke(app, ["init", str(tmp_path)])

    assert result.exit_code == 0, result.output
    ignored = (tmp_path / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".cairn/review-log.jsonl" in ignored

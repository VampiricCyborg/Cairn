"""Tests for `cairn review`, driven through CliRunner with scripted input."""

import json
from datetime import UTC, datetime
from pathlib import Path

import frontmatter
import pytest
from typer.testing import CliRunner

from cairn.cli import app
from cairn.core.models import Confidence, Entry, EntryStatus, EntryType, Evidence
from cairn.core.store import Store, load_entry
from cairn.review import cli_review

runner = CliRunner()


def _make_entry(**overrides: object) -> Entry:
    defaults: dict[str, object] = dict(
        id="fact-abcdef",
        type=EntryType.FACT,
        title="Config values load from .env before defaults",
        status=EntryStatus.STAGED,
        spec_version="0.1.0",
        scope=["src/**"],
        confidence=Confidence.MEDIUM,
        evidence=Evidence(
            harness="claude-code",
            session_id="session-1",
            captured_at=datetime(2026, 9, 12, 11, 0, 0, tzinfo=UTC),
            artifacts=["src/config.py"],
        ),
        created=datetime(2026, 9, 12, 11, 0, 0, tzinfo=UTC),
        updated=datetime(2026, 9, 12, 11, 0, 0, tzinfo=UTC),
    )
    defaults.update(overrides)
    return Entry(**defaults)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _reviewer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAIRN_REVIEWER", "tester")


@pytest.fixture
def store(tmp_path: Path) -> Store:
    result = runner.invoke(app, ["init", str(tmp_path)])
    assert result.exit_code == 0, result.output
    return Store(tmp_path / ".cairn")


def _stage(store: Store, count: int) -> list[Path]:
    """Stage `count` fact candidates whose filenames sort in creation order."""

    return [
        store.write_entry(
            _make_entry(id=f"fact-00000{i}", title=f"Candidate {chr(ord('a') + i)}"),
            f"Body of candidate {i}.",
        )
        for i in range(count)
    ]


def _review(store: Store, keys: str) -> tuple[int, str]:
    result = runner.invoke(app, ["review", str(store.root.parent)], input=keys)
    return result.exit_code, result.output


def test_empty_staging_prints_nothing_to_review(store: Store) -> None:
    exit_code, output = _review(store, "")

    assert exit_code == 0, output
    assert "nothing to review" in output
    assert "approve" not in output


def test_displays_candidate_panel(store: Store) -> None:
    _stage(store, 2)

    exit_code, output = _review(store, "q\n")

    assert exit_code == 0, output
    assert "cairn review" in output
    assert "candidate 1 of 2" in output
    assert "[fact] Candidate a" in output
    assert "confidence: medium" in output
    assert "claude-code" in output and "session-1" in output and "src/config.py" in output
    assert "Body of candidate 0." in output


def test_approve_moves_entry_to_entries_dir(store: Store) -> None:
    (staged,) = _stage(store, 1)

    exit_code, output = _review(store, "a\n\n")  # approve, accept default reviewer

    assert exit_code == 0, output
    assert not staged.exists()
    approved_files = list((store.entries_dir / "fact").glob("*.md"))
    assert len(approved_files) == 1
    entry = load_entry(approved_files[0])
    assert entry.status is EntryStatus.APPROVED
    assert entry.review is not None
    assert entry.review.approved_by == "tester"
    assert entry.review.approved_at is not None
    assert frontmatter.load(approved_files[0]).content == "Body of candidate 0."
    assert "1 approved, 0 rejected, 0 skipped, 0 remaining in staging" in output


def test_reject_writes_tombstone_and_removes_staging_file(store: Store) -> None:
    (staged,) = _stage(store, 1)

    exit_code, output = _review(store, "r\nnot project-specific\n")

    assert exit_code == 0, output
    assert not staged.exists()
    tombstone = json.loads((store.rejected_dir / "fact-000000.json").read_text(encoding="utf-8"))
    assert tombstone == {
        "id": "fact-000000",
        "title": "Candidate a",
        "reason": "not project-specific",
        "excerpt_sha256": None,
    }
    assert not list((store.entries_dir / "fact").glob("*.md"))
    assert "0 approved, 1 rejected, 0 skipped, 0 remaining in staging" in output

    # A tombstone is not a full entry, and must not break `cairn validate`.
    validate = runner.invoke(app, ["validate", str(store.root.parent)])
    assert validate.exit_code == 0, validate.output


def test_reject_tombstone_carries_candidates_excerpt_hash(store: Store) -> None:
    entry = _make_entry(
        evidence=Evidence(
            harness="claude-code",
            session_id="session-1",
            captured_at=datetime(2026, 9, 12, 11, 0, 0, tzinfo=UTC),
            excerpt_sha256="a" * 64,
        )
    )
    store.write_entry(entry, "Body.")

    exit_code, output = _review(store, "r\nnot stable\n")

    assert exit_code == 0, output
    tombstone = json.loads((store.rejected_dir / "fact-abcdef.json").read_text(encoding="utf-8"))
    assert tombstone["excerpt_sha256"] == "a" * 64


def test_skip_leaves_staging_file_byte_identical(store: Store) -> None:
    (staged,) = _stage(store, 1)
    before = staged.read_bytes()

    exit_code, output = _review(store, "s\n")

    assert exit_code == 0, output
    assert staged.read_bytes() == before
    assert "0 approved, 0 rejected, 1 skipped, 1 remaining in staging" in output


def test_quit_stops_and_leaves_remaining_untouched(store: Store) -> None:
    staged = _stage(store, 4)
    untouched = {path: path.read_bytes() for path in staged[2:]}

    # approve #1, reject #2, quit on #3
    exit_code, output = _review(store, "a\n\nr\nduplicate\nq\n")

    assert exit_code == 0, output
    assert not staged[0].exists()
    assert not staged[1].exists()
    for path, content in untouched.items():
        assert path.read_bytes() == content
    assert "candidate 3 of 4" in output
    assert "candidate 4 of 4" not in output
    assert "1 approved, 1 rejected, 0 skipped, 2 remaining in staging" in output


def test_merge_amends_target_entry_in_place(store: Store) -> None:
    target = _make_entry(id="fact-999999", title="Existing fact", status=EntryStatus.APPROVED)
    store.write_entry(target, "Existing body.")
    (staged,) = _stage(store, 1)

    # merge, target id, accept default reviewer for the amendment note
    exit_code, output = _review(store, "m\nfact-999999\n\n")

    assert exit_code == 0, output
    assert "fact-999999  Existing fact" in output
    assert not staged.exists()
    assert list(store.rejected_dir.glob("*.json")) == []
    (approved_path,) = (store.entries_dir / "fact").glob("*.md")
    amended = load_entry(approved_path)
    assert amended.id == "fact-999999"
    assert amended.updated > target.updated
    body = frontmatter.load(approved_path).content
    assert "Existing body." in body
    assert "Body of candidate 0." in body
    assert "## Amendment" in body
    assert "1 approved, 0 rejected, 0 skipped, 0 remaining in staging" in output


def test_display_shows_proposed_amendment_marker(store: Store) -> None:
    candidate = _make_entry(proposed_amendment_of="fact-999999")
    store.write_entry(candidate, "New evidence.")

    exit_code, output = _review(store, "q\n")

    assert exit_code == 0, output
    assert "PROPOSED AMENDMENT of fact-999999" in output


def test_display_shows_proposed_supersession_marker(store: Store) -> None:
    candidate = _make_entry(proposed_supersession_of="fact-999999")
    store.write_entry(candidate, "Conflicting evidence.")

    exit_code, output = _review(store, "q\n")

    assert exit_code == 0, output
    assert "PROPOSED SUPERSESSION of fact-999999" in output


def test_approve_amendment_marked_candidate_updates_target_in_place(store: Store) -> None:
    target = _make_entry(id="fact-999999", title="Existing fact", status=EntryStatus.APPROVED)
    store.write_entry(target, "Existing body.")
    candidate = _make_entry(proposed_amendment_of="fact-999999")
    store.write_entry(candidate, "New evidence for the existing fact.")

    exit_code, output = _review(store, "a\n\n")  # approve, accept default reviewer

    assert exit_code == 0, output
    assert not list(store.staging_dir.glob("*.md"))
    approved_files = list((store.entries_dir / "fact").glob("*.md"))
    assert len(approved_files) == 1  # amended in place, not a second entry
    amended = load_entry(approved_files[0])
    assert amended.id == "fact-999999"
    body = frontmatter.load(approved_files[0]).content
    assert "Existing body." in body
    assert "New evidence for the existing fact." in body
    assert "amended fact-999999" in output


def test_approve_amendment_with_missing_target_falls_back_to_new_entry(store: Store) -> None:
    candidate = _make_entry(proposed_amendment_of="fact-doesnotexist")
    store.write_entry(candidate, "Evidence with no live target.")

    exit_code, output = _review(store, "a\n\n")  # approve, accept default reviewer

    assert exit_code == 0, output
    assert "approving fact-abcdef as a new entry instead" in output
    approved_files = list((store.entries_dir / "fact").glob("*.md"))
    assert len(approved_files) == 1
    approved = load_entry(approved_files[0])
    assert approved.status is EntryStatus.APPROVED
    assert approved.proposed_amendment_of is None


def test_invalid_edit_asks_to_retry_then_approves(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    (staged,) = _stage(store, 1)
    calls: list[str] = []

    def fake_editor(path: Path) -> None:
        calls.append(path.read_text(encoding="utf-8"))
        if len(calls) == 1:
            # Drop the required `confidence` field.
            post = frontmatter.load(path)
            del post.metadata["confidence"]
            path.write_text(frontmatter.dumps(post), encoding="utf-8")
        else:
            post = frontmatter.load(path)
            post.metadata["confidence"] = "high"
            post.metadata["title"] = "Edited title"
            post.content = "Edited body."
            path.write_text(frontmatter.dumps(post), encoding="utf-8")

    monkeypatch.setattr(cli_review, "_launch_editor", fake_editor)

    exit_code, output = _review(store, "e\nr\n\n")  # edit, retry, default reviewer

    assert exit_code == 0, output
    assert "edited entry is invalid" in output
    assert "confidence" in output
    assert len(calls) == 2
    # The retry reopens the user's broken edit rather than discarding it.
    assert "confidence" not in frontmatter.loads(calls[1]).metadata
    assert not staged.exists()
    (approved,) = (store.entries_dir / "fact").glob("*.md")
    entry = load_entry(approved)
    assert entry.title == "Edited title"
    assert entry.confidence is Confidence.HIGH
    assert entry.status is EntryStatus.APPROVED
    assert frontmatter.load(approved).content == "Edited body."


def test_abandoned_edit_leaves_candidate_staged(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    (staged,) = _stage(store, 1)
    before = staged.read_bytes()

    def broken_editor(path: Path) -> None:
        path.write_text("---\nid: [unclosed\n---\nbody\n", encoding="utf-8")

    monkeypatch.setattr(cli_review, "_launch_editor", broken_editor)

    exit_code, output = _review(store, "e\na\ns\n")  # edit, abandon, skip

    assert exit_code == 0, output
    assert "edited entry is invalid" in output
    assert staged.read_bytes() == before
    assert "0 approved, 0 rejected, 1 skipped, 1 remaining in staging" in output

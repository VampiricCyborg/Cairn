"""Tests for cairn.core.curator."""

import json
from datetime import UTC, datetime
from pathlib import Path

from cairn.core.curator import Curator
from cairn.core.models import Confidence, Entry, EntryStatus, EntryType, Evidence
from cairn.core.store import Store, load_entry

_CAPTURED_AT = datetime(2026, 9, 12, 11, 0, 0, tzinfo=UTC)


def _make_entry(**overrides: object) -> Entry:
    defaults: dict[str, object] = dict(
        id="fact-abcdef",
        type=EntryType.FACT,
        title="Config values load from .env before defaults",
        status=EntryStatus.STAGED,
        spec_version="0.1.0",
        confidence=Confidence.MEDIUM,
        evidence=Evidence(harness="claude-code", session_id="session-1", captured_at=_CAPTURED_AT),
        created=_CAPTURED_AT,
        updated=_CAPTURED_AT,
    )
    defaults.update(overrides)
    return Entry(**defaults)  # type: ignore[arg-type]


def _new_store(tmp_path: Path) -> Store:
    root = tmp_path / ".cairn"
    root.mkdir()
    return Store(root)


def _write_tombstone(store: Store, **fields: object) -> Path:
    store.rejected_dir.mkdir(parents=True, exist_ok=True)
    tombstone = {"id": "fact-000001", "title": "Some rejected lesson", "reason": "not stable"}
    tombstone.update(fields)
    path = store.rejected_dir / f"{tombstone['id']}.json"
    path.write_text(json.dumps(tombstone, indent=2) + "\n", encoding="utf-8")
    return path


# -- check_tombstoned ---------------------------------------------------------------


def test_check_tombstoned_matches_evidence_hash(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    _write_tombstone(store, excerpt_sha256="a" * 64, title="Unrelated title entirely")
    curator = Curator(store)
    candidate = _make_entry(
        title="A totally different title",
        evidence=Evidence(harness="claude-code", captured_at=_CAPTURED_AT, excerpt_sha256="a" * 64),
    )

    assert curator.check_tombstoned(candidate) is True


def test_check_tombstoned_matches_similar_title_with_different_evidence(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    _write_tombstone(
        store,
        title="Auth tokens expire after 15 minutes in staging",
        excerpt_sha256="b" * 64,
    )
    curator = Curator(store)
    candidate = _make_entry(
        title="Auth tokens expire after 15 minutes in staging",
        evidence=Evidence(harness="claude-code", captured_at=_CAPTURED_AT, excerpt_sha256="c" * 64),
    )

    assert curator.check_tombstoned(candidate) is True


def test_check_tombstoned_false_for_unrelated_candidate(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    _write_tombstone(store, title="Some rejected lesson", excerpt_sha256="b" * 64)
    curator = Curator(store)
    candidate = _make_entry(title="A completely unrelated lesson about caching")

    assert curator.check_tombstoned(candidate) is False


def test_check_tombstoned_false_when_no_tombstones_exist(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    curator = Curator(store)

    assert curator.check_tombstoned(_make_entry()) is False


# -- find_near_duplicate ------------------------------------------------------------


def test_find_near_duplicate_exact_title_match(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    curator = Curator(store)
    existing = _make_entry(id="fact-111111", status=EntryStatus.APPROVED)
    candidate = _make_entry(id="fact-222222", title=existing.title)

    assert curator.find_near_duplicate(candidate, [existing]) is existing


def test_find_near_duplicate_gray_zone_confirmed_by_body(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    curator = Curator(store)
    existing = _make_entry(
        id="fact-111111",
        title="Staging auth tokens expire after 15 minutes",
        status=EntryStatus.APPROVED,
    )
    store.write_entry(
        existing,
        "Staging tokens are short-lived by design, so refresh bugs surface early during review.",
    )
    candidate = _make_entry(
        id="fact-222222",
        title="Staging auth tokens expire after fifteen minutes, not thirty",
    )
    candidate_body = (
        "Staging tokens are short lived by design so that refresh bugs surface early during review."
    )

    found = curator.find_near_duplicate(candidate, [existing], candidate_body=candidate_body)

    assert found is existing


def test_find_near_duplicate_gray_zone_without_body_corroboration_is_none(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    curator = Curator(store)
    existing = _make_entry(
        id="fact-111111",
        title="Staging auth tokens expire after 15 minutes",
        status=EntryStatus.APPROVED,
    )
    store.write_entry(existing, "Original unrelated body text about token refresh internals.")
    candidate = _make_entry(
        id="fact-222222",
        title="Staging auth tokens expire after fifteen minutes, not thirty",
    )

    # No candidate_body passed: title alone is in the gray zone, not enough on its own.
    assert curator.find_near_duplicate(candidate, [existing]) is None


def test_find_near_duplicate_returns_none_for_distinct_titles(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    curator = Curator(store)
    existing = _make_entry(
        id="fact-111111",
        title="Auth tokens expire after 15 minutes in staging",
        status=EntryStatus.APPROVED,
    )
    candidate = _make_entry(
        id="fact-222222", title="Retry logic in the auth client swallows 429 responses"
    )

    assert curator.find_near_duplicate(candidate, [existing]) is None


# -- find_contradiction -------------------------------------------------------------


def test_find_contradiction_flags_structural_match(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    curator = Curator(store)
    existing = _make_entry(
        id="fact-111111",
        title="Auth tokens expire after 15 minutes in staging",
        scope=["src/auth/**"],
        tags=["auth", "staging"],
        status=EntryStatus.APPROVED,
    )
    candidate = _make_entry(
        id="fact-222222",
        title="Retry logic in the auth client swallows 429 responses",
        scope=["src/auth/**"],
        tags=["auth", "staging", "retries"],
    )

    assert curator.find_contradiction(candidate, [existing]) is existing


def test_find_contradiction_requires_scope_overlap(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    curator = Curator(store)
    existing = _make_entry(
        id="fact-111111",
        title="Auth tokens expire after 15 minutes in staging",
        scope=["src/auth/**"],
        tags=["auth", "staging"],
        status=EntryStatus.APPROVED,
    )
    candidate = _make_entry(
        id="fact-222222",
        title="Retry logic in the auth client swallows 429 responses",
        scope=["src/billing/**"],
        tags=["auth", "staging", "retries"],
    )

    assert curator.find_contradiction(candidate, [existing]) is None


def test_find_contradiction_requires_same_type(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    curator = Curator(store)
    existing = _make_entry(
        id="gotcha-111111",
        type=EntryType.GOTCHA,
        title="Auth tokens expire after 15 minutes in staging",
        scope=["src/auth/**"],
        tags=["auth", "staging"],
        status=EntryStatus.APPROVED,
    )
    candidate = _make_entry(
        id="fact-222222",
        type=EntryType.FACT,
        title="Retry logic in the auth client swallows 429 responses",
        scope=["src/auth/**"],
        tags=["auth", "staging", "retries"],
    )

    assert curator.find_contradiction(candidate, [existing]) is None


def test_find_contradiction_excludes_near_duplicate_titles(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    curator = Curator(store)
    existing = _make_entry(
        id="fact-111111",
        title="Auth tokens expire after 15 minutes in staging",
        scope=["src/auth/**"],
        tags=["auth", "staging"],
        status=EntryStatus.APPROVED,
    )
    candidate = _make_entry(
        id="fact-222222",
        title="Auth tokens expire after 15 minutes in staging",
        scope=["src/auth/**"],
        tags=["auth", "staging"],
    )

    # Same title -> near-duplicate territory, not a contradiction candidate.
    assert curator.find_contradiction(candidate, [existing]) is None


def test_find_contradiction_requires_tag_overlap(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    curator = Curator(store)
    existing = _make_entry(
        id="fact-111111",
        title="Auth tokens expire after 15 minutes in staging",
        scope=["src/auth/**"],
        tags=["auth", "staging"],
        status=EntryStatus.APPROVED,
    )
    candidate = _make_entry(
        id="fact-222222",
        title="Retry logic in the auth client swallows 429 responses",
        scope=["src/auth/**"],
        tags=["unrelated", "topic"],
    )

    assert curator.find_contradiction(candidate, [existing]) is None


# -- stage_candidate ----------------------------------------------------------------


def test_stage_candidate_drops_when_evidence_hash_tombstoned(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    _write_tombstone(store, excerpt_sha256="a" * 64, title="Unrelated")
    curator = Curator(store)
    candidate = _make_entry(
        evidence=Evidence(harness="claude-code", captured_at=_CAPTURED_AT, excerpt_sha256="a" * 64)
    )

    result = curator.stage_candidate(candidate, "Body text.")

    assert result.outcome == "dropped_tombstoned"
    assert result.written_path is None
    assert result.related_entry_id is None
    assert list(store.staging_dir.glob("*.md")) == []


def test_stage_candidate_drops_when_title_matches_tombstoned_title(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    _write_tombstone(
        store, title="Auth tokens expire after 15 minutes in staging", excerpt_sha256="b" * 64
    )
    curator = Curator(store)
    candidate = _make_entry(
        title="Auth tokens expire after 15 minutes in staging",
        evidence=Evidence(harness="claude-code", captured_at=_CAPTURED_AT, excerpt_sha256="z" * 64),
    )

    result = curator.stage_candidate(candidate, "Body text.")

    assert result.outcome == "dropped_tombstoned"
    assert list(store.staging_dir.glob("*.md")) == []


def test_stage_candidate_near_duplicate_of_approved_becomes_amendment(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    curator = Curator(store)
    existing = _make_entry(id="fact-111111", status=EntryStatus.APPROVED)
    store.write_entry(existing, "Existing body.")
    candidate = _make_entry(id="fact-222222", title=existing.title)

    result = curator.stage_candidate(candidate, "New evidence for the same fact.")

    assert result.outcome == "amendment"
    assert result.related_entry_id == existing.id
    assert result.written_path is not None
    staged = load_entry(result.written_path)
    assert staged.status is EntryStatus.STAGED
    assert staged.proposed_amendment_of == existing.id
    assert staged.proposed_supersession_of is None


def test_stage_candidate_structural_contradiction_becomes_supersession_not_auto_approved(
    tmp_path: Path,
) -> None:
    store = _new_store(tmp_path)
    curator = Curator(store)
    existing = _make_entry(
        id="fact-111111",
        title="Auth tokens expire after 15 minutes in staging",
        scope=["src/auth/**"],
        tags=["auth", "staging"],
        status=EntryStatus.APPROVED,
    )
    store.write_entry(existing, "Existing body.")
    candidate = _make_entry(
        id="fact-222222",
        title="Retry logic in the auth client swallows 429 responses",
        scope=["src/auth/**"],
        tags=["auth", "staging", "retries"],
    )

    result = curator.stage_candidate(candidate, "New conflicting evidence.")

    assert result.outcome == "supersession"
    assert result.related_entry_id == existing.id
    assert result.written_path is not None
    staged = load_entry(result.written_path)
    assert staged.status is EntryStatus.STAGED
    assert staged.proposed_supersession_of == existing.id
    assert staged.proposed_amendment_of is None
    # Not auto-approved: the target's status is untouched, and the candidate
    # itself stays staged for a human to decide.
    (approved_path,) = (store.entries_dir / "fact").glob("*.md")
    assert load_entry(approved_path).status is EntryStatus.APPROVED


def test_stage_candidate_novel_candidate_stages_normally(tmp_path: Path) -> None:
    store = _new_store(tmp_path)
    curator = Curator(store)
    candidate = _make_entry(title="A genuinely novel lesson nobody has proposed before")

    result = curator.stage_candidate(candidate, "Fresh body.")

    assert result.outcome == "new"
    assert result.related_entry_id is None
    assert result.written_path is not None
    staged = load_entry(result.written_path)
    assert staged.status is EntryStatus.STAGED
    assert staged.proposed_amendment_of is None
    assert staged.proposed_supersession_of is None


def test_stage_candidate_checks_staged_peers_too(tmp_path: Path) -> None:
    """The gap test_reflect_second_session_same_error_duplicates_in_staging
    surfaced: a near-duplicate of a peer still sitting in staging/ (not yet
    approved) must not be re-staged as an unrelated second candidate."""

    store = _new_store(tmp_path)
    curator = Curator(store)
    first = _make_entry(id="fact-111111", title="A repeated lesson from session one")
    store.write_entry(first, "Body from the first session.")

    second = _make_entry(id="fact-222222", title="A repeated lesson from session one")
    result = curator.stage_candidate(second, "Body from a second, later session.")

    assert result.outcome == "amendment"
    assert result.related_entry_id == first.id

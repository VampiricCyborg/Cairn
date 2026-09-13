"""End-to-end test for `cairn reflect`: init -> reflect -> re-load staged entries.

This is the P1 exit gate: a synthetic session trace, run through the real CLI
and the real MockProvider, must land as valid, re-loadable staged entries.
"""

from pathlib import Path

from typer.testing import CliRunner

from cairn.cli import app
from cairn.core.models import EntryStatus
from cairn.core.store import Store, load_entry

runner = CliRunner()

_FIXTURE = Path(__file__).parent / "fixtures" / "session_trace.json"
_FIXTURE_DISTINCT_ERRORS = 2  # see tests/fixtures/session_trace.json


def _init(tmp_path: Path) -> None:
    result = runner.invoke(app, ["init", str(tmp_path)])
    assert result.exit_code == 0, result.output


def test_reflect_stages_valid_entries_from_trace(tmp_path: Path) -> None:
    _init(tmp_path)

    result = runner.invoke(app, ["reflect", str(tmp_path), "--trace", str(_FIXTURE)])

    assert result.exit_code == 0, result.output
    assert f"{_FIXTURE_DISTINCT_ERRORS} entries staged" in result.output

    store = Store(tmp_path / ".cairn")
    staged_files = sorted(store.staging_dir.glob("*.md"))
    assert len(staged_files) == _FIXTURE_DISTINCT_ERRORS

    for path in staged_files:
        entry = load_entry(path)
        assert entry.status is EntryStatus.STAGED
        assert entry.evidence.session_id
        assert entry.evidence.captured_at is not None


def test_reflect_caps_at_max_candidates(tmp_path: Path) -> None:
    _init(tmp_path)

    result = runner.invoke(
        app,
        ["reflect", str(tmp_path), "--trace", str(_FIXTURE), "--max-candidates", "1"],
    )

    assert result.exit_code == 0, result.output
    assert "1 entries staged" in result.output

    store = Store(tmp_path / ".cairn")
    assert len(list(store.staging_dir.glob("*.md"))) == 1


def test_reflect_missing_store_errors_cleanly(tmp_path: Path) -> None:
    result = runner.invoke(app, ["reflect", str(tmp_path), "--trace", str(_FIXTURE)])

    assert result.exit_code == 1
    assert "error" in result.output.lower()


def test_reflect_rerun_same_session_does_not_grow_staging(tmp_path: Path) -> None:
    """Re-running `reflect` with the *same* trace file (same session_id) must
    not keep piling entries into staging/.

    NOTE: this passes, but only incidentally. `MockProvider` derives an
    entry's id (and therefore `Store._target_path`'s filename) from
    `f"{trace.session_id}:{error_text}"`, so replaying the identical trace
    produces the identical id/filename and the second write overwrites the
    first in place. Nothing here is a real near-duplicate check: the
    provider's dedup only ever consults `known` (here `store.approved()`),
    never `staging/`. See test_reflect_second_session_same_error_duplicates_in_staging
    for the real gap this masks.
    """

    _init(tmp_path)

    first = runner.invoke(app, ["reflect", str(tmp_path), "--trace", str(_FIXTURE)])
    assert first.exit_code == 0, first.output

    store = Store(tmp_path / ".cairn")
    after_first = sorted(p.name for p in store.staging_dir.glob("*.md"))
    assert len(after_first) == _FIXTURE_DISTINCT_ERRORS

    second = runner.invoke(app, ["reflect", str(tmp_path), "--trace", str(_FIXTURE)])
    assert second.exit_code == 0, second.output

    after_second = sorted(p.name for p in store.staging_dir.glob("*.md"))
    assert after_second == after_first


def test_reflect_second_session_same_error_duplicates_in_staging(tmp_path: Path) -> None:
    """The real gap: two *distinct* sessions hitting the same underlying error
    each get their own entry id (id is keyed on session_id), so the staged
    near-duplicate check -- which only looks at `store.approved()`, never
    `staging/` -- does not stop the second session's candidate from being
    staged alongside the first. This is a real P1/P2 gap in the mock
    provider's dedup, not something this test suite works around.
    """

    _init(tmp_path)
    raw = _FIXTURE.read_text(encoding="utf-8")

    first = runner.invoke(app, ["reflect", str(tmp_path), "--trace", str(_FIXTURE)])
    assert first.exit_code == 0, first.output

    second_trace = tmp_path / "session_trace_2.json"
    second_trace.write_text(
        raw.replace(
            "b1a2c3d4-5e6f-4a1b-9c3d-7f8e9a0b1c2d",
            "c2b3d4e5-6f70-4a1b-9c3d-7f8e9a0b1c2e",
        ),
        encoding="utf-8",
    )

    second = runner.invoke(app, ["reflect", str(tmp_path), "--trace", str(second_trace)])
    assert second.exit_code == 0, second.output

    store = Store(tmp_path / ".cairn")
    staged_titles = [load_entry(p).title for p in store.staging_dir.glob("*.md")]

    # Same underlying errors, two different (unreviewed) sessions -> the same
    # gotcha title now appears twice in staging/, unstopped by the provider's
    # known-only dedup check.
    assert len(staged_titles) == 2 * _FIXTURE_DISTINCT_ERRORS
    assert len(set(staged_titles)) == _FIXTURE_DISTINCT_ERRORS

"""`cairn reflect` stages through the Curator, exactly like the SessionStart sweep.

Every candidate, from every code path, goes through `Curator.stage_candidate`
and lands in `staging/`; nothing on the extraction side may write to
`entries/`. These tests pin that for the manual `reflect` path, which used to
call `Store.write_entry` directly and so skipped the tombstone, near-duplicate
and contradiction gates.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cairn import cli
from cairn.cli import app
from cairn.core.models import (
    Confidence,
    Entry,
    EntryStatus,
    EntryType,
    Evidence,
    SessionTrace,
)
from cairn.core.store import Store, load_entry
from cairn.providers.mock import MockProvider

runner = CliRunner()

_FIXTURE = Path(__file__).parent / "fixtures" / "session_trace.json"
_TRACE = SessionTrace.model_validate(json.loads(_FIXTURE.read_text(encoding="utf-8")))

# The titles MockProvider gives the fixture's two distinct tool errors, in
# trace order (it truncates long first lines, so they are derived, not typed).
_ERROR_1_TITLE, _ERROR_2_TITLE = (
    entry.title for entry, _body in MockProvider().extract(_TRACE, known=[], max_candidates=3)
)


def _make_entry(**overrides: object) -> Entry:
    defaults: dict[str, object] = dict(
        id="gotcha-abcdef",
        type=EntryType.GOTCHA,
        title="A lesson",
        status=EntryStatus.STAGED,
        spec_version="0.1.0",
        scope=[],
        confidence=Confidence.MEDIUM,
        evidence=Evidence(
            harness="claude-code",
            session_id="session-1",
            captured_at=datetime(2026, 9, 12, 11, 0, 0, tzinfo=UTC),
        ),
        created=datetime(2026, 9, 12, 11, 0, 0, tzinfo=UTC),
        updated=datetime(2026, 9, 12, 11, 0, 0, tzinfo=UTC),
    )
    defaults.update(overrides)
    return Entry(**defaults)  # type: ignore[arg-type]


def _init(root: Path) -> Store:
    root.mkdir(parents=True, exist_ok=True)
    result = runner.invoke(app, ["init", str(root)])
    assert result.exit_code == 0, result.output
    return Store(root / ".cairn")


def _files_under(directory: Path) -> list[Path]:
    """Files under `directory`, ignoring the .gitkeep placeholders that keep the
    committed store layout intact."""

    return sorted(
        path for path in directory.rglob("*") if path.is_file() and path.name != ".gitkeep"
    )


def _write_tombstone(store: Store, entry_id: str, title: str) -> None:
    store.rejected_dir.mkdir(parents=True, exist_ok=True)
    tombstone = {"id": entry_id, "title": title, "reason": "not stable", "excerpt_sha256": None}
    (store.rejected_dir / f"{entry_id}.json").write_text(json.dumps(tombstone), encoding="utf-8")


def _snapshot(directory: Path) -> dict[str, str]:
    return {path.name: path.read_text(encoding="utf-8") for path in sorted(directory.glob("*.md"))}


def _seed_gate_state(store: Store) -> None:
    """A tombstone for the fixture's first error, plus a staged peer that
    near-duplicates its second: between them, every gate has something to do."""

    _write_tombstone(store, "gotcha-dead01", _ERROR_1_TITLE)
    store.write_staged(
        _make_entry(id="gotcha-peer01", title=_ERROR_2_TITLE),
        "## What happens\n\nA peer candidate still awaiting review.\n",
    )


def test_reflect_never_creates_a_file_under_entries(tmp_path: Path) -> None:
    store = _init(tmp_path)

    result = runner.invoke(app, ["reflect", str(tmp_path), "--trace", str(_FIXTURE)])

    assert result.exit_code == 0, result.output
    assert len(list(store.staging_dir.glob("*.md"))) == 2
    assert _files_under(store.entries_dir) == []


def test_reflect_leaves_existing_trusted_entries_untouched(tmp_path: Path) -> None:
    store = _init(tmp_path)
    approved = _make_entry(
        id="fact-111111", type=EntryType.FACT, title="Unrelated", status=EntryStatus.APPROVED
    )
    store.write_trusted(approved, "Existing body.\n")
    before = {path: path.read_bytes() for path in _files_under(store.entries_dir)}

    result = runner.invoke(app, ["reflect", str(tmp_path), "--trace", str(_FIXTURE)])

    assert result.exit_code == 0, result.output
    after = {path: path.read_bytes() for path in _files_under(store.entries_dir)}
    assert after == before


def test_a_provider_cannot_route_a_non_staged_candidate_into_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even a misbehaving provider that returns an already-approved entry
    can't reach `entries/` through reflect: the extraction path has no
    trusted-store writer to reach it with."""

    class _ApprovedEmitter:
        def extract(
            self, trace: SessionTrace, known: list[Entry], max_candidates: int
        ) -> list[tuple[Entry, str]]:
            return [(_make_entry(status=EntryStatus.APPROVED), "Sneaky body.\n")]

    store = _init(tmp_path)
    monkeypatch.setattr(cli, "_resolve_provider", lambda _root: _ApprovedEmitter())

    result = runner.invoke(app, ["reflect", str(tmp_path), "--trace", str(_FIXTURE)])

    assert result.exit_code != 0
    assert _files_under(store.entries_dir) == []
    assert list(store.staging_dir.glob("*.md")) == []


def test_reflect_drops_a_candidate_matching_a_rejected_tombstone(tmp_path: Path) -> None:
    store = _init(tmp_path)
    _write_tombstone(store, "gotcha-dead01", _ERROR_1_TITLE)

    result = runner.invoke(app, ["reflect", str(tmp_path), "--trace", str(_FIXTURE)])

    assert result.exit_code == 0, result.output
    staged_titles = [load_entry(path).title for path in sorted(store.staging_dir.glob("*.md"))]
    assert staged_titles == [_ERROR_2_TITLE]
    assert "dropped" in result.output
    assert "1 entries staged, 1 dropped" in result.output


def test_reflect_flags_a_near_duplicate_of_a_staged_peer_as_an_amendment(tmp_path: Path) -> None:
    store = _init(tmp_path)
    _seed_gate_state(store)

    result = runner.invoke(app, ["reflect", str(tmp_path), "--trace", str(_FIXTURE)])

    assert result.exit_code == 0, result.output
    by_title = {
        entry.title: entry
        for entry in (load_entry(path) for path in sorted(store.staging_dir.glob("*.md")))
        if entry.id != "gotcha-peer01"
    }
    assert set(by_title) == {_ERROR_2_TITLE}
    assert by_title[_ERROR_2_TITLE].proposed_amendment_of == "gotcha-peer01"
    assert "proposes amendment of gotcha-peer01" in result.output


def test_reflect_and_sweep_produce_identical_staging_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace = _TRACE

    reflect_store = _init(tmp_path / "reflect")
    _seed_gate_state(reflect_store)
    reflected = runner.invoke(app, ["reflect", str(tmp_path / "reflect"), "--trace", str(_FIXTURE)])
    assert reflected.exit_code == 0, reflected.output

    sweep_store = _init(tmp_path / "sweep")
    _seed_gate_state(sweep_store)
    sweep_store.queue_dir.mkdir(parents=True, exist_ok=True)
    (sweep_store.queue_dir / "sess-1.json").write_text(
        json.dumps({"session_id": trace.session_id, "transcript_path": "unused"}),
        encoding="utf-8",
    )
    # The sweep normalizes a harness transcript; hand it the same trace
    # `reflect` reads, so the only difference left is the code path.
    monkeypatch.setattr(cli, "normalize", lambda *_args, **_kwargs: trace)
    swept = runner.invoke(app, ["context", str(tmp_path / "sweep"), "--hook"])
    assert swept.exit_code == 0, swept.output

    reflect_staging = _snapshot(reflect_store.staging_dir)
    sweep_staging = _snapshot(sweep_store.staging_dir)

    # Guard against an empty-equals-empty pass: the gates really ran.
    assert len(reflect_staging) == 2  # the seeded peer + the un-tombstoned candidate
    assert reflect_staging == sweep_staging
    assert _files_under(reflect_store.entries_dir) == _files_under(sweep_store.entries_dir) == []

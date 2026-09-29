"""The store's write surface is narrow by construction, not by convention.

Nothing that extracts or stages candidates can write into `entries/`:
`Store.write_staged` only ever writes `staging/` and refuses any entry that
is not `staged`; the one method that writes `entries/`, `Store.write_trusted`,
refuses `staged` entries and is only callable from the review flow -- which
the AST test below enforces for every module under `cairn/`.
"""

import ast
from datetime import UTC, datetime
from pathlib import Path

import pytest

from cairn.core.models import Confidence, Entry, EntryStatus, EntryType, Evidence
from cairn.core.store import Store, StoreError

_CAIRN_PACKAGE = Path(__file__).resolve().parent.parent / "cairn"
_CAPTURED_AT = datetime(2026, 9, 12, 11, 0, 0, tzinfo=UTC)

#: Package directories allowed to call `Store.write_trusted`.
_TRUSTED_WRITERS = ("review",)


def _make_entry(status: EntryStatus) -> Entry:
    return Entry(
        id="fact-abcdef",
        type=EntryType.FACT,
        title="A lesson",
        status=status,
        spec_version="0.1.0",
        confidence=Confidence.MEDIUM,
        evidence=Evidence(harness="claude-code", captured_at=_CAPTURED_AT),
        created=_CAPTURED_AT,
        updated=_CAPTURED_AT,
    )


@pytest.fixture
def store(tmp_path: Path) -> Store:
    root = tmp_path / ".cairn"
    root.mkdir()
    return Store(root)


def _all_files(store: Store) -> list[Path]:
    return sorted(path for path in store.root.rglob("*") if path.is_file())


def test_write_staged_writes_only_under_staging(store: Store) -> None:
    path = store.write_staged(_make_entry(EntryStatus.STAGED), "Body.\n")

    assert path.parent == store.staging_dir
    assert _all_files(store) == [path]


@pytest.mark.parametrize(
    "status",
    [EntryStatus.APPROVED, EntryStatus.REJECTED, EntryStatus.SUPERSEDED, EntryStatus.ARCHIVED],
)
def test_write_staged_refuses_anything_that_is_not_staged(
    store: Store, status: EntryStatus
) -> None:
    with pytest.raises(StoreError, match="staged"):
        store.write_staged(_make_entry(status), "Body.\n")

    assert _all_files(store) == []


def test_write_trusted_writes_only_under_entries(store: Store) -> None:
    path = store.write_trusted(_make_entry(EntryStatus.APPROVED), "Body.\n")

    assert path.parent == store.entries_dir / "fact"
    assert _all_files(store) == [path]


@pytest.mark.parametrize("status", [EntryStatus.STAGED, EntryStatus.REJECTED])
def test_write_trusted_refuses_staged_and_rejected(store: Store, status: EntryStatus) -> None:
    with pytest.raises(StoreError, match="trusted"):
        store.write_trusted(_make_entry(status), "Body.\n")

    assert _all_files(store) == []


def test_the_status_dispatching_write_entry_is_gone() -> None:
    assert not hasattr(Store, "write_entry")


def _callers_of(method: str) -> set[str]:
    """Package-relative paths of every `cairn/` module that calls `.method(...)`."""

    callers: set[str] = set()
    for source in sorted(_CAIRN_PACKAGE.rglob("*.py")):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == method
            ):
                callers.add(source.relative_to(_CAIRN_PACKAGE).as_posix())
    return callers


def test_only_the_review_flow_calls_write_trusted() -> None:
    callers = _callers_of("write_trusted")

    assert callers, "expected the review flow to call write_trusted"
    outside_review = {caller for caller in callers if caller.split("/")[0] not in _TRUSTED_WRITERS}
    assert outside_review == set(), (
        f"{sorted(outside_review)} call Store.write_trusted, but only modules under "
        f"{_TRUSTED_WRITERS} may write to entries/"
    )

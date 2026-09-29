"""Evidence is Cairn's to fill, not the model's.

The model supplies `title`, `body`, `type`, `tags` and `confidence`, and picks
`artifacts` out of a closed list it is given. Everything that makes an entry
auditable -- the commit, the hash of the excerpt actually sent, the session id,
the capture time -- is computed here. Evidence a model can write is evidence a
model can invent, and an entry whose provenance is invented is worse than no
entry, because it still looks auditable in a diff.
"""

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from cairn.core.models import Diff, EntryType, SessionTrace, Turn, TurnRole
from cairn.providers._json_schema_common import (
    build_evidence,
    derive_scope,
    raw_candidates_to_entries,
    touched_files,
)

_ENDED_AT = datetime(2026, 9, 29, 12, 0, 0, tzinfo=UTC)
_ENQUEUE = (
    Path(__file__).resolve().parent.parent / "cairn" / "adapters" / "claude_code" / "enqueue.py"
)


def _trace(**overrides: object) -> SessionTrace:
    defaults: dict[str, object] = dict(
        session_id="session-1",
        harness="claude-code",
        started_at=_ENDED_AT,
        ended_at=_ENDED_AT,
        turns=[Turn(role=TurnRole.USER, content="add a column")],
        errors=[],
        diffs=[
            Diff(file="tests/conftest.py", patch="@@ fixture"),
            Diff(file="alembic/env.py", patch="@@ env"),
        ],
        outcome="success",
    )
    defaults.update(overrides)
    return SessionTrace(**defaults)  # type: ignore[arg-type]


def _candidate(**overrides: object) -> dict:
    record: dict = {
        "id": "migrations-before-fixtures",
        "type": "gotcha",
        "title": "Run alembic upgrade head before the fixtures import",
        "artifacts": ["tests/conftest.py"],
        "scope": [],
        "tags": ["database"],
        "confidence": "high",
        "body": "## What happens\n\nThe fixture sees the pre-migration schema.\n",
    }
    record.update(overrides)
    return record


def _convert(raw: dict, trace: SessionTrace | None = None, excerpt: str = "the excerpt"):
    trace = trace or _trace()
    return raw_candidates_to_entries([raw], trace, build_evidence(trace, excerpt), 3)


# -- the artifact gate ------------------------------------------------------------


def test_a_candidate_citing_an_untouched_file_is_rejected() -> None:
    """The gate the brief asks for: artifacts must be a subset of the trace's
    touched files, and a candidate that is not is dropped rather than stored."""

    results = _convert(_candidate(artifacts=["src/api/handlers.py"]))

    assert results == []


def test_one_invented_path_rejects_the_whole_candidate() -> None:
    """Not silently filtered down to the valid subset: if the model cited a
    file this session never touched, its reasoning is about a different
    session, and the remaining citation does not rescue it."""

    results = _convert(_candidate(artifacts=["tests/conftest.py", "does/not/exist.py"]))

    assert results == []


def test_a_candidate_selecting_a_real_subset_is_kept() -> None:
    ((entry, _body),) = _convert(_candidate(artifacts=["tests/conftest.py"]))

    assert entry.evidence.artifacts == ["tests/conftest.py"]


def test_no_selection_falls_back_to_every_touched_file() -> None:
    ((entry, _body),) = _convert(_candidate(artifacts=[]))

    assert entry.evidence.artifacts == ["tests/conftest.py", "alembic/env.py"]


# -- derived scope ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("artifacts", "expected"),
    [
        (["tests/conftest.py"], ["tests/**"]),
        (["alembic/versions/0042.py"], ["alembic/versions/**"]),
        (["tests/a.py", "tests/b.py"], ["tests/**"]),
        (["README.md"], ["README.md"]),
        (["src/api/h.py", "tests/c.py"], ["src/api/**", "tests/**"]),
    ],
)
def test_scope_is_derived_from_artifacts(artifacts: list[str], expected: list[str]) -> None:
    assert derive_scope(artifacts) == expected


def test_model_scope_that_does_not_follow_from_artifacts_is_dropped() -> None:
    ((entry, _body),) = _convert(
        _candidate(artifacts=["tests/conftest.py"], scope=["src/**", "everything/**"])
    )

    assert entry.scope == ["tests/**"]


def test_model_scope_may_refine_by_narrowing_to_a_derived_pattern() -> None:
    ((entry, _body),) = _convert(
        _candidate(artifacts=["tests/conftest.py", "alembic/env.py"], scope=["tests/**"])
    )

    assert entry.scope == ["tests/**"]


# -- the rest of evidence ---------------------------------------------------------


def test_excerpt_hash_is_of_the_excerpt_actually_sent() -> None:
    excerpt = "turn 3: UndefinedColumn users.last_seen_at"

    ((entry, _body),) = _convert(_candidate(), excerpt=excerpt)

    assert entry.evidence.excerpt_sha256 == hashlib.sha256(excerpt.encode("utf-8")).hexdigest()


def test_commit_comes_from_the_trace_not_the_model() -> None:
    ((entry, _body),) = _convert(_candidate(), trace=_trace(commit="a65592a"))

    assert entry.evidence.commit == "a65592a"


def test_session_id_and_capture_time_come_from_the_trace() -> None:
    ((entry, _body),) = _convert(_candidate())

    assert entry.evidence.session_id == "session-1"
    assert entry.evidence.harness == "claude-code"
    assert entry.evidence.captured_at == _ENDED_AT


def test_touched_files_are_deduplicated_in_trace_order() -> None:
    trace = _trace(
        diffs=[
            Diff(file="b.py", patch="@@"),
            Diff(file="a.py", patch="@@"),
            Diff(file="b.py", patch="@@ again"),
        ]
    )

    assert touched_files(trace) == ["b.py", "a.py"]


def test_the_model_cannot_set_its_own_id() -> None:
    ((entry, _body),) = _convert(_candidate(id="i-picked-this"))

    assert entry.id != "i-picked-this"
    assert entry.type is EntryType.GOTCHA
    assert entry.id.startswith("gotcha-")


# -- the capture hook records the commit ------------------------------------------


def test_the_hook_records_git_head_in_the_job_record(tmp_path: Path) -> None:
    """Captured at session end rather than at reflect time: by the time a job
    is swept, the working tree has usually moved on."""

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, timeout=60)
    (tmp_path / "f.txt").write_text("x", encoding="utf-8")
    for args in (
        ["git", "-C", str(tmp_path), "config", "user.email", "t@example.com"],
        ["git", "-C", str(tmp_path), "config", "user.name", "t"],
        ["git", "-C", str(tmp_path), "add", "f.txt"],
        ["git", "-C", str(tmp_path), "commit", "-qm", "init"],
    ):
        subprocess.run(args, check=True, timeout=60)
    head = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    ).stdout.strip()

    payload = json.dumps(
        {
            "session_id": "sess-commit",
            "transcript_path": str(tmp_path / "t.jsonl"),
            "cwd": str(tmp_path),
            "reason": "other",
        }
    )
    result = subprocess.run(
        [sys.executable, str(_ENQUEUE)], input=payload, capture_output=True, text=True, timeout=60
    )

    assert result.returncode == 0, result.stderr
    job = json.loads(
        (tmp_path / ".cairn" / "queue" / "sess-commit.json").read_text(encoding="utf-8")
    )
    assert job["commit"] == head


def test_the_hook_records_a_null_commit_outside_a_repository(tmp_path: Path) -> None:
    """A missing commit is a weaker entry, not a failed capture."""

    payload = json.dumps(
        {"session_id": "sess-nogit", "transcript_path": "t.jsonl", "cwd": str(tmp_path)}
    )
    result = subprocess.run(
        [sys.executable, str(_ENQUEUE)], input=payload, capture_output=True, text=True, timeout=60
    )

    assert result.returncode == 0, result.stderr
    job = json.loads(
        (tmp_path / ".cairn" / "queue" / "sess-nogit.json").read_text(encoding="utf-8")
    )
    assert job["commit"] is None

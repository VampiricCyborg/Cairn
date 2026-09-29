"""Subprocess tests for cairn/adapters/claude_code/enqueue.py.

Run through a real interpreter rather than by importing the module, because
the property under test is "this script can never fail a SessionEnd hook":
that is a property of the process's exit code, not of a function's return.

The script is stdlib-only and must not import `cairn`, so these also pin that
it works with the package entirely absent from the interpreter's path.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parent.parent / "cairn" / "adapters" / "claude_code" / "enqueue.py"
)


def _run(payload: str, *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Execute the hook exactly as Claude Code spawns it: interpreter path,
    script path, event JSON on stdin."""

    return subprocess.run(
        [sys.executable, str(_SCRIPT)],
        input=payload,
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd else None,
        timeout=30,
    )


def _payload(tmp_path: Path, **overrides: object) -> str:
    record: dict[str, object] = {
        "session_id": "0f3c1a9e-6b2d-4f77-9a10-2d4c8e51b3aa",
        "transcript_path": str(tmp_path / "transcript.jsonl"),
        "cwd": str(tmp_path),
        "hook_event_name": "SessionEnd",
        "reason": "other",
    }
    record.update(overrides)
    return json.dumps(record)


def _queue_files(tmp_path: Path) -> list[Path]:
    queue = tmp_path / ".cairn" / "queue"
    return sorted(queue.glob("*.json")) if queue.is_dir() else []


def test_writes_a_job_record_for_a_real_payload(tmp_path: Path) -> None:
    result = _run(_payload(tmp_path))

    assert result.returncode == 0, result.stderr
    (job,) = _queue_files(tmp_path)
    assert job.name == "0f3c1a9e-6b2d-4f77-9a10-2d4c8e51b3aa.json"
    record = json.loads(job.read_text(encoding="utf-8"))
    assert record["session_id"] == "0f3c1a9e-6b2d-4f77-9a10-2d4c8e51b3aa"
    assert record["transcript_path"] == str(tmp_path / "transcript.jsonl")
    assert record["harness"] == "claude-code"
    assert record["enqueued_at"].endswith("Z")


def test_needs_no_shell_and_no_third_party_imports(tmp_path: Path) -> None:
    """Runs with PATH emptied and the repo off PYTHONPATH: no bash, no jq, no
    `cairn` import. This is the whole point of the rewrite."""

    env = dict(os.environ)
    env["PATH"] = ""
    env["PYTHONPATH"] = ""
    env["PYTHONNOUSERSITE"] = "1"
    result = subprocess.run(
        [sys.executable, "-I", str(_SCRIPT)],
        input=_payload(tmp_path),
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert len(_queue_files(tmp_path)) == 1


@pytest.mark.parametrize(
    ("name", "payload"),
    [
        ("empty stdin", ""),
        ("not json", "this is not json"),
        ("json but not an object", "[1, 2, 3]"),
        ("missing session_id", '{"cwd": "."}'),
        ("missing cwd", '{"session_id": "abc"}'),
        ("null fields", '{"session_id": null, "cwd": null}'),
        ("session_id is a path", '{"session_id": "../../escape", "cwd": "."}'),
    ],
)
def test_exits_zero_and_writes_nothing_on_bad_input(
    tmp_path: Path, name: str, payload: str
) -> None:
    result = _run(payload, cwd=tmp_path)

    assert result.returncode == 0, f"{name}: {result.stderr}"
    assert _queue_files(tmp_path) == [], name


def test_a_path_like_session_id_cannot_escape_the_queue_directory(tmp_path: Path) -> None:
    outside = tmp_path / "outside.json"
    result = _run(_payload(tmp_path, session_id=f"../../{outside.stem}"))

    assert result.returncode == 0, result.stderr
    assert not outside.exists()
    assert _queue_files(tmp_path) == []


def test_exits_zero_when_the_queue_directory_cannot_be_created(tmp_path: Path) -> None:
    # A file where .cairn/ should be: makedirs cannot succeed.
    (tmp_path / ".cairn").write_text("not a directory", encoding="utf-8")

    result = _run(_payload(tmp_path))

    assert result.returncode == 0
    assert "skipping capture" in result.stderr


def test_rerunning_the_same_session_overwrites_rather_than_duplicating(tmp_path: Path) -> None:
    _run(_payload(tmp_path))
    _run(_payload(tmp_path))

    assert len(_queue_files(tmp_path)) == 1


def test_leaves_no_temporary_files_behind(tmp_path: Path) -> None:
    _run(_payload(tmp_path))

    queue = tmp_path / ".cairn" / "queue"
    assert [p.name for p in queue.iterdir()] == ["0f3c1a9e-6b2d-4f77-9a10-2d4c8e51b3aa.json"]


def test_completes_well_inside_the_session_end_budget(tmp_path: Path) -> None:
    """Claude Code's SessionEnd hooks share a 1.5 s default budget, and that
    -- not the README's per-hook figure -- is the constraint that can actually
    break a session. Asserted with wide headroom so a loaded CI runner does
    not fail the build; the measured p95 on Windows is ~190 ms, dominated by
    interpreter startup."""

    import time

    start = time.perf_counter()
    result = _run(_payload(tmp_path))
    elapsed = time.perf_counter() - start

    assert result.returncode == 0, result.stderr
    assert elapsed < 1.5, f"took {elapsed:.2f}s, at or over Claude Code's SessionEnd budget"

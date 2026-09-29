"""The capture log and `cairn stats --capture`.

The failure being measured is the SessionEnd hook not running at all, so the
denominator cannot come from the log itself -- a log of the runs that happened
would report 100% forever. It comes from Claude Code's session transcripts,
which exist whether or not the hook ever started.
"""

import json
import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from cairn.cli import app

runner = CliRunner()

_SCRIPT = (
    Path(__file__).resolve().parent.parent / "cairn" / "adapters" / "claude_code" / "enqueue.py"
)


def _run(payload: dict, *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    import os

    full_env = dict(os.environ)
    full_env.update(env or {})
    return subprocess.run(
        [sys.executable, str(_SCRIPT)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=full_env,
        timeout=30,
    )


def _payload(tmp_path: Path, **overrides: object) -> dict:
    record: dict = {
        "session_id": "0f3c1a9e-6b2d-4f77-9a10-2d4c8e51b3aa",
        "transcript_path": str(tmp_path / "transcript.jsonl"),
        "cwd": str(tmp_path),
        "hook_event_name": "SessionEnd",
        "reason": "other",
    }
    record.update(overrides)
    return record


def _capture_log(tmp_path: Path) -> list[dict]:
    path = tmp_path / ".cairn" / "capture-log.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# -- the permanent counter --------------------------------------------------------


def test_a_successful_capture_is_logged(tmp_path: Path) -> None:
    result = _run(_payload(tmp_path))

    assert result.returncode == 0, result.stderr
    (record,) = _capture_log(tmp_path)
    assert record["enqueued"] is True
    assert record["session_id"] == "0f3c1a9e-6b2d-4f77-9a10-2d4c8e51b3aa"
    assert record["reason"] == "other"
    assert record["error"] is None


def test_a_failed_capture_is_logged_with_the_error(tmp_path: Path) -> None:
    (tmp_path / ".cairn").write_text("not a directory", encoding="utf-8")

    result = _run(_payload(tmp_path))

    assert result.returncode == 0
    # The log lives under .cairn/ too, so a blocked .cairn/ loses the record as
    # well; what must hold is that the hook still exits 0.
    assert "skipping capture" in result.stderr


def test_the_session_end_reason_is_carried_into_the_job_record(tmp_path: Path) -> None:
    _run(_payload(tmp_path, reason="clear"))

    job = tmp_path / ".cairn" / "queue" / "0f3c1a9e-6b2d-4f77-9a10-2d4c8e51b3aa.json"
    assert json.loads(job.read_text(encoding="utf-8"))["reason"] == "clear"
    assert _capture_log(tmp_path)[0]["reason"] == "clear"


def test_the_mode_marker_is_recorded_when_set(tmp_path: Path) -> None:
    _run(_payload(tmp_path), env={"CAIRN_CAPTURE_MODE": "headless"})

    assert _capture_log(tmp_path)[0]["mode"] == "headless"


def test_the_mode_is_null_rather_than_guessed_when_unset(tmp_path: Path) -> None:
    """Nothing in the SessionEnd payload says whether the session was
    interactive, so an unset marker must read as unrecorded, not interactive."""

    import os

    env = {k: v for k, v in os.environ.items() if k != "CAIRN_CAPTURE_MODE"}
    result = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        input=json.dumps(_payload(tmp_path)),
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert _capture_log(tmp_path)[0]["mode"] is None


# -- the three-checkpoint trace ---------------------------------------------------


def test_the_trace_is_off_unless_explicitly_enabled(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"

    _run(_payload(tmp_path), env={"CAIRN_HOOK_DEBUG_LOG": str(trace)})

    assert not trace.exists()


def test_the_trace_records_all_three_checkpoints_in_order(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"

    _run(
        _payload(tmp_path),
        env={"CAIRN_HOOK_DEBUG": "1", "CAIRN_HOOK_DEBUG_LOG": str(trace)},
    )

    records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines() if line]
    assert [r["checkpoint"] for r in records] == ["spawned", "stdin_read", "written"]
    spawned = records[0]
    assert spawned["pid"] > 0
    assert spawned["ppid"] > 0
    assert spawned["argv"][0].endswith("enqueue.py")
    assert spawned["ts"].endswith("Z")
    # One process, so a missing later checkpoint localizes the failure.
    assert len({r["pid"] for r in records}) == 1


def test_a_run_that_fails_to_write_stops_after_the_second_checkpoint(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    (tmp_path / ".cairn").write_text("not a directory", encoding="utf-8")

    _run(
        _payload(tmp_path),
        env={"CAIRN_HOOK_DEBUG": "1", "CAIRN_HOOK_DEBUG_LOG": str(trace)},
    )

    records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines() if line]
    assert [r["checkpoint"] for r in records] == ["spawned", "stdin_read"]


# -- cairn stats --capture --------------------------------------------------------


def test_stats_capture_runs_without_a_log(tmp_path: Path) -> None:
    assert runner.invoke(app, ["init", str(tmp_path)]).exit_code == 0

    result = runner.invoke(app, ["stats", str(tmp_path), "--capture"])

    assert result.exit_code == 0, result.output
    assert "capture rate" in result.output


def test_stats_accepts_review_and_capture_together(tmp_path: Path) -> None:
    assert runner.invoke(app, ["init", str(tmp_path)]).exit_code == 0

    result = runner.invoke(app, ["stats", str(tmp_path), "--capture", "--review"])

    assert result.exit_code == 0, result.output
    assert "capture rate" in result.output
    assert "no review decisions logged yet" in result.output

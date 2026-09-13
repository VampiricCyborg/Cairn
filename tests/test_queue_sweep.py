"""End-to-end tests for the SessionStart queue sweep (`cairn context --hook`).

The sweep is what closes the loop the enqueue.sh hook starts: it normalizes
each queued job's transcript, extracts candidates with the configured
provider (mock by default), stages them via the Curator, and removes the
queue file -- all before the context block that same command renders.
"""

import json
from pathlib import Path

from typer.testing import CliRunner

from cairn.cli import app
from cairn.core.store import Store, load_entry

runner = CliRunner()


def _init(tmp_path: Path) -> Store:
    result = runner.invoke(app, ["init", str(tmp_path)])
    assert result.exit_code == 0, result.output
    return Store(tmp_path / ".cairn")


def _write_transcript_with_error(path: Path, session_id: str) -> None:
    records = [
        {
            "type": "user",
            "sessionId": session_id,
            "timestamp": "2026-01-01T10:00:00Z",
            "message": {"role": "user", "content": "Run the migration."},
        },
        {
            "type": "assistant",
            "sessionId": session_id,
            "timestamp": "2026-01-01T10:01:00Z",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Running the test suite."},
                    {
                        "type": "tool_use",
                        "id": "tool_1",
                        "name": "Bash",
                        "input": {"command": "pytest -q"},
                    },
                ],
            },
        },
        {
            "type": "user",
            "sessionId": session_id,
            "timestamp": "2026-01-01T10:02:00Z",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_1",
                        "content": "UndefinedColumn: users.last_seen_at does not exist",
                        "is_error": True,
                    }
                ],
            },
        },
        {
            "type": "assistant",
            "sessionId": session_id,
            "timestamp": "2026-01-01T10:05:00Z",
            "message": {"role": "assistant", "content": "Fixed; tests pass now."},
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def _write_job(store: Store, session_id: str, transcript_path: Path) -> Path:
    store.queue_dir.mkdir(parents=True, exist_ok=True)
    job_path = store.queue_dir / f"{session_id}.json"
    job_path.write_text(
        json.dumps(
            {
                "session_id": session_id,
                "transcript_path": str(transcript_path),
                "harness": "claude-code",
                "enqueued_at": "2026-01-01T10:05:00Z",
            }
        ),
        encoding="utf-8",
    )
    return job_path


def test_hook_with_empty_queue_still_renders_valid_json(tmp_path: Path) -> None:
    _init(tmp_path)

    result = runner.invoke(app, ["context", str(tmp_path), "--hook"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    assert payload["hookSpecificOutput"]["additionalContext"] == ""


def test_hook_sweeps_a_valid_queued_job_end_to_end(tmp_path: Path) -> None:
    store = _init(tmp_path)
    transcript = tmp_path / "session.jsonl"
    _write_transcript_with_error(transcript, "sess-1")
    job_path = _write_job(store, "sess-1", transcript)

    result = runner.invoke(app, ["context", str(tmp_path), "--hook"])

    assert result.exit_code == 0, result.output
    assert not job_path.exists()
    staged_files = list(store.staging_dir.glob("*.md"))
    assert len(staged_files) == 1
    entry = load_entry(staged_files[0])
    assert entry.evidence.session_id == "sess-1"
    assert entry.evidence.harness == "claude-code"

    # The hook output is still well-formed JSON.
    payload = json.loads(result.output)
    assert payload["hookSpecificOutput"]["hookEventName"] == "SessionStart"


def test_malformed_job_does_not_block_a_valid_job(tmp_path: Path) -> None:
    store = _init(tmp_path)

    # Malformed: no transcript_path at all.
    store.queue_dir.mkdir(parents=True, exist_ok=True)
    malformed_path = store.queue_dir / "bad-session.json"
    malformed_path.write_text(json.dumps({"session_id": "bad-session"}), encoding="utf-8")

    # Also malformed: not even valid JSON.
    garbage_path = store.queue_dir / "garbage-session.json"
    garbage_path.write_text("not json at all", encoding="utf-8")

    # A genuinely valid job alongside the two broken ones.
    transcript = tmp_path / "session.jsonl"
    _write_transcript_with_error(transcript, "good-session")
    good_path = _write_job(store, "good-session", transcript)

    result = runner.invoke(app, ["context", str(tmp_path), "--hook"])

    assert result.exit_code == 0, result.output
    # All three queue files are drained (malformed jobs are not retried forever).
    assert not malformed_path.exists()
    assert not garbage_path.exists()
    assert not good_path.exists()

    staged_files = list(store.staging_dir.glob("*.md"))
    assert len(staged_files) == 1
    entry = load_entry(staged_files[0])
    assert entry.evidence.session_id == "good-session"


def test_job_pointing_at_missing_transcript_is_skipped_not_fatal(tmp_path: Path) -> None:
    store = _init(tmp_path)
    job_path = _write_job(store, "missing-transcript", tmp_path / "does-not-exist.jsonl")

    result = runner.invoke(app, ["context", str(tmp_path), "--hook"])

    assert result.exit_code == 0, result.output
    assert not job_path.exists()
    assert list(store.staging_dir.glob("*.md")) == []


def test_plain_context_without_hook_does_not_sweep_queue(tmp_path: Path) -> None:
    """Only `--hook` triggers the sweep; a plain `cairn context` (e.g. a
    manual preview) must not have the side effect of draining the queue."""

    store = _init(tmp_path)
    transcript = tmp_path / "session.jsonl"
    _write_transcript_with_error(transcript, "sess-1")
    job_path = _write_job(store, "sess-1", transcript)

    result = runner.invoke(app, ["context", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert job_path.exists()
    assert list(store.staging_dir.glob("*.md")) == []

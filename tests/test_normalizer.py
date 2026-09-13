"""Tests for cairn.core.normalizer."""

import json
from pathlib import Path

from cairn.core.models import TurnRole
from cairn.core.normalizer import normalize_claude_code_transcript


def _write_transcript(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def test_normalizes_turns_tool_calls_and_results(tmp_path: Path) -> None:
    transcript = tmp_path / "session.jsonl"
    _write_transcript(
        transcript,
        [
            {
                "type": "user",
                "sessionId": "sess-1",
                "timestamp": "2026-01-01T10:00:00Z",
                "message": {"role": "user", "content": "Add a migration for last_seen_at."},
            },
            {
                "type": "assistant",
                "sessionId": "sess-1",
                "timestamp": "2026-01-01T10:01:00Z",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "I'll run the test suite."},
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
                "sessionId": "sess-1",
                "timestamp": "2026-01-01T10:02:00Z",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool_1",
                            "content": "UndefinedColumn: users.last_seen_at",
                            "is_error": True,
                        }
                    ],
                },
            },
            {
                "type": "assistant",
                "sessionId": "sess-1",
                "timestamp": "2026-01-01T10:05:00Z",
                "message": {"role": "assistant", "content": "Fixed it; tests pass now."},
            },
        ],
    )

    trace = normalize_claude_code_transcript(transcript)

    assert trace.session_id == "sess-1"
    assert trace.harness == "claude-code"
    assert trace.started_at.isoformat() == "2026-01-01T10:00:00+00:00"
    assert trace.ended_at.isoformat() == "2026-01-01T10:05:00+00:00"
    assert len(trace.turns) == 3  # tool_result merges into the preceding assistant turn

    user_turn, assistant_turn, closing_turn = trace.turns
    assert user_turn.role is TurnRole.USER
    assert user_turn.content == "Add a migration for last_seen_at."

    assert assistant_turn.role is TurnRole.ASSISTANT
    assert assistant_turn.content == "I'll run the test suite."
    assert len(assistant_turn.tool_calls) == 1
    assert assistant_turn.tool_calls[0].name == "Bash"
    assert assistant_turn.tool_calls[0].input == {"command": "pytest -q"}
    assert len(assistant_turn.tool_results) == 1
    assert assistant_turn.tool_results[0].name == "Bash"
    assert assistant_turn.tool_results[0].is_error is True
    assert assistant_turn.tool_results[0].output == "UndefinedColumn: users.last_seen_at"

    assert closing_turn.role is TurnRole.ASSISTANT
    assert closing_turn.content == "Fixed it; tests pass now."

    # The last turn carrying a tool result had an error -> outcome reflects it.
    assert trace.outcome == "error"


def test_extracts_diff_from_edit_tool_call(tmp_path: Path) -> None:
    transcript = tmp_path / "session.jsonl"
    _write_transcript(
        transcript,
        [
            {
                "type": "assistant",
                "sessionId": "sess-2",
                "timestamp": "2026-01-01T10:00:00Z",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tool_1",
                            "name": "Edit",
                            "input": {
                                "file_path": "tests/conftest.py",
                                "old_string": "Base.metadata.create_all(engine)",
                                "new_string": "command.upgrade(alembic_cfg, 'head')",
                            },
                        }
                    ],
                },
            },
        ],
    )

    trace = normalize_claude_code_transcript(transcript)

    assert len(trace.diffs) == 1
    assert trace.diffs[0].file == "tests/conftest.py"
    assert "Base.metadata.create_all(engine)" in trace.diffs[0].patch
    assert "command.upgrade" in trace.diffs[0].patch


def test_skips_malformed_lines_without_failing(tmp_path: Path) -> None:
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        "not json at all\n"
        + json.dumps(
            {
                "type": "user",
                "sessionId": "sess-3",
                "timestamp": "2026-01-01T10:00:00Z",
                "message": {"role": "user", "content": "hello"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    trace = normalize_claude_code_transcript(transcript)

    assert trace.session_id == "sess-3"
    assert len(trace.turns) == 1
    assert trace.turns[0].content == "hello"


def test_no_tool_results_defaults_to_completed_outcome(tmp_path: Path) -> None:
    transcript = tmp_path / "session.jsonl"
    _write_transcript(
        transcript,
        [
            {
                "type": "user",
                "sessionId": "sess-4",
                "timestamp": "2026-01-01T10:00:00Z",
                "message": {"role": "user", "content": "hi"},
            },
        ],
    )

    trace = normalize_claude_code_transcript(transcript)

    assert trace.outcome == "completed"


def test_session_id_falls_back_to_filename_stem(tmp_path: Path) -> None:
    transcript = tmp_path / "my-session-id.jsonl"
    _write_transcript(
        transcript,
        [{"type": "user", "message": {"role": "user", "content": "hi"}}],
    )

    trace = normalize_claude_code_transcript(transcript)

    assert trace.session_id == "my-session-id"

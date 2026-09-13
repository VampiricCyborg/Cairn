"""Tests for cairn.core.normalizer."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from cairn.core.models import TurnRole
from cairn.core.normalizer import (
    normalize,
    normalize_claude_code_transcript,
    normalize_opencode_transcript,
)


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


# -- opencode -----------------------------------------------------------------------
#
# Fixtures below use the real `{info: Message, parts: Part[]}` shape
# confirmed against opencode's published `@opencode-ai/sdk` types (see
# cairn/core/normalizer.py's module docstring) -- the verbatim response of
# `client.session.messages`, which is what `cairn/adapters/opencode/plugin.ts`
# writes to the transcript side-car file.


def _write_opencode_transcript(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(records), encoding="utf-8")


def test_opencode_normalizes_turns_tool_calls_and_results(tmp_path: Path) -> None:
    transcript = tmp_path / "sess-oc-1.transcript.json"
    _write_opencode_transcript(
        transcript,
        [
            {
                "info": {
                    "id": "msg-1",
                    "sessionID": "sess-oc-1",
                    "role": "user",
                    "time": {"created": 1700000000000},
                },
                "parts": [{"type": "text", "text": "Add a migration for last_seen_at."}],
            },
            {
                "info": {
                    "id": "msg-2",
                    "sessionID": "sess-oc-1",
                    "role": "assistant",
                    "time": {"created": 1700000060000, "completed": 1700000090000},
                    "parentID": "msg-1",
                    "modelID": "some-model",
                    "providerID": "some-provider",
                    "mode": "build",
                    "path": {"cwd": "/repo", "root": "/repo"},
                    "cost": 0.0,
                    "tokens": {
                        "input": 10,
                        "output": 5,
                        "reasoning": 0,
                        "cache": {"read": 0, "write": 0},
                    },
                },
                "parts": [
                    {"type": "text", "text": "I'll run the test suite."},
                    {
                        "type": "tool",
                        "id": "part-1",
                        "sessionID": "sess-oc-1",
                        "messageID": "msg-2",
                        "callID": "call-1",
                        "tool": "bash",
                        "state": {
                            "status": "error",
                            "input": {"command": "pytest -q"},
                            "error": "UndefinedColumn: users.last_seen_at",
                            "time": {"start": 1, "end": 2},
                        },
                    },
                ],
            },
        ],
    )

    trace = normalize_opencode_transcript(transcript)

    assert trace.session_id == "sess-oc-1"
    assert trace.harness == "opencode"
    assert len(trace.turns) == 2

    user_turn, assistant_turn = trace.turns
    assert user_turn.role is TurnRole.USER
    assert user_turn.content == "Add a migration for last_seen_at."

    assert assistant_turn.role is TurnRole.ASSISTANT
    assert assistant_turn.content == "I'll run the test suite."
    assert len(assistant_turn.tool_calls) == 1
    assert assistant_turn.tool_calls[0].name == "bash"
    assert assistant_turn.tool_calls[0].input == {"command": "pytest -q"}
    assert len(assistant_turn.tool_results) == 1
    assert assistant_turn.tool_results[0].is_error is True
    assert assistant_turn.tool_results[0].output == "UndefinedColumn: users.last_seen_at"

    # The only tool result present is an error -> outcome reflects it.
    assert trace.outcome == "error"

    assert trace.started_at == datetime.fromtimestamp(1700000000000 / 1000, tz=UTC)
    assert trace.ended_at == datetime.fromtimestamp(1700000060000 / 1000, tz=UTC)


def test_opencode_completed_tool_result_gives_completed_outcome(tmp_path: Path) -> None:
    transcript = tmp_path / "sess-oc-2.transcript.json"
    _write_opencode_transcript(
        transcript,
        [
            {
                "info": {
                    "id": "msg-1",
                    "sessionID": "sess-oc-2",
                    "role": "assistant",
                    "time": {"created": 1700000000000},
                },
                "parts": [
                    {
                        "type": "tool",
                        "tool": "edit",
                        "state": {
                            "status": "completed",
                            "input": {"filePath": "a.py"},
                            "output": "ok",
                            "title": "edit",
                            "metadata": {},
                            "time": {"start": 1, "end": 2},
                        },
                    }
                ],
            }
        ],
    )

    trace = normalize_opencode_transcript(transcript)

    assert trace.outcome == "completed"
    assert trace.turns[0].tool_results[0].is_error is False
    assert trace.turns[0].tool_results[0].output == "ok"


def test_opencode_never_fabricates_diffs_from_patch_parts(tmp_path: Path) -> None:
    """A `patch` part only names files + a hash, never diff text -- the
    normalizer must not invent a `Diff` from it."""

    transcript = tmp_path / "sess-oc-3.transcript.json"
    _write_opencode_transcript(
        transcript,
        [
            {
                "info": {
                    "id": "msg-1",
                    "sessionID": "sess-oc-3",
                    "role": "assistant",
                    "time": {"created": 1700000000000},
                },
                "parts": [{"type": "patch", "hash": "abc123", "files": ["a.py", "b.py"]}],
            }
        ],
    )

    trace = normalize_opencode_transcript(transcript)

    assert trace.diffs == []


def test_opencode_preserves_assistant_error_verbatim(tmp_path: Path) -> None:
    transcript = tmp_path / "sess-oc-4.transcript.json"
    _write_opencode_transcript(
        transcript,
        [
            {
                "info": {
                    "id": "msg-1",
                    "sessionID": "sess-oc-4",
                    "role": "assistant",
                    "time": {"created": 1700000000000},
                    "error": {"name": "UnknownError", "data": {"message": "boom"}},
                },
                "parts": [],
            }
        ],
    )

    trace = normalize_opencode_transcript(transcript)

    assert len(trace.errors) == 1
    assert "boom" in trace.errors[0]


def test_opencode_skips_malformed_records_without_failing(tmp_path: Path) -> None:
    transcript = tmp_path / "sess-oc-5.transcript.json"
    _write_opencode_transcript(
        transcript,
        [
            "not a record",
            {"info": {"role": "user"}},  # missing "parts"
            {
                "info": {"id": "m1", "sessionID": "sess-oc-5", "role": "user", "time": {}},
                "parts": [{"type": "text", "text": "hello"}],
            },
        ],
    )

    trace = normalize_opencode_transcript(transcript)

    assert trace.session_id == "sess-oc-5"
    assert len(trace.turns) == 1
    assert trace.turns[0].content == "hello"


def test_opencode_session_id_falls_back_to_filename_stem(tmp_path: Path) -> None:
    transcript = tmp_path / "my-session-id.transcript.json"
    _write_opencode_transcript(transcript, [])

    trace = normalize_opencode_transcript(transcript)

    assert trace.session_id == "my-session-id"


def test_opencode_rejects_non_array_content(tmp_path: Path) -> None:
    transcript = tmp_path / "bad.transcript.json"
    transcript.write_text(json.dumps({"not": "an array"}), encoding="utf-8")

    with pytest.raises(ValueError, match="expected a JSON array"):
        normalize_opencode_transcript(transcript)


# -- dispatch -----------------------------------------------------------------------


def test_normalize_dispatches_to_claude_code(tmp_path: Path) -> None:
    transcript = tmp_path / "session.jsonl"
    _write_transcript(
        transcript,
        [{"type": "user", "sessionId": "sess-1", "message": {"role": "user", "content": "hi"}}],
    )

    trace = normalize(transcript, harness="claude-code")

    assert trace.harness == "claude-code"
    assert trace.session_id == "sess-1"


def test_normalize_dispatches_to_opencode(tmp_path: Path) -> None:
    transcript = tmp_path / "sess-1.transcript.json"
    _write_opencode_transcript(
        transcript,
        [
            {
                "info": {
                    "id": "m1",
                    "sessionID": "sess-1",
                    "role": "user",
                    "time": {"created": 1700000000000},
                },
                "parts": [{"type": "text", "text": "hi"}],
            }
        ],
    )

    trace = normalize(transcript, harness="opencode")

    assert trace.harness == "opencode"
    assert trace.session_id == "sess-1"


def test_normalize_raises_for_unknown_harness(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no normalizer registered"):
        normalize(tmp_path / "whatever.json", harness="codex")

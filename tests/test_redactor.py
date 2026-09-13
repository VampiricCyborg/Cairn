"""Tests for cairn.core.redactor.

These are written around failure modes, not just the happy path: a denied
diff must be *absent*, not merely blanked; a named-pattern match must leave
its surrounding sentence untouched; an unknown secret shape must still be
caught by the entropy check; and that same check must not fire on ordinary,
if unusually long, English text. Calibration for the entropy examples below
was checked numerically (Shannon entropy per character): the coined
36-letter word here scores ~3.66 bits/char, well under the 4.0 default
threshold, while a random 40-character base64-alphabet string scores
~4.5-4.9, comfortably above it.
"""

from datetime import UTC, datetime

from cairn.core.models import Diff, SessionTrace, ToolResult, Turn, TurnRole
from cairn.core.redactor import Redactor

DENY_GLOBS = [".env*", "**/secrets/**", "**/*.pem", "**/*.key"]
PATTERNS = [
    r"sk-[A-Za-z0-9]{20,}",
    r"ghp_[A-Za-z0-9]{36}",
    r"AKIA[0-9A-Z]{16}",
]

# A real (if famously long) English word: 34 letters, ~3.66 bits/char of
# unigram entropy thanks to heavy letter repetition — realistic stand-in for
# a long compound identifier that must NOT be flagged as a secret.
LONG_ENGLISH_WORD = "supercalifragilisticexpialidocious"

# A fixed, non-random 40-char base64-alphabet token matching none of
# PATTERNS but with high per-character entropy (mixed case + digits).
HIGH_ENTROPY_UNKNOWN_SECRET = "Q7xM2pLwZ9vK3rN8sT1yB6hD4jF0cA5uE7gR2kX9"


def _make_trace(**overrides: object) -> SessionTrace:
    defaults: dict[str, object] = dict(
        session_id="session-1",
        harness="claude-code",
        started_at=datetime(2026, 9, 12, 11, 0, 0, tzinfo=UTC),
        ended_at=datetime(2026, 9, 12, 11, 30, 0, tzinfo=UTC),
        turns=[],
        errors=[],
        diffs=[],
        outcome="success",
    )
    defaults.update(overrides)
    return SessionTrace(**defaults)  # type: ignore[arg-type]


def _redactor(entropy_threshold: float = 4.0) -> Redactor:
    return Redactor(DENY_GLOBS, PATTERNS, entropy_threshold=entropy_threshold)


def test_deny_glob_removes_diff_entirely_not_just_redacted() -> None:
    trace = _make_trace(
        diffs=[
            Diff(file=".env.production", patch="DATABASE_PASSWORD=hunter2"),
            Diff(file="app.py", patch="print('hello')"),
        ]
    )

    redacted = _redactor().redact_trace(trace)

    assert [diff.file for diff in redacted.diffs] == ["app.py"]
    # The denied content must not survive anywhere, even redacted.
    assert "hunter2" not in "".join(diff.patch for diff in redacted.diffs)


def test_deny_glob_matches_nested_path() -> None:
    trace = _make_trace(
        diffs=[
            Diff(file="config/secrets/db.yaml", patch="password: hunter2"),
            Diff(file="src/main.py", patch="x = 1"),
        ]
    )

    redacted = _redactor().redact_trace(trace)

    assert [diff.file for diff in redacted.diffs] == ["src/main.py"]


def test_named_pattern_redacted_with_surrounding_text_preserved() -> None:
    secret = "sk-abcdefghijklmnopqrstuvwxyz0123456789"
    trace = _make_trace(
        turns=[
            Turn(
                role=TurnRole.ASSISTANT,
                content=f"Sure, use this key {secret} to authenticate the client.",
            )
        ]
    )

    redacted = _redactor().redact_trace(trace)
    content = redacted.turns[0].content

    assert secret not in content
    assert content.startswith("Sure, use this key [REDACTED:sk]")
    assert content.endswith("to authenticate the client.")


def test_tool_result_output_is_stringified_and_redacted() -> None:
    trace = _make_trace(
        turns=[
            Turn(
                role=TurnRole.TOOL,
                content="",
                tool_results=[
                    ToolResult(name="run_shell", output="AKIA1234567890ABCDEF found in output")
                ],
            )
        ]
    )

    redacted = _redactor().redact_trace(trace)

    assert redacted.turns[0].tool_results[0].output == "[REDACTED:AKIA] found in output"


def test_errors_are_redacted() -> None:
    trace = _make_trace(errors=[f"auth failed for ghp_{'a' * 36}"])

    redacted = _redactor().redact_trace(trace)

    assert redacted.errors == ["auth failed for [REDACTED:ghp]"]


def test_entropy_check_catches_unknown_secret_format() -> None:
    trace = _make_trace(
        turns=[
            Turn(
                role=TurnRole.TOOL,
                content=f"leaked token: {HIGH_ENTROPY_UNKNOWN_SECRET} in logs",
            )
        ]
    )

    redacted = _redactor().redact_trace(trace)
    content = redacted.turns[0].content

    assert HIGH_ENTROPY_UNKNOWN_SECRET not in content
    assert "[REDACTED:high-entropy]" in content
    assert content == "leaked token: [REDACTED:high-entropy] in logs"


def test_entropy_check_does_not_flag_normal_english_text() -> None:
    sentence = (
        f"The word {LONG_ENGLISH_WORD} is long but it is not a secret, "
        "just an unusually large English word."
    )
    trace = _make_trace(turns=[Turn(role=TurnRole.USER, content=sentence)])

    redacted = _redactor().redact_trace(trace)

    assert redacted.turns[0].content == sentence
    assert "REDACTED" not in redacted.turns[0].content


def test_round_trip_no_secrets_is_content_identical_but_new_object() -> None:
    trace = _make_trace(
        turns=[Turn(role=TurnRole.USER, content="Please fix the failing test in test_foo.py.")],
        diffs=[Diff(file="test_foo.py", patch="-assert x == 1\n+assert x == 2\n")],
        errors=["AssertionError: 1 != 2"],
    )

    redacted = _redactor().redact_trace(trace)

    assert redacted == trace
    assert redacted is not trace
    assert redacted.turns is not trace.turns
    assert redacted.diffs is not trace.diffs


def test_redact_trace_does_not_mutate_input() -> None:
    secret = "sk-" + "a" * 25
    original_content = f"key is {secret}"
    trace = _make_trace(turns=[Turn(role=TurnRole.USER, content=original_content)])

    _redactor().redact_trace(trace)

    assert trace.turns[0].content == original_content

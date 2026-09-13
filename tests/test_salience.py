"""Tests for cairn.core.salience."""

from datetime import UTC, datetime

from cairn.core.models import SessionTrace, ToolResult, Turn, TurnRole
from cairn.core.salience import find_salient_spans, is_reflectable, render_salient_excerpt

MIN_TURNS = 4


def _turn(content: str = "turn", *, tool_results: list[ToolResult] | None = None) -> Turn:
    return Turn(role=TurnRole.ASSISTANT, content=content, tool_results=tool_results or [])


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


def test_below_min_turns_is_never_reflectable_even_with_errors() -> None:
    trace = _make_trace(
        turns=[
            _turn("investigating"),
            _turn("found the bug", tool_results=[ToolResult(name="run", is_error=True)]),
        ],
        errors=["boom"],
    )

    assert not is_reflectable(trace, min_turns=MIN_TURNS)


def test_uneventful_session_at_min_turns_is_not_reflectable() -> None:
    trace = _make_trace(
        turns=[_turn(f"routine turn {i}") for i in range(MIN_TURNS)],
    )

    assert not is_reflectable(trace, min_turns=MIN_TURNS)


def test_isolated_error_makes_session_reflectable() -> None:
    trace = _make_trace(
        turns=[
            _turn("turn 0"),
            _turn("turn 1, running the tests"),
            _turn("turn 2, it failed", tool_results=[ToolResult(name="pytest", is_error=True)]),
            _turn("turn 3, let's look at why"),
        ],
    )

    assert is_reflectable(trace, min_turns=MIN_TURNS)


def test_isolated_error_span_covers_error_turn_plus_one_each_side() -> None:
    trace = _make_trace(
        turns=[
            _turn("turn 0"),
            _turn("turn 1, running the tests"),
            _turn(
                "turn 2, it failed",
                tool_results=[ToolResult(name="pytest", output="AssertionError", is_error=True)],
            ),
            _turn("turn 3, let's look at why"),
        ],
    )

    spans = find_salient_spans(trace)

    assert len(spans) == 1
    assert spans[0].reason == "error"
    assert spans[0].turn_indices == [1, 2, 3]
    assert spans[0].signal_text == "AssertionError"


def test_two_close_errors_merge_into_one_span_not_two() -> None:
    trace = _make_trace(
        turns=[
            _turn("turn 0"),
            _turn("turn 1"),
            _turn("turn 2, first failure", tool_results=[ToolResult(name="a", is_error=True)]),
            _turn("turn 3, retry"),
            _turn("turn 4, second failure", tool_results=[ToolResult(name="b", is_error=True)]),
            _turn("turn 5"),
        ],
    )

    spans = find_salient_spans(trace)

    error_spans = [span for span in spans if span.reason == "error"]
    assert len(error_spans) == 1
    assert error_spans[0].turn_indices == [1, 2, 3, 4, 5]


def test_far_apart_errors_stay_as_separate_spans() -> None:
    trace = _make_trace(
        turns=[
            _turn("turn 0", tool_results=[ToolResult(name="a", is_error=True)]),
            _turn("turn 1"),
            _turn("turn 2"),
            _turn("turn 3"),
            _turn("turn 4"),
            _turn("turn 5", tool_results=[ToolResult(name="b", is_error=True)]),
        ],
    )

    spans = find_salient_spans(trace)

    error_spans = [span for span in spans if span.reason == "error"]
    assert len(error_spans) == 2
    assert error_spans[0].turn_indices == [0, 1]
    assert error_spans[1].turn_indices == [4, 5]


def test_retry_phrase_flagged_with_own_reason() -> None:
    trace = _make_trace(
        turns=[
            _turn("turn 0"),
            _turn("that didn't work, let me try again"),
            _turn("turn 2"),
            _turn("turn 3"),
        ],
    )

    assert is_reflectable(trace, min_turns=MIN_TURNS)
    spans = find_salient_spans(trace)
    assert len(spans) == 1
    assert spans[0].reason == "retry"
    assert spans[0].turn_indices == [1]


def test_reversal_phrase_flagged_when_no_retry_phrase_present() -> None:
    trace = _make_trace(
        turns=[
            _turn("turn 0"),
            _turn("turn 1"),
            _turn("reverting that last change"),
            _turn("turn 3"),
        ],
    )

    spans = find_salient_spans(trace)
    assert len(spans) == 1
    assert spans[0].reason == "reversal"
    assert spans[0].turn_indices == [2]


def test_fail_then_pass_test_transition_spans_all_turns_between() -> None:
    trace = _make_trace(
        turns=[
            _turn("running tests"),
            _turn("FAILED tests/test_foo.py::test_bar"),
            _turn("let me look at the fixture"),
            _turn("PASSED tests/test_foo.py::test_bar"),
        ],
    )

    spans = find_salient_spans(trace)

    assert len(spans) == 1
    assert spans[0].reason == "test_transition"
    assert spans[0].turn_indices == [1, 2, 3]
    assert spans[0].signal_text == "tests/test_foo.py::test_bar"


def test_test_transition_beyond_cap_is_dropped() -> None:
    turns = [_turn("FAILED tests/test_foo.py::test_bar")]
    turns.extend(_turn(f"filler {i}") for i in range(10))
    turns.append(_turn("PASSED tests/test_foo.py::test_bar"))
    trace = _make_trace(turns=turns)

    spans = find_salient_spans(trace, max_test_transition_span=6)

    assert not any(span.reason == "test_transition" for span in spans)


def test_render_salient_excerpt_is_shorter_than_full_transcript_and_marks_gaps() -> None:
    turns = [_turn(f"routine filler turn number {i}") for i in range(20)]
    turns[3] = _turn(
        "boom, it broke", tool_results=[ToolResult(name="run", output="oops", is_error=True)]
    )
    turns[15] = _turn("reverting that change")
    trace = _make_trace(turns=turns)

    spans = find_salient_spans(trace)
    excerpt = render_salient_excerpt(trace, spans)
    full_transcript = "\n".join(f"[{i}] {t.role.value}: {t.content}" for i, t in enumerate(turns))

    assert len(excerpt) < len(full_transcript)
    assert "turns omitted" in excerpt
    assert "boom, it broke" in excerpt
    assert "reverting that change" in excerpt
    assert "routine filler turn number 10" not in excerpt

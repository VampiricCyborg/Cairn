"""Salience filter: surface errors, retries, reversals, test transitions.

Per the README: "Sessions are mostly uneventful; the lesson density is
concentrated around errors, retries, reversals, and test-status
transitions." This module decides whether a (redacted) `SessionTrace` is
worth reflecting on at all, and if so, which turns actually carry that
lesson density — so the reflector prompt in P2 never has to see (and pay
for) the raw full transcript.
"""

import re
from dataclasses import dataclass
from typing import Literal

from cairn.core.models import SessionTrace, Turn

#: Substring signals (case-insensitive, not fuzzy) that an agent is retrying
#: an approach that just failed. Deliberately short and literal: a longer or
#: fuzzy-matched list would start flagging ordinary conversation.
RETRY_PHRASES = [
    "let me try again",
    "let's try again",
    "trying again",
    "i'll retry",
    "let me retry",
]

#: Substring signals that an agent is abandoning or undoing a prior action.
REVERSAL_PHRASES = [
    "that didn't work",
    "that did not work",
    "reverting",
    "let me revert",
    "rolling back",
    "undo that",
    "let me undo",
]

DEFAULT_MAX_TEST_TRANSITION_SPAN = 6

#: A failing/passing test result, tolerant of pytest-style `FAILED <id>` /
#: `PASSED <id>` lines wherever they appear (turn content or tool output).
#: `<id>` (a test node id or file path) is what pairs a failure to its fix.
_TEST_FAIL_RE = re.compile(r"(?i)\bFAILED\b[:\s]+(\S+)")
_TEST_PASS_RE = re.compile(r"(?i)\bPASSED\b[:\s]+(\S+)")

Reason = Literal["error", "retry", "reversal", "test_transition"]


@dataclass(frozen=True)
class SalientSpan:
    """A contiguous run of turn indices worth showing the reflector, and why."""

    turn_indices: list[int]
    reason: Reason
    signal_text: str


def _turn_text(turn: Turn) -> str:
    """`turn.content` plus every tool result's stringified output, for
    signals (like test pass/fail output) that live in tool output rather
    than the turn's own text."""

    parts = [turn.content]
    parts.extend(str(result.output) for result in turn.tool_results if result.output is not None)
    return "\n".join(parts)


def _find_phrase(text: str, phrases: list[str]) -> str | None:
    """The first phrase from `phrases` found in `text` (case-insensitive
    substring match), returned as it actually appears in `text`."""

    lowered = text.lower()
    for phrase in phrases:
        index = lowered.find(phrase)
        if index != -1:
            return text[index : index + len(phrase)]
    return None


def is_reflectable(trace: SessionTrace, min_turns: int) -> bool:
    """Whether `trace` is worth running through the reflector at all.

    False below `min_turns` regardless of content (mirrors
    `reflect.min_session_turns`). Otherwise false unless the trace shows at
    least one concrete signal: a top-level error, a failed tool result, or
    a turn whose content matches a retry/reversal phrase. An uneventful,
    error-free session has nothing to teach and should not reach the
    reflector.
    """

    if len(trace.turns) < min_turns:
        return False

    if trace.errors:
        return True

    for turn in trace.turns:
        if any(result.is_error for result in turn.tool_results):
            return True
        if _find_phrase(turn.content, RETRY_PHRASES) is not None:
            return True
        if _find_phrase(turn.content, REVERSAL_PHRASES) is not None:
            return True

    return False


def _error_spans(trace: SessionTrace) -> list[SalientSpan]:
    """One span per turn with a failed tool result, widened by one turn on
    each side so the reflector sees the attempted action and the reaction
    to it, not an isolated error line."""

    spans: list[SalientSpan] = []
    last_index = len(trace.turns) - 1
    for index, turn in enumerate(trace.turns):
        errored = next((result for result in turn.tool_results if result.is_error), None)
        if errored is None:
            continue
        indices = [i for i in (index - 1, index, index + 1) if 0 <= i <= last_index]
        signal_text = str(errored.output) if errored.output is not None else errored.name
        spans.append(SalientSpan(turn_indices=indices, reason="error", signal_text=signal_text))
    return spans


def _retry_and_reversal_spans(trace: SessionTrace) -> list[SalientSpan]:
    """One single-turn span per turn whose content matches a retry or
    reversal phrase (retry checked first, so a turn matching both is
    reported once, as a retry)."""

    spans: list[SalientSpan] = []
    for index, turn in enumerate(trace.turns):
        retry_match = _find_phrase(turn.content, RETRY_PHRASES)
        if retry_match is not None:
            spans.append(SalientSpan(turn_indices=[index], reason="retry", signal_text=retry_match))
            continue
        reversal_match = _find_phrase(turn.content, REVERSAL_PHRASES)
        if reversal_match is not None:
            spans.append(
                SalientSpan(turn_indices=[index], reason="reversal", signal_text=reversal_match)
            )
    return spans


def _test_transition_spans(
    trace: SessionTrace, max_span_length: int
) -> list[SalientSpan]:
    """A span from a failing-test turn to the later turn where that same
    test (or file) is reported passing, capped at `max_span_length` turns
    so one stale failure can't drag half the session into the excerpt. A
    failure with no matching pass within the cap is simply dropped rather
    than paired with an unrelated, much later pass.
    """

    pending_failures: dict[str, int] = {}
    spans: list[SalientSpan] = []

    for index, turn in enumerate(trace.turns):
        text = _turn_text(turn)

        for match in _TEST_FAIL_RE.finditer(text):
            identifier = match.group(1)
            pending_failures.setdefault(identifier, index)

        for match in _TEST_PASS_RE.finditer(text):
            identifier = match.group(1)
            fail_index = pending_failures.pop(identifier, None)
            if fail_index is None:
                continue
            if index - fail_index + 1 > max_span_length:
                continue
            spans.append(
                SalientSpan(
                    turn_indices=list(range(fail_index, index + 1)),
                    reason="test_transition",
                    signal_text=identifier,
                )
            )

    return spans


def _merge_spans(spans: list[SalientSpan]) -> list[SalientSpan]:
    """Merge spans whose turn ranges overlap, so a turn is never rendered
    twice. Every span produced by this module covers a contiguous range of
    indices, so a merged span's `(min, max)` fully describes it."""

    if not spans:
        return []

    ordered = sorted(spans, key=lambda span: (min(span.turn_indices), max(span.turn_indices)))
    merged = [ordered[0]]

    for span in ordered[1:]:
        current = merged[-1]
        current_start, current_end = min(current.turn_indices), max(current.turn_indices)
        span_start, span_end = min(span.turn_indices), max(span.turn_indices)

        if span_start <= current_end and current_start <= span_end:
            combined = sorted(set(current.turn_indices) | set(span.turn_indices))
            merged[-1] = SalientSpan(
                turn_indices=combined, reason=current.reason, signal_text=current.signal_text
            )
        else:
            merged.append(span)

    return merged


def find_salient_spans(
    trace: SessionTrace,
    max_test_transition_span: int = DEFAULT_MAX_TEST_TRANSITION_SPAN,
) -> list[SalientSpan]:
    """The turn-indexed spans of `trace` worth showing the reflector: errors
    (with one turn of surrounding context on each side), retry/reversal
    turns, and fail-then-pass test transitions. Overlapping spans are
    merged.

    This only covers signals anchored to a specific turn. `trace.errors` is
    a session-level list with no turn index to anchor to, so it is never
    represented here — `render_salient_excerpt` renders it separately, as a
    preface, not as a `SalientSpan`.
    """

    raw_spans = [
        *_error_spans(trace),
        *_retry_and_reversal_spans(trace),
        *_test_transition_spans(trace, max_test_transition_span),
    ]
    return _merge_spans(raw_spans)


def _render_turn(index: int, turn: Turn) -> str:
    lines = [f"[{index}] {turn.role.value}: {turn.content}"]
    for result in turn.tool_results:
        status = "error" if result.is_error else "ok"
        lines.append(f"    tool_result({result.name}, {status}): {result.output}")
    return "\n".join(lines)


def render_salient_excerpt(trace: SessionTrace, spans: list[SalientSpan]) -> str:
    """Render only the turns covered by `spans`, in order, with an
    `[... N turns omitted ...]` marker standing in for each gap. This is
    the text that actually reaches the reflector prompt — never the raw
    transcript.

    `trace.errors` (session-level, not turn-indexed) is never in `spans`
    (see `find_salient_spans`), so it is prepended here as a distinct
    "## Session-level errors" block, one line per entry, whenever
    `trace.errors` is non-empty. This keeps the two signal sources —
    per-turn spans and session-level errors — visually separate rather
    than forcing the latter into fake turn indices.
    """

    merged = _merge_spans(spans)
    total_turns = len(trace.turns)

    pieces: list[str] = []
    cursor = 0
    for span in merged:
        start, end = min(span.turn_indices), max(span.turn_indices)
        if start > cursor:
            pieces.append(f"[... {start - cursor} turns omitted ...]")
        pieces.extend(_render_turn(i, trace.turns[i]) for i in range(start, end + 1))
        cursor = end + 1

    if cursor < total_turns:
        pieces.append(f"[... {total_turns - cursor} turns omitted ...]")

    body = "\n".join(pieces)

    if trace.errors:
        error_lines = ["## Session-level errors", *(f"- {error}" for error in trace.errors)]
        error_block = "\n".join(error_lines)
        return f"{error_block}\n\n{body}" if body else error_block

    return body

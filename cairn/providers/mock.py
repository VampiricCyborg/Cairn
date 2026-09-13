"""Deterministic mock provider, for CI and offline runs."""

from cairn.core.models import (
    Confidence,
    Diff,
    Entry,
    EntryStatus,
    EntryType,
    Evidence,
    SessionTrace,
)
from cairn.core.similarity import DUPLICATE_WORD_OVERLAP, title_overlap
from cairn.providers.base import make_entry_id

_TITLE_MAX_LEN = 100


def _collect_error_texts(trace: SessionTrace) -> list[str]:
    """Distinct, non-empty error strings from `trace.errors` and any failed
    tool result, in trace order."""

    seen: set[str] = set()
    texts: list[str] = []

    def _add(text: str) -> None:
        normalized = text.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            texts.append(normalized)

    for error in trace.errors:
        _add(error)
    for turn in trace.turns:
        for result in turn.tool_results:
            if result.is_error:
                _add(str(result.output) if result.output is not None else f"{result.name} failed")

    return texts


def _title_from_error(error_text: str) -> str:
    first_line = error_text.strip().splitlines()[0].strip()
    if len(first_line) > _TITLE_MAX_LEN:
        first_line = first_line[: _TITLE_MAX_LEN - 1].rstrip() + "…"
    return first_line or "Unspecified error"


def _related_diffs(error_text: str, diffs: list[Diff]) -> list[Diff]:
    lowered = error_text.lower()
    return [diff for diff in diffs if diff.file.lower() in lowered]


def _render_body(error_text: str, related: list[Diff]) -> str:
    if related:
        files = "\n".join(f"- {diff.file}" for diff in related)
        what_to_do = (
            f"Review the changes below, made in the same session, for the likely cause:\n\n{files}"
        )
    else:
        what_to_do = "Investigate the error above; no diff in this session touched a related file."

    return f"## What happens\n\n{error_text}\n\n## What to do\n\n{what_to_do}\n"


def _is_near_duplicate(title: str, known: list[Entry]) -> bool:
    """A crude substring/word-overlap check (see `title_overlap`). Real
    near-duplicate detection (rapidfuzz-backed) is the Curator's job in P2/P3."""

    return any(title_overlap(title, entry.title) >= DUPLICATE_WORD_OVERLAP for entry in known)


class MockProvider:
    """Deterministic extractor for CI and offline runs: no network calls, no
    randomness.

    Scans `trace.errors` and failed tool results for distinct error signals
    and emits one staged `gotcha` candidate per signal, up to
    `max_candidates`, skipping anything that near-duplicates `known`.
    """

    def extract(
        self, trace: SessionTrace, known: list[Entry], max_candidates: int
    ) -> list[tuple[Entry, str]]:
        candidates: list[tuple[Entry, str]] = []

        for error_text in _collect_error_texts(trace):
            if len(candidates) >= max_candidates:
                break

            title = _title_from_error(error_text)
            if _is_near_duplicate(title, known):
                continue

            related = _related_diffs(error_text, trace.diffs)
            entry = Entry(
                id=make_entry_id(EntryType.GOTCHA, f"{trace.session_id}:{error_text}"),
                type=EntryType.GOTCHA,
                title=title,
                status=EntryStatus.STAGED,
                spec_version="0.1.0",
                scope=[diff.file for diff in related],
                confidence=Confidence.MEDIUM,
                evidence=Evidence(
                    harness=trace.harness,
                    session_id=trace.session_id,
                    captured_at=trace.ended_at,
                    artifacts=[diff.file for diff in related],
                ),
                created=trace.ended_at,
                updated=trace.ended_at,
            )
            body = _render_body(error_text, related)
            candidates.append((entry, body))

        return candidates

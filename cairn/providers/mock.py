"""Deterministic mock provider, for CI and offline runs."""

import hashlib

from cairn.core.models import (
    Confidence,
    Diff,
    Entry,
    EntryStatus,
    EntryType,
    Evidence,
    SessionTrace,
)

_TITLE_MAX_LEN = 100
_DUPLICATE_WORD_OVERLAP = 0.6


def _entry_id(entry_type: EntryType, seed: str) -> str:
    """A stable id, `"{type}-{6 hex chars}"`, derived from `seed` — no
    `uuid4`, so the same seed always produces the same id."""

    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return f"{entry_type.value}-{digest[:6]}"


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
            "Review the changes below, made in the same session, for the likely "
            f"cause:\n\n{files}"
        )
    else:
        what_to_do = "Investigate the error above; no diff in this session touched a related file."

    return f"## What happens\n\n{error_text}\n\n## What to do\n\n{what_to_do}\n"


def _word_set(text: str) -> set[str]:
    return {word for word in text.lower().split() if word}


def _is_near_duplicate(title: str, known: list[Entry]) -> bool:
    """A crude substring/word-overlap check. Real near-duplicate detection
    (rapidfuzz-backed) is the Curator's job in P2/P3."""

    candidate_lower = title.lower()
    candidate_words = _word_set(title)

    for entry in known:
        known_lower = entry.title.lower()
        if candidate_lower in known_lower or known_lower in candidate_lower:
            return True

        known_words = _word_set(entry.title)
        if not candidate_words or not known_words:
            continue
        smaller = min(len(candidate_words), len(known_words))
        if len(candidate_words & known_words) / smaller >= _DUPLICATE_WORD_OVERLAP:
            return True

    return False


class MockProvider:
    """Deterministic extractor for CI and offline runs: no network calls, no
    randomness.

    Scans `trace.errors` and failed tool results for distinct error signals
    and emits one staged `gotcha` candidate per signal, up to
    `max_candidates`, skipping anything that near-duplicates `known`.
    """

    def __init__(self) -> None:
        self._bodies: dict[str, str] = {}

    def extract(
        self, trace: SessionTrace, known: list[Entry], max_candidates: int
    ) -> list[Entry]:
        candidates: list[Entry] = []

        for error_text in _collect_error_texts(trace):
            if len(candidates) >= max_candidates:
                break

            title = _title_from_error(error_text)
            if _is_near_duplicate(title, known):
                continue

            related = _related_diffs(error_text, trace.diffs)
            entry = Entry(
                id=_entry_id(EntryType.GOTCHA, f"{trace.session_id}:{error_text}"),
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
            self._bodies[entry.id] = _render_body(error_text, related)
            candidates.append(entry)

        return candidates

    def body_for(self, entry_id: str) -> str | None:
        """The Markdown body built for `entry_id` by the most recent
        `extract()` call, or `None` if no such candidate was produced."""

        return self._bodies.get(entry_id)

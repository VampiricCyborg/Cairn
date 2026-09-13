"""Shared logic for JSON-schema-constrained providers (Ollama, OpenAI-compatible).

Unlike the Anthropic provider, these backends have no forced tool-use: the
model is merely asked to conform to `candidate_entry_json_schema()`, and the
model behind them is often smaller/faster and thus more prone than Claude to
producing slightly malformed JSON (a stray trailing comma, or the whole
object wrapped in a ```json fence). This module factors the resulting
request-shape and repair logic so `ollama.py` and `openai.py` don't each
reimplement it.
"""

import hashlib
import logging
import re
from json import JSONDecodeError, loads
from typing import Any

from pydantic import ValidationError

from cairn.core.models import (
    CandidateEntry,
    Entry,
    EntryStatus,
    Evidence,
    SessionTrace,
    candidate_entry_json_schema,
)
from cairn.providers.base import make_entry_id

logger = logging.getLogger(__name__)

DEFAULT_MIN_SESSION_TURNS = 4

_SPEC_VERSION = "0.1.0"
_LEADING_FENCE_RE = re.compile(r"^```(?:json)?\s*")
_TRAILING_FENCE_RE = re.compile(r"\s*```$")


def candidates_json_schema() -> dict[str, Any]:
    """`candidate_entry_json_schema()` wrapped as `{"candidates": [...]}`.

    Schema-constrained output needs one root object, and wrapping a list in
    it is what lets the model answer with zero candidates. `$defs` (the
    candidate schema's enums) are hoisted to the root so its `#/$defs/...`
    refs still resolve, and `$schema`/`$id` are dropped because they only
    belong on a root document.
    """

    candidate = candidate_entry_json_schema()
    candidate.pop("$schema", None)
    candidate.pop("$id", None)
    defs = candidate.pop("$defs", None)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"candidates": {"type": "array", "items": candidate}},
        "required": ["candidates"],
        "additionalProperties": False,
    }
    if defs:
        schema["$defs"] = defs
    return schema


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    stripped = _LEADING_FENCE_RE.sub("", stripped)
    stripped = _TRAILING_FENCE_RE.sub("", stripped)
    return stripped.strip()


def parse_candidates(text: str, *, provider: str) -> list[object]:
    """Parse a model's raw response body into a `candidates` list.

    Tries `json.loads` on `text` as-is; on failure, strips a markdown code
    fence and retries once. If both attempts fail, or the parsed JSON isn't
    a `{"candidates": [...]}` object, logs a warning and returns `[]` rather
    than raising -- one malformed response must never crash extraction.
    """

    for candidate_text in (text, _strip_code_fence(text)):
        try:
            parsed = loads(candidate_text)
        except JSONDecodeError:
            continue

        if not isinstance(parsed, dict):
            logger.warning(
                "%s: response JSON is not an object; treating as zero candidates", provider
            )
            return []
        raw = parsed.get("candidates")
        if not isinstance(raw, list):
            logger.warning(
                "%s: response JSON has no candidates list; treating as zero candidates", provider
            )
            return []
        return raw

    logger.warning(
        "%s: response is not valid JSON even after stripping code fences; treating as zero "
        "candidates",
        provider,
    )
    return []


def build_evidence(trace: SessionTrace, excerpt: str) -> Evidence:
    """`Evidence` for a candidate extracted from `trace`'s salient `excerpt`."""

    return Evidence(
        harness=trace.harness,
        session_id=trace.session_id,
        captured_at=trace.ended_at,
        artifacts=[diff.file for diff in trace.diffs],
        excerpt_sha256=hashlib.sha256(excerpt.encode("utf-8")).hexdigest(),
    )


def _to_entry(raw: object, trace: SessionTrace, evidence: Evidence) -> tuple[Entry, str] | None:
    """Validate one raw candidate and complete it into a staged `Entry` plus
    body. Returns `None`, logging a warning, if it fails validation as either
    a `CandidateEntry` or the completed `Entry`."""

    try:
        candidate = CandidateEntry.model_validate(raw)
        entry = Entry(
            id=make_entry_id(candidate.type, f"{trace.session_id}:{candidate.title}"),
            type=candidate.type,
            title=candidate.title,
            status=EntryStatus.STAGED,
            spec_version=_SPEC_VERSION,
            scope=candidate.scope,
            tags=candidate.tags,
            confidence=candidate.confidence,
            evidence=evidence.model_copy(deep=True),
            created=trace.ended_at,
            updated=trace.ended_at,
        )
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or '<root>'}: {error['msg']}"
            for error in exc.errors()
        )
        logger.warning("dropping invalid candidate (%s)", problems)
        return None

    return entry, candidate.body


def raw_candidates_to_entries(
    raw_candidates: list[object],
    trace: SessionTrace,
    evidence: Evidence,
    max_candidates: int,
) -> list[tuple[Entry, str]]:
    """Validate and cap `raw_candidates` into at most `max_candidates` staged
    `(Entry, body)` pairs, dropping invalid or duplicate-id entries."""

    if len(raw_candidates) > max_candidates:
        logger.info(
            "model returned %d candidates; keeping at most %d", len(raw_candidates), max_candidates
        )

    candidates: list[tuple[Entry, str]] = []
    seen_ids: set[str] = set()
    for raw in raw_candidates:
        if len(candidates) >= max_candidates:
            break
        converted = _to_entry(raw, trace, evidence)
        if converted is None or converted[0].id in seen_ids:
            continue
        seen_ids.add(converted[0].id)
        candidates.append(converted)

    return candidates

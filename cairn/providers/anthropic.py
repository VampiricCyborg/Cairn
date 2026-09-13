"""Anthropic provider implementation."""

import hashlib
import logging
from typing import Any

import anthropic
from anthropic.types import Message, ToolParam, ToolUseBlock
from pydantic import ValidationError

from cairn.core.models import (
    CandidateEntry,
    Entry,
    EntryStatus,
    Evidence,
    SessionTrace,
    candidate_entry_json_schema,
)
from cairn.core.redactor import Redactor
from cairn.core.reflector import build_reflector_prompt
from cairn.core.salience import find_salient_spans, is_reflectable, render_salient_excerpt
from cairn.providers.base import make_entry_id

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MIN_SESSION_TURNS = 4
TOOL_NAME = "record_candidate_entries"

_SPEC_VERSION = "0.1.0"


def _tool_input_schema() -> dict[str, Any]:
    """`candidate_entry_json_schema()` wrapped as `{"candidates": [...]}`.

    A tool's input must be one object, and wrapping a list in it is what
    lets the model answer with zero candidates. The candidate schema's
    `$defs` (its enums) are hoisted to the root so its `#/$defs/...` refs
    still resolve, and `$schema`/`$id` are dropped because they only belong
    on a root document.
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


def _tool(max_candidates: int) -> ToolParam:
    return {
        "name": TOOL_NAME,
        "description": (
            "Record the candidate entries this session excerpt supports, at most "
            f"{max_candidates}. Call it with an empty `candidates` list when nothing in the "
            "excerpt meets the acceptance bar."
        ),
        "input_schema": _tool_input_schema(),
    }


def _raw_candidates(response: Message) -> list[object]:
    """The unvalidated `candidates` list from the forced tool call, or `[]`
    (with a warning) if the response doesn't contain a usable one."""

    if response.stop_reason == "max_tokens":
        logger.warning("response hit max_tokens; candidates may be truncated")

    for block in response.content:
        if isinstance(block, ToolUseBlock) and block.name == TOOL_NAME:
            raw = block.input.get("candidates")
            if isinstance(raw, list):
                return raw
            logger.warning("%s input has no candidates list; treating as zero", TOOL_NAME)
            return []

    logger.warning(
        "response (stop_reason=%s) has no %s call; treating as zero candidates",
        response.stop_reason,
        TOOL_NAME,
    )
    return []


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


class AnthropicProvider:
    """Extracts candidate entries with a Claude model through the Anthropic API.

    Config-agnostic: redaction rules and the minimum session length are
    constructor arguments rather than read from `config.toml`. The trace is
    redacted before anything else looks at it, an unreflectable trace
    returns `[]` without an API call, and only the salient excerpt, never
    the full transcript, is sent to the model.

    The client is injectable for tests; by default an `anthropic.Anthropic()`
    client is created on first use, so a provider that only ever sees
    unreflectable traces never needs credentials.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_output_tokens: int = 2000,
        *,
        deny_globs: list[str] | None = None,
        patterns: list[str] | None = None,
        min_session_turns: int = DEFAULT_MIN_SESSION_TURNS,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.min_session_turns = min_session_turns
        self._redactor = Redactor(deny_globs=deny_globs or [], patterns=patterns or [])
        self._client = client

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client

    def extract(
        self, trace: SessionTrace, known: list[Entry], max_candidates: int
    ) -> list[tuple[Entry, str]]:
        if max_candidates <= 0:
            return []

        redacted = self._redactor.redact_trace(trace)
        if not is_reflectable(redacted, self.min_session_turns):
            return []

        excerpt = render_salient_excerpt(redacted, find_salient_spans(redacted))
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_output_tokens,
            tools=[_tool(max_candidates)],
            tool_choice={"type": "tool", "name": TOOL_NAME},
            messages=[{"role": "user", "content": build_reflector_prompt(excerpt, known)}],
        )

        raw_candidates = _raw_candidates(response)
        if len(raw_candidates) > max_candidates:
            logger.info(
                "model returned %d candidates; keeping at most %d",
                len(raw_candidates),
                max_candidates,
            )

        evidence = Evidence(
            harness=redacted.harness,
            session_id=redacted.session_id,
            captured_at=redacted.ended_at,
            artifacts=[diff.file for diff in redacted.diffs],
            excerpt_sha256=hashlib.sha256(excerpt.encode("utf-8")).hexdigest(),
        )

        candidates: list[tuple[Entry, str]] = []
        seen_ids: set[str] = set()
        for raw in raw_candidates:
            if len(candidates) >= max_candidates:
                break
            converted = _to_entry(raw, redacted, evidence)
            if converted is None or converted[0].id in seen_ids:
                continue
            seen_ids.add(converted[0].id)
            candidates.append(converted)

        return candidates

"""Pydantic v2 models for the `.cairn/` store.

These models are the single source of truth for the store format. The JSON
Schemas checked into `cairn/schema/` are *generated* from them via
`entry_json_schema()` / `trace_json_schema()` (see `scripts/generate_schemas.py`)
rather than hand-maintained, so the two can never silently drift apart.
`tests/test_models.py` asserts the checked-in files still match.
"""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
ENTRY_SCHEMA_ID = "https://cairn.dev/schema/entry.schema.json"
TRACE_SCHEMA_ID = "https://cairn.dev/schema/trace.schema.json"


class EntryType(StrEnum):
    """The three kinds of knowledge a Cairn entry can hold."""

    STRATEGY = "strategy"
    GOTCHA = "gotcha"
    FACT = "fact"


class EntryStatus(StrEnum):
    """Persisted lifecycle state of an entry file. See SPEC.md for the full
    state machine, including the pre-persistence states (candidate, dropped)
    that never become a `status` value."""

    STAGED = "staged"
    APPROVED = "approved"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"


class Confidence(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TurnRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Evidence(BaseModel):
    """Provenance for an entry: where it came from and what it points at."""

    model_config = ConfigDict(extra="forbid")

    harness: str
    session_id: str | None = None
    captured_at: datetime
    artifacts: list[str] = Field(default_factory=list)
    commit: str | None = None
    excerpt_sha256: str | None = None


class Review(BaseModel):
    """Who approved an entry, and when. Absent until a human acts on it."""

    model_config = ConfigDict(extra="forbid")

    approved_by: str | None = None
    approved_at: datetime | None = None


class Usage(BaseModel):
    """Injection and feedback counters, updated as the entry is used."""

    model_config = ConfigDict(extra="forbid")

    injected: int = 0
    marked_useful: int = 0
    marked_stale: int = 0


class Entry(BaseModel):
    """A single unit of curated, human-approved knowledge in the `.cairn/` store."""

    model_config = ConfigDict(extra="forbid", title="Cairn entry")

    id: str
    type: EntryType
    title: str
    status: EntryStatus
    spec_version: str
    scope: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    confidence: Confidence
    evidence: Evidence
    created: datetime
    updated: datetime
    supersedes: list[str] = Field(default_factory=list)
    review: Review | None = None
    usage: Usage = Field(default_factory=Usage)


class ToolCall(BaseModel):
    """A tool invocation made by the agent during a turn."""

    model_config = ConfigDict(extra="forbid")

    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """The result of one tool invocation."""

    model_config = ConfigDict(extra="forbid")

    name: str
    output: Any = None
    is_error: bool = False


class Turn(BaseModel):
    """One turn of a session transcript, normalized across harnesses."""

    model_config = ConfigDict(extra="forbid")

    role: TurnRole
    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)


class Diff(BaseModel):
    """A file changed during the session, as a unified diff."""

    model_config = ConfigDict(extra="forbid")

    file: str
    patch: str


class SessionTrace(BaseModel):
    """Canonical, harness-agnostic representation of one coding session."""

    model_config = ConfigDict(extra="forbid", title="Cairn session trace")

    session_id: str
    harness: str
    started_at: datetime
    ended_at: datetime
    turns: list[Turn] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    diffs: list[Diff] = Field(default_factory=list)
    outcome: str


def entry_json_schema() -> dict[str, Any]:
    """The JSON Schema 2020-12 document for `Entry`, as checked into
    `cairn/schema/entry.schema.json`."""

    return {
        "$schema": JSON_SCHEMA_DIALECT,
        "$id": ENTRY_SCHEMA_ID,
        **Entry.model_json_schema(),
    }


def trace_json_schema() -> dict[str, Any]:
    """The JSON Schema 2020-12 document for `SessionTrace`, as checked into
    `cairn/schema/trace.schema.json`."""

    return {
        "$schema": JSON_SCHEMA_DIALECT,
        "$id": TRACE_SCHEMA_ID,
        **SessionTrace.model_json_schema(),
    }

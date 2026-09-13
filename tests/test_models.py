"""Tests for cairn.core.models.

`cairn/schema/*.json` is generated from these models (see
`scripts/generate_schemas.py`), so the tests here mainly guard against drift:
if a model changes and nobody regenerates the schema files, this fails in CI.
"""

import json
from pathlib import Path

from cairn.core.models import (
    CANDIDATE_ENTRY_SCHEMA_ID,
    candidate_entry_json_schema,
    entry_json_schema,
    trace_json_schema,
)

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "cairn" / "schema"


def _load(name: str) -> dict[str, object]:
    return json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))


def test_entry_schema_matches_model() -> None:
    assert _load("entry.schema.json") == entry_json_schema()


def test_trace_schema_matches_model() -> None:
    assert _load("trace.schema.json") == trace_json_schema()


def test_candidate_entry_schema_covers_only_model_produced_fields() -> None:
    schema = candidate_entry_json_schema()

    assert schema["$id"] == CANDIDATE_ENTRY_SCHEMA_ID
    assert set(schema["properties"]) == {
        "id",
        "type",
        "title",
        "scope",
        "tags",
        "confidence",
        "body",
    }
    assert schema["additionalProperties"] is False

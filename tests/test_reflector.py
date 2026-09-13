"""Tests for cairn.core.reflector."""

import re
from datetime import UTC, datetime
from pathlib import Path

from cairn.core.models import Confidence, Entry, EntryStatus, EntryType, Evidence
from cairn.core.reflector import REFLECTOR_PROMPT_TEMPLATE, build_reflector_prompt

SPEC_PATH = Path(__file__).resolve().parent.parent / "SPEC.md"
_CRITERION_ROW_RE = re.compile(r"^\| \*\*[^*]+\*\* \| .+ \|$", re.MULTILINE)


def _spec_quality_criteria_rows() -> list[str]:
    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec.split("\n## Entry quality criteria", 1)[1].split("\n## ", 1)[0]
    return _CRITERION_ROW_RE.findall(section)


def _entry(entry_id: str, entry_type: EntryType, title: str) -> Entry:
    return Entry(
        id=entry_id,
        type=entry_type,
        title=title,
        status=EntryStatus.APPROVED,
        spec_version="0.1.0",
        scope=["zz-scope-marker/**"],
        tags=["zz-tag-marker"],
        confidence=Confidence.HIGH,
        evidence=Evidence(harness="manual", captured_at=datetime(2026, 9, 1, 0, 0, 0, tzinfo=UTC)),
        created=datetime(2026, 9, 1, 0, 0, 0, tzinfo=UTC),
        updated=datetime(2026, 9, 1, 0, 0, 0, tzinfo=UTC),
    )


def test_prompt_includes_spec_quality_criteria_verbatim() -> None:
    rows = _spec_quality_criteria_rows()

    assert len(rows) == 7
    prompt = build_reflector_prompt("excerpt", [])
    for row in rows:
        assert row in prompt


def test_prompt_includes_excerpt_verbatim() -> None:
    excerpt = "[2] assistant: FAILED tests/test_db.py::test_x with {'code': 42} and {excerpt}"

    prompt = build_reflector_prompt(excerpt, [])

    assert f"<session_excerpt>\n{excerpt}\n</session_excerpt>" in prompt


def test_known_entries_render_id_type_and_title_only() -> None:
    known = [
        _entry("gotcha-1a2b3c", EntryType.GOTCHA, "Run alembic upgrade before the test suite"),
        _entry("fact-4d5e6f", EntryType.FACT, "CI runs on Python 3.11 only"),
    ]

    prompt = build_reflector_prompt("excerpt", known)

    assert "- gotcha-1a2b3c [gotcha] Run alembic upgrade before the test suite" in prompt
    assert "- fact-4d5e6f [fact] CI runs on Python 3.11 only" in prompt
    assert "zz-scope-marker" not in prompt
    assert "zz-tag-marker" not in prompt


def test_no_known_entries_is_stated_explicitly() -> None:
    prompt = build_reflector_prompt("excerpt", [])

    assert "<known_entries>\n(none yet)\n</known_entries>" in prompt


def test_prompt_allows_zero_candidates_instead_of_filling_a_quota() -> None:
    prompt = build_reflector_prompt("excerpt", [])

    assert "return\n  zero candidates" in prompt
    assert "never invent" in prompt


def test_prompt_is_a_template_constant_not_inlined() -> None:
    assert "{excerpt}" in REFLECTOR_PROMPT_TEMPLATE
    assert "{known_entries}" in REFLECTOR_PROMPT_TEMPLATE

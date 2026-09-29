"""Tests for `cairn stats --review`, end to end through the real CLI.

Decisions are made by driving `cairn reflect` and `cairn review` rather than
by hand-writing log lines, so these also pin that the review flow and the
extraction path actually record what they did.

Assertions run over whitespace-collapsed output: the numbers and their labels
are the contract, the column padding is not.
"""

import re
from pathlib import Path

import frontmatter
import pytest
from typer.testing import CliRunner

from cairn.cli import app
from cairn.core import review_log
from cairn.core.review_log import HUMAN_ACTIONS, ReviewAction
from cairn.core.store import Store
from cairn.review import cli_review

runner = CliRunner()

_FIXTURE = Path(__file__).parent / "fixtures" / "session_trace.json"


@pytest.fixture(autouse=True)
def _reviewer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAIRN_REVIEWER", "tester")


def _norm(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text)


def _init(tmp_path: Path) -> Store:
    result = runner.invoke(app, ["init", str(tmp_path)])
    assert result.exit_code == 0, result.output
    return Store(tmp_path / ".cairn")


def _reflect(tmp_path: Path, trace: Path = _FIXTURE) -> None:
    result = runner.invoke(app, ["reflect", str(tmp_path), "--trace", str(trace)])
    assert result.exit_code == 0, result.output


def _review(tmp_path: Path, keys: str) -> str:
    result = runner.invoke(app, ["review", str(tmp_path)], input=keys)
    assert result.exit_code == 0, result.output
    return result.output


def _stats(tmp_path: Path, *args: str) -> tuple[int, str]:
    result = runner.invoke(app, ["stats", str(tmp_path), *args])
    return result.exit_code, _norm(result.output)


def _second_trace(tmp_path: Path, *, distinct_errors: bool) -> Path:
    """The fixture trace under a fresh session id, so its candidates get fresh
    ids. With `distinct_errors`, the error text differs too, so the candidates
    are about different columns rather than restatements."""

    raw = _FIXTURE.read_text(encoding="utf-8").replace(
        "b1a2c3d4-5e6f-4a1b-9c3d-7f8e9a0b1c2d", "c2b3d4e5-6f70-4a1b-9c3d-7f8e9a0b1c2e"
    )
    if distinct_errors:
        raw = raw.replace("users.last_seen_at", "orders.settled_at").replace(
            "user_sessions", "order_lines"
        )
    path = tmp_path / "trace2.json"
    path.write_text(raw, encoding="utf-8")
    return path


def test_stats_requires_the_review_flag(tmp_path: Path) -> None:
    _init(tmp_path)

    exit_code, output = _stats(tmp_path)

    assert exit_code == 1
    assert "--review" in output


def test_stats_on_an_empty_log(tmp_path: Path) -> None:
    _init(tmp_path)

    exit_code, output = _stats(tmp_path, "--review")

    assert exit_code == 0, output
    assert "no review decisions logged yet" in output


def test_stats_on_a_missing_store_errors_cleanly(tmp_path: Path) -> None:
    exit_code, output = _stats(tmp_path, "--review")

    assert exit_code == 1
    assert "error" in output.lower()


def test_reflect_records_staged_candidates_with_no_human_records(tmp_path: Path) -> None:
    store = _init(tmp_path)
    _reflect(tmp_path)

    records = review_log.read(store.root)

    assert [record.action for record in records] == [ReviewAction.STAGED] * 2
    # MockProvider exposes no model, so nothing is guessed into the field.
    assert {record.model for record in records} == {None}

    exit_code, output = _stats(tmp_path, "--review")
    assert exit_code == 0, output
    assert "shown to a human 0" in output
    assert "approval rate n/a (0/0 shown)" in output
    assert "staged for review 2" in output
    assert "gate-drop rate 0.0% (0/2 extracted)" in output


def test_gate_drop_is_recorded_with_the_gate_that_fired(tmp_path: Path) -> None:
    store = _init(tmp_path)
    _reflect(tmp_path)
    # Reject one candidate, which leaves a tombstone...
    _review(tmp_path, "r\nnot project-specific\ns\n")
    # ...so re-extracting the same error from a different session drops it.
    _reflect(tmp_path, _second_trace(tmp_path, distinct_errors=False))

    drops = [r for r in review_log.read(store.root) if r.action is ReviewAction.GATE_DROP]

    assert len(drops) == 1
    assert drops[0].reason == "tombstone"

    exit_code, output = _stats(tmp_path, "--review")
    assert exit_code == 0, output
    assert "dropped by a gate 1" in output
    assert "1 tombstone" in output


def test_approval_rate_counts_only_candidates_shown_to_a_human(tmp_path: Path) -> None:
    """Four staged candidates, two approved, must read as 50% over the humans'
    four -- not diluted by the extraction-side records sharing the log."""

    store = _init(tmp_path)
    _reflect(tmp_path)
    _reflect(tmp_path, _second_trace(tmp_path, distinct_errors=True))
    assert len(list(store.staging_dir.glob("*.md"))) == 4

    # approve (and accept the default reviewer name, which is prompted once
    # per session), approve, reject with a reason, skip
    _review(tmp_path, "a\n\na\nr\nnot stable\ns\n")

    records = review_log.read(store.root)
    human = [r for r in records if r.action in HUMAN_ACTIONS]
    assert len(human) == 4
    assert len(records) == 8  # 4 staged + 4 human decisions

    exit_code, output = _stats(tmp_path, "--review")
    assert exit_code == 0, output
    assert "shown to a human 4" in output
    assert "approved 2" in output
    assert "rejected 1" in output
    assert "skipped 1" in output
    assert "approval rate 50.0% (2/4 shown)" in output
    assert "excluding skipped 66.7% (2/3 decided)" in output
    # The four `staged` records are counted in their own block, and only there.
    assert "extracted 4" in output
    assert "rejection reasons" in output
    assert "1 not stable" in output


def test_edited_approval_records_the_changed_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _init(tmp_path)
    _reflect(tmp_path)

    def _rewrite(path: Path) -> None:
        post = frontmatter.load(path)
        post["title"] = "A sharper title written by the reviewer"
        post["confidence"] = "high"
        path.write_text(frontmatter.dumps(post) + "\n", encoding="utf-8")

    monkeypatch.setattr(cli_review, "_launch_editor", _rewrite)

    _review(tmp_path, "e\n\ns\n")

    edits = [r for r in review_log.read(store.root) if r.action is ReviewAction.APPROVE_WITH_EDIT]
    assert len(edits) == 1
    assert edits[0].edited_fields == ["confidence", "title"]
    assert edits[0].title == "A sharper title written by the reviewer"

    exit_code, output = _stats(tmp_path, "--review")
    assert exit_code == 0, output
    assert "approved with edit 1" in output
    assert "approval rate 50.0% (1/2 shown)" in output


def test_merge_counts_as_an_approval(tmp_path: Path) -> None:
    store = _init(tmp_path)
    _reflect(tmp_path)
    # Approve the first candidate so there is something to merge into, then quit.
    _review(tmp_path, "a\n\nq\n")
    (approved,) = store.approved()

    _review(tmp_path, f"m\n{approved.id}\n\n")

    merges = [r for r in review_log.read(store.root) if r.action is ReviewAction.MERGE]
    assert len(merges) == 1
    assert merges[0].reason == f"merged into {approved.id}"

    exit_code, output = _stats(tmp_path, "--review")
    assert exit_code == 0, output
    assert "merged 1" in output
    assert "approval rate 100.0% (2/2 shown)" in output

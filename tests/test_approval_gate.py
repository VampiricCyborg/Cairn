"""`[review] require_human_approval` in config.toml is enforced, not decorative.

v0 has no auto-approval mode, so the key must be the boolean `true`. Anything
else -- `false`, absent, mistyped, or an unreadable config -- makes every
command that would write to `entries/` refuse, with an error that names the
key, before a single candidate is shown or touched.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cairn.cli import app
from cairn.core.config import HumanApprovalRequiredError
from cairn.core.models import Confidence, Entry, EntryStatus, EntryType, Evidence
from cairn.core.store import Store
from cairn.review.cli_review import ReviewSession, run_review

runner = CliRunner()

_CAPTURED_AT = datetime(2026, 9, 12, 11, 0, 0, tzinfo=UTC)

#: config.toml contents (None = no file at all) under which the gate must refuse.
_REFUSING_CONFIGS = {
    "file-missing": None,
    "unparsable": "this is = not [valid toml",
    "no-review-table": '[provider]\nname = "mock"\n',
    "key-absent": "[review]\n",
    "false": "[review]\nrequire_human_approval = false\n",
    "string-true": '[review]\nrequire_human_approval = "true"\n',
    "integer-one": "[review]\nrequire_human_approval = 1\n",
}


@pytest.fixture(autouse=True)
def _reviewer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAIRN_REVIEWER", "tester")


@pytest.fixture
def store(tmp_path: Path) -> Store:
    result = runner.invoke(app, ["init", str(tmp_path)])
    assert result.exit_code == 0, result.output
    return Store(tmp_path / ".cairn")


def _stage_one(store: Store) -> Path:
    entry = Entry(
        id="fact-abcdef",
        type=EntryType.FACT,
        title="Config values load from .env before defaults",
        status=EntryStatus.STAGED,
        spec_version="0.1.0",
        confidence=Confidence.MEDIUM,
        evidence=Evidence(harness="claude-code", captured_at=_CAPTURED_AT),
        created=_CAPTURED_AT,
        updated=_CAPTURED_AT,
    )
    return store.write_staged(entry, "Body.\n")


def _set_config(store: Store, contents: str | None) -> None:
    config_path = store.root / "config.toml"
    if contents is None:
        config_path.unlink()
    else:
        config_path.write_text(contents, encoding="utf-8")


def _entries_files(store: Store) -> list[Path]:
    """Entry files under entries/, ignoring the .gitkeep placeholders that keep
    the committed layout intact (git cannot track an empty directory)."""

    return sorted(
        path for path in store.entries_dir.rglob("*") if path.is_file() and path.name != ".gitkeep"
    )


def test_init_writes_a_config_that_passes_the_gate(store: Store) -> None:
    staged = _stage_one(store)

    result = runner.invoke(app, ["review", str(store.root.parent)], input="a\n\n")

    assert result.exit_code == 0, result.output
    assert not staged.exists()
    assert len(_entries_files(store)) == 1


@pytest.mark.parametrize("name", _REFUSING_CONFIGS)
def test_review_refuses_and_names_the_key(store: Store, name: str) -> None:
    staged = _stage_one(store)
    before = staged.read_bytes()
    _set_config(store, _REFUSING_CONFIGS[name])

    # Scripted input that would approve if the gate let review start.
    result = runner.invoke(app, ["review", str(store.root.parent)], input="a\n\n")

    assert result.exit_code == 1, result.output
    assert "require_human_approval" in result.output
    assert "candidate 1 of 1" not in result.output  # refused before showing anything
    assert staged.read_bytes() == before
    assert _entries_files(store) == []


def test_refusal_explains_what_to_set(store: Store) -> None:
    _stage_one(store)
    _set_config(store, "[review]\nrequire_human_approval = false\n")

    result = runner.invoke(app, ["review", str(store.root.parent)])

    assert "[review]" in result.output
    assert "= true" in result.output
    assert "config.toml" in result.output


def test_refuses_even_with_nothing_staged(store: Store) -> None:
    """A misconfigured gate is reported up front, not only once there is
    something to approve."""

    _set_config(store, "[review]\nrequire_human_approval = false\n")

    result = runner.invoke(app, ["review", str(store.root.parent)])

    assert result.exit_code == 1, result.output
    assert "require_human_approval" in result.output


@pytest.mark.parametrize("name", _REFUSING_CONFIGS)
def test_the_review_session_itself_refuses(store: Store, name: str) -> None:
    """Not just the CLI command: constructing the object that owns the
    entries/ writes is what raises, so programmatic callers are gated too."""

    _set_config(store, _REFUSING_CONFIGS[name])

    with pytest.raises(HumanApprovalRequiredError, match="require_human_approval"):
        ReviewSession(store)
    with pytest.raises(HumanApprovalRequiredError, match="require_human_approval"):
        run_review(store)

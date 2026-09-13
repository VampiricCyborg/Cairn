"""Tests for the `cairn` CLI: init, validate, context."""

import json
from datetime import UTC, datetime
from pathlib import Path

import frontmatter
from typer.testing import CliRunner

from cairn.cli import app
from cairn.core.models import Confidence, Entry, EntryStatus, EntryType, Evidence

runner = CliRunner()


def _make_entry(**overrides: object) -> Entry:
    defaults: dict[str, object] = dict(
        id="fact-abcdef",
        type=EntryType.FACT,
        title="Config values load from .env before defaults",
        status=EntryStatus.APPROVED,
        spec_version="0.1.0",
        confidence=Confidence.MEDIUM,
        evidence=Evidence(
            harness="manual",
            captured_at=datetime(2026, 9, 12, 11, 0, 0, tzinfo=UTC),
        ),
        created=datetime(2026, 9, 12, 11, 0, 0, tzinfo=UTC),
        updated=datetime(2026, 9, 12, 11, 0, 0, tzinfo=UTC),
    )
    defaults.update(overrides)
    return Entry(**defaults)  # type: ignore[arg-type]


def _write_entry(
    directory: Path, entry: Entry, filename: str, body: str = "Some knowledge."
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    post = frontmatter.Post(body, **entry.model_dump(mode="json"))
    path = directory / filename
    path.write_text(frontmatter.dumps(post) + "\n", encoding="utf-8")
    return path


class TestInit:
    def test_creates_expected_layout(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["init", str(tmp_path)])

        assert result.exit_code == 0, result.output
        cairn_root = tmp_path / ".cairn"
        assert (cairn_root / "VERSION").read_text(encoding="utf-8") == "0.1.0"
        config = (cairn_root / "config.toml").read_text(encoding="utf-8")
        assert 'name  = "mock"' in config
        assert (cairn_root / "CONTEXT.md").exists()
        for entry_type in ("strategy", "gotcha", "fact"):
            assert (cairn_root / "entries" / entry_type).is_dir()
        assert (cairn_root / "staging").is_dir()
        assert (cairn_root / "rejected").is_dir()
        assert (cairn_root / "schema" / "entry.schema.json").exists()
        assert (cairn_root / "schema" / "trace.schema.json").exists()

        gitignore = (tmp_path / ".gitignore").read_text(encoding="utf-8")
        assert ".cairn/queue/" in gitignore
        assert ".cairn/traces/" in gitignore

    def test_appends_to_existing_gitignore(self, tmp_path: Path) -> None:
        (tmp_path / ".gitignore").write_text("*.pyc\n", encoding="utf-8")

        result = runner.invoke(app, ["init", str(tmp_path)])

        assert result.exit_code == 0, result.output
        gitignore = (tmp_path / ".gitignore").read_text(encoding="utf-8")
        assert "*.pyc" in gitignore
        assert ".cairn/queue/" in gitignore
        assert ".cairn/traces/" in gitignore

    def test_fails_when_cairn_already_exists(self, tmp_path: Path) -> None:
        (tmp_path / ".cairn").mkdir()

        result = runner.invoke(app, ["init", str(tmp_path)])

        assert result.exit_code == 1
        assert not (tmp_path / ".cairn" / "VERSION").exists()


class TestValidate:
    def test_reports_valid_and_malformed_entries(self, tmp_path: Path) -> None:
        runner.invoke(app, ["init", str(tmp_path)])
        cairn_root = tmp_path / ".cairn"

        valid = _make_entry(id="fact-000001", title="A valid fact", scope=["src/**"])
        _write_entry(cairn_root / "entries" / "fact", valid, "valid.md")

        malformed_path = cairn_root / "entries" / "gotcha" / "malformed.md"
        malformed_path.parent.mkdir(parents=True, exist_ok=True)
        malformed_path.write_text(
            "---\nid: gotcha-bad\ntype: gotcha\ntitle: Missing fields\nstatus: approved\n"
            "spec_version: 0.1.0\n---\n\nbody\n",
            encoding="utf-8",
        )

        result = runner.invoke(app, ["validate", str(tmp_path)])

        assert result.exit_code == 1
        assert "PASS" in result.output
        assert "FAIL" in result.output
        assert str(malformed_path) in result.output

    def test_passes_on_clean_store(self, tmp_path: Path) -> None:
        runner.invoke(app, ["init", str(tmp_path)])
        cairn_root = tmp_path / ".cairn"
        entry = _make_entry(scope=["src/**"])
        _write_entry(cairn_root / "entries" / "fact", entry, "ok.md")

        result = runner.invoke(app, ["validate", str(tmp_path)])

        assert result.exit_code == 0, result.output

    def test_strict_fails_on_empty_scope_warning(self, tmp_path: Path) -> None:
        runner.invoke(app, ["init", str(tmp_path)])
        cairn_root = tmp_path / ".cairn"
        entry = _make_entry()
        _write_entry(cairn_root / "entries" / "fact", entry, "ok.md")

        lenient = runner.invoke(app, ["validate", str(tmp_path)])
        strict = runner.invoke(app, ["validate", str(tmp_path), "--strict"])

        assert lenient.exit_code == 0, lenient.output
        assert strict.exit_code == 1, strict.output
        assert "WARN" in strict.output


class TestContext:
    def test_empty_store_text(self, tmp_path: Path) -> None:
        runner.invoke(app, ["init", str(tmp_path)])

        result = runner.invoke(app, ["context", str(tmp_path)])

        assert result.exit_code == 0, result.output
        assert "no approved entries" in result.output

    def test_empty_store_json(self, tmp_path: Path) -> None:
        runner.invoke(app, ["init", str(tmp_path)])

        result = runner.invoke(app, ["context", str(tmp_path), "--format", "json"])

        assert result.exit_code == 0, result.output
        assert json.loads(result.output) == []

    def test_two_entry_store_text_and_json(self, tmp_path: Path) -> None:
        runner.invoke(app, ["init", str(tmp_path)])
        cairn_root = tmp_path / ".cairn"

        entry_a = _make_entry(
            id="fact-000001", type=EntryType.FACT, title="First fact", scope=["src/**"]
        )
        entry_b = _make_entry(
            id="strategy-000002",
            type=EntryType.STRATEGY,
            title="Second strategy",
            scope=["tests/**"],
        )
        _write_entry(cairn_root / "entries" / "fact", entry_a, "a.md", body="Fact body.")
        _write_entry(
            cairn_root / "entries" / "strategy", entry_b, "b.md", body="Strategy body."
        )

        text_result = runner.invoke(app, ["context", str(tmp_path)])
        assert text_result.exit_code == 0, text_result.output
        assert "First fact" in text_result.output
        assert "Second strategy" in text_result.output

        json_result = runner.invoke(app, ["context", str(tmp_path), "--format", "json"])
        assert json_result.exit_code == 0, json_result.output
        payload = json.loads(json_result.output)
        assert {item["id"] for item in payload} == {"fact-000001", "strategy-000002"}

        scoped = runner.invoke(
            app, ["context", str(tmp_path), "--scope", "src/main.py", "--format", "json"]
        )
        assert scoped.exit_code == 0, scoped.output
        scoped_payload = json.loads(scoped.output)
        assert [item["id"] for item in scoped_payload] == ["fact-000001"]

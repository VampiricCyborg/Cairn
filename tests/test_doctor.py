"""Tests for `cairn doctor`."""

import subprocess
from pathlib import Path

from typer.testing import CliRunner

import cairn.cli as cli
from cairn.cli import app

runner = CliRunner()


def _init(tmp_path: Path) -> None:
    result = runner.invoke(app, ["init", str(tmp_path)])
    assert result.exit_code == 0, result.output


def test_reports_missing_store(tmp_path: Path) -> None:
    result = runner.invoke(app, ["doctor", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "FAIL store" in result.output


def test_reports_healthy_store_and_mock_provider(tmp_path: Path) -> None:
    _init(tmp_path)

    result = runner.invoke(app, ["doctor", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "PASS store" in result.output
    assert "0 entries, 0 staged" in result.output
    assert "PASS provider         mock" in result.output


def test_reports_claude_code_not_installed(tmp_path: Path) -> None:
    _init(tmp_path)

    result = runner.invoke(app, ["doctor", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "WARN claude-code" in result.output
    assert "cairn install claude-code" in result.output


def test_reports_claude_code_installed_after_install(tmp_path: Path) -> None:
    _init(tmp_path)
    install_result = runner.invoke(app, ["install", "claude-code", str(tmp_path)])
    assert install_result.exit_code == 0, install_result.output

    result = runner.invoke(app, ["doctor", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "PASS claude-code      SessionEnd + SessionStart hooks registered" in result.output


def test_reports_anthropic_provider_missing_key(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _init(tmp_path)
    config_path = tmp_path / ".cairn" / "config.toml"
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace('name  = "mock"', 'name  = "anthropic"'),
        encoding="utf-8",
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    result = runner.invoke(app, ["doctor", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "WARN provider         anthropic, ANTHROPIC_API_KEY not set" in result.output


def test_reports_opencode_not_installed(tmp_path: Path) -> None:
    _init(tmp_path)

    result = runner.invoke(app, ["doctor", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "WARN opencode" in result.output
    assert "cairn install opencode" in result.output


def test_reports_opencode_installed_after_install(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _init(tmp_path)
    install_result = runner.invoke(app, ["install", "opencode", str(tmp_path)])
    assert install_result.exit_code == 0, install_result.output

    # No `opencode` binary resolvable in this test environment: nothing to
    # warn about, so it's a plain PASS.
    def _missing_binary(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError()

    monkeypatch.setattr(cli.subprocess, "run", _missing_binary)

    result = runner.invoke(app, ["doctor", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "PASS opencode         plugin linked at" in result.output


def test_reports_opencode_version_warning_when_older_than_verified(
    tmp_path: Path,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    _init(tmp_path)
    install_result = runner.invoke(app, ["install", "opencode", str(tmp_path)])
    assert install_result.exit_code == 0, install_result.output

    def _fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="0.1.0\n", stderr="")

    monkeypatch.setattr(cli.subprocess, "run", _fake_run)

    result = runner.invoke(app, ["doctor", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "WARN opencode" in result.output
    assert "older than" in result.output


def test_reports_opencode_pass_when_at_or_above_verified_version(
    tmp_path: Path,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    _init(tmp_path)
    install_result = runner.invoke(app, ["install", "opencode", str(tmp_path)])
    assert install_result.exit_code == 0, install_result.output

    def _fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="99.0.0\n", stderr="")

    monkeypatch.setattr(cli.subprocess, "run", _fake_run)

    result = runner.invoke(app, ["doctor", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "PASS opencode         plugin linked at" in result.output

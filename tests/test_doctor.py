"""Tests for `cairn doctor`'s non-hook rows.

Executing the Claude Code hooks is covered by tests/test_doctor_executes.py;
this file covers the store, git, provider, review-gate, opencode and
agents-md rows, and the exit code.

Assertions run over whitespace-collapsed output: the words are the contract,
the column padding is not.
"""

import re
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

import cairn.cli as cli
from cairn.cli import app

runner = CliRunner()


def _norm(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text)


def _init(tmp_path: Path) -> None:
    result = runner.invoke(app, ["init", str(tmp_path)])
    assert result.exit_code == 0, result.output


def _doctor(tmp_path: Path) -> tuple[int, str]:
    result = runner.invoke(app, ["doctor", str(tmp_path)])
    return result.exit_code, _norm(result.output)


def _fake_binary(**versions: str | None) -> object:
    """A `subprocess.run` stand-in keyed by binary name.

    Needed because doctor probes several binaries by running them, so a blunt
    monkeypatch of `subprocess.run` would break the `git` row while trying to
    fake the `opencode` one. A name mapped to None is treated as absent.
    """

    real = subprocess.run

    def _run(args, *pos, **kwargs):  # type: ignore[no-untyped-def]
        name = str(args[0])
        for binary, version in versions.items():
            if name == binary or name.endswith(f"{binary}.exe"):
                if version is None:
                    raise FileNotFoundError(name)
                return subprocess.CompletedProcess(
                    args=args, returncode=0, stdout=version, stderr=""
                )
        return real(args, *pos, **kwargs)

    return _run


# -- store -----------------------------------------------------------------------


def test_missing_store_fails_and_exits_non_zero(tmp_path: Path) -> None:
    exit_code, output = _doctor(tmp_path)

    assert "FAIL store" in output
    assert exit_code == 1, "a missing store is a failure, not a warning"


def test_healthy_store_and_mock_provider(tmp_path: Path) -> None:
    _init(tmp_path)

    exit_code, output = _doctor(tmp_path)

    assert exit_code == 0, output
    assert "PASS store" in output
    assert "0 entries, 0 staged" in output
    assert "PASS provider mock" in output


# -- git -------------------------------------------------------------------------


def test_git_row_reports_the_version_it_actually_ran(tmp_path: Path) -> None:
    _init(tmp_path)

    _, output = _doctor(tmp_path)

    # Probed by running it, so the row carries real `git --version` output.
    assert "git version" in output


def test_unrunnable_git_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _init(tmp_path)
    monkeypatch.setattr(cli.subprocess, "run", _fake_binary(git=None))

    exit_code, output = _doctor(tmp_path)

    assert exit_code == 1
    assert "FAIL git" in output


# -- provider --------------------------------------------------------------------


def test_anthropic_provider_without_a_key_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _init(tmp_path)
    config_path = tmp_path / ".cairn" / "config.toml"
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace('name  = "mock"', 'name  = "anthropic"'),
        encoding="utf-8",
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    exit_code, output = _doctor(tmp_path)

    assert exit_code == 0, "a missing key is a warning, not a failure"
    assert "WARN provider anthropic, ANTHROPIC_API_KEY not set" in output


# -- review gate -----------------------------------------------------------------


def test_review_gate_enabled_by_init(tmp_path: Path) -> None:
    _init(tmp_path)

    exit_code, output = _doctor(tmp_path)

    assert exit_code == 0, output
    assert "PASS review human approval required" in output


def test_review_gate_disabled_fails(tmp_path: Path) -> None:
    _init(tmp_path)
    config_path = tmp_path / ".cairn" / "config.toml"
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "require_human_approval = true", "require_human_approval = false"
        ),
        encoding="utf-8",
    )

    exit_code, output = _doctor(tmp_path)

    assert exit_code == 1
    assert "FAIL review" in output
    assert "require_human_approval is false" in output
    assert "will refuse" in output


def test_review_gate_fails_when_config_absent(tmp_path: Path) -> None:
    _init(tmp_path)
    (tmp_path / ".cairn" / "config.toml").unlink()

    exit_code, output = _doctor(tmp_path)

    assert exit_code == 1
    assert "FAIL review" in output


# -- adapters --------------------------------------------------------------------


def test_claude_code_not_installed_warns(tmp_path: Path) -> None:
    _init(tmp_path)

    exit_code, output = _doctor(tmp_path)

    assert exit_code == 0, "not installed is a warning, not a failure"
    assert "WARN claude-code" in output
    assert "cairn install claude-code" in output


def test_opencode_not_installed_warns(tmp_path: Path) -> None:
    _init(tmp_path)

    exit_code, output = _doctor(tmp_path)

    assert exit_code == 0, output
    assert "WARN opencode" in output
    assert "cairn install opencode" in output


def test_opencode_without_the_binary_is_unverified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _init(tmp_path)
    assert runner.invoke(app, ["install", "opencode", str(tmp_path)]).exit_code == 0
    monkeypatch.setattr(cli.subprocess, "run", _fake_binary(opencode=None))

    exit_code, output = _doctor(tmp_path)

    assert exit_code == 0, "unverified is not a failure"
    assert "UNVERIFIED opencode" in output
    assert "capture unproven" in output


def test_opencode_older_than_verified_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _init(tmp_path)
    assert runner.invoke(app, ["install", "opencode", str(tmp_path)]).exit_code == 0
    monkeypatch.setattr(cli.subprocess, "run", _fake_binary(opencode="0.1.0\n"))

    exit_code, output = _doctor(tmp_path)

    assert exit_code == 0, output
    assert "WARN opencode" in output
    assert "older than" in output


def test_opencode_at_or_above_verified_is_still_unverified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A current binary proves the version, not that its session.idle hook
    fires. Doctor cannot exercise that from here, so it says so."""

    _init(tmp_path)
    assert runner.invoke(app, ["install", "opencode", str(tmp_path)]).exit_code == 0
    monkeypatch.setattr(cli.subprocess, "run", _fake_binary(opencode="99.0.0\n"))

    exit_code, output = _doctor(tmp_path)

    assert exit_code == 0, output
    assert "UNVERIFIED opencode" in output
    assert "only a real opencode session proves capture" in output


def test_agents_md_missing_warns(tmp_path: Path) -> None:
    _init(tmp_path)

    exit_code, output = _doctor(tmp_path)

    assert exit_code == 0, output
    assert "WARN agents-md" in output

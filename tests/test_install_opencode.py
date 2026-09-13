"""Tests for `cairn install opencode`."""

import hashlib
from pathlib import Path

from typer.testing import CliRunner

from cairn.cli import app

runner = CliRunner()


def _init(tmp_path: Path) -> None:
    result = runner.invoke(app, ["init", str(tmp_path)])
    assert result.exit_code == 0, result.output


def _install(tmp_path: Path) -> tuple[int, str]:
    result = runner.invoke(app, ["install", "opencode", str(tmp_path)])
    return result.exit_code, result.output


def test_requires_cairn_init_first(tmp_path: Path) -> None:
    exit_code, output = _install(tmp_path)

    assert exit_code == 1
    assert "cairn init" in output


def test_writes_plugin_file(tmp_path: Path) -> None:
    _init(tmp_path)

    exit_code, output = _install(tmp_path)

    assert exit_code == 0, output
    plugin = tmp_path / ".opencode" / "plugins" / "cairn.ts"
    assert plugin.is_file()
    content = plugin.read_text(encoding="utf-8")
    assert "session.idle" in content
    assert "CairnPlugin" in content


def test_running_twice_is_idempotent(tmp_path: Path) -> None:
    _init(tmp_path)

    first_exit, first_output = _install(tmp_path)
    assert first_exit == 0, first_output
    plugin_path = tmp_path / ".opencode" / "plugins" / "cairn.ts"
    first_hash = hashlib.sha256(plugin_path.read_bytes()).hexdigest()

    second_exit, second_output = _install(tmp_path)
    assert second_exit == 0, second_output
    second_hash = hashlib.sha256(plugin_path.read_bytes()).hexdigest()

    assert first_hash == second_hash


def test_does_not_touch_unrelated_opencode_files(tmp_path: Path) -> None:
    _init(tmp_path)
    opencode_dir = tmp_path / ".opencode"
    opencode_dir.mkdir()
    (opencode_dir / "opencode.json").write_text('{"some": "config"}', encoding="utf-8")

    exit_code, output = _install(tmp_path)

    assert exit_code == 0, output
    assert (opencode_dir / "opencode.json").read_text(encoding="utf-8") == '{"some": "config"}'
    assert (opencode_dir / "plugins" / "cairn.ts").is_file()

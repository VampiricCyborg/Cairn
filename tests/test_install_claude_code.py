"""Tests for `cairn install claude-code`."""

import hashlib
import json
import os
from pathlib import Path

from typer.testing import CliRunner

from cairn.cli import app

runner = CliRunner()


def _init(tmp_path: Path) -> None:
    result = runner.invoke(app, ["init", str(tmp_path)])
    assert result.exit_code == 0, result.output


def _install(tmp_path: Path) -> tuple[int, str]:
    result = runner.invoke(app, ["install", "claude-code", str(tmp_path)])
    return result.exit_code, result.output


def _settings(tmp_path: Path) -> dict[str, object]:
    return json.loads((tmp_path / ".claude" / "settings.json").read_text(encoding="utf-8"))


def test_requires_cairn_init_first(tmp_path: Path) -> None:
    exit_code, output = _install(tmp_path)

    assert exit_code == 1
    assert "cairn init" in output


def test_creates_settings_with_both_hooks(tmp_path: Path) -> None:
    _init(tmp_path)

    exit_code, output = _install(tmp_path)

    assert exit_code == 0, output
    settings = _settings(tmp_path)
    session_end = settings["hooks"]["SessionEnd"][0]["hooks"][0]
    assert session_end["command"].endswith(".cairn/hooks/enqueue.sh")
    session_start = settings["hooks"]["SessionStart"][0]["hooks"][0]
    assert session_start["command"] == "cairn"
    assert "--hook" in session_start["args"]


def test_writes_executable_enqueue_script(tmp_path: Path) -> None:
    _init(tmp_path)

    _install(tmp_path)

    enqueue = tmp_path / ".cairn" / "hooks" / "enqueue.sh"
    assert enqueue.is_file()
    assert "enqueue.sh" in enqueue.read_text(encoding="utf-8")
    if os.name != "nt":
        assert enqueue.stat().st_mode & 0o111


def test_writes_skill_file(tmp_path: Path) -> None:
    _init(tmp_path)

    _install(tmp_path)

    skill = tmp_path / ".claude" / "skills" / "cairn" / "SKILL.md"
    assert skill.is_file()
    assert "description:" in skill.read_text(encoding="utf-8")


def test_merges_into_existing_unrelated_hooks_config(tmp_path: Path) -> None:
    _init(tmp_path)
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    existing = {
        "permissions": {"allow": ["Bash(git status)"]},
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "Bash",
                    "hooks": [{"type": "command", "command": "echo pre-tool-use", "timeout": 5}],
                }
            ],
            "SessionStart": [
                {"hooks": [{"type": "command", "command": "echo some-other-tool-hook"}]}
            ],
        },
    }
    (claude_dir / "settings.json").write_text(json.dumps(existing, indent=2), encoding="utf-8")

    exit_code, output = _install(tmp_path)

    assert exit_code == 0, output
    settings = _settings(tmp_path)

    # Unrelated top-level settings and unrelated event untouched.
    assert settings["permissions"] == {"allow": ["Bash(git status)"]}
    assert settings["hooks"]["PreToolUse"] == existing["hooks"]["PreToolUse"]

    # The other tool's SessionStart entry is preserved alongside cairn's own.
    session_start_commands = [
        hook["command"] for group in settings["hooks"]["SessionStart"] for hook in group["hooks"]
    ]
    assert "echo some-other-tool-hook" in session_start_commands
    assert "cairn" in session_start_commands

    # SessionEnd (absent before) was added.
    assert settings["hooks"]["SessionEnd"][0]["hooks"][0]["command"].endswith(
        ".cairn/hooks/enqueue.sh"
    )


def test_running_twice_is_idempotent(tmp_path: Path) -> None:
    _init(tmp_path)

    first_exit, first_output = _install(tmp_path)
    assert first_exit == 0, first_output
    settings_path = tmp_path / ".claude" / "settings.json"
    first_bytes = settings_path.read_bytes()
    first_hash = hashlib.sha256(first_bytes).hexdigest()

    second_exit, second_output = _install(tmp_path)
    assert second_exit == 0, second_output
    second_bytes = settings_path.read_bytes()
    second_hash = hashlib.sha256(second_bytes).hexdigest()

    assert second_hash == first_hash
    assert second_bytes == first_bytes

    settings = json.loads(second_bytes)
    assert len(settings["hooks"]["SessionEnd"]) == 1
    assert len(settings["hooks"]["SessionEnd"][0]["hooks"]) == 1
    assert len(settings["hooks"]["SessionStart"]) == 1
    assert len(settings["hooks"]["SessionStart"][0]["hooks"]) == 1


def test_running_twice_preserves_other_tools_hooks_added_between_runs(tmp_path: Path) -> None:
    _init(tmp_path)
    _install(tmp_path)

    settings_path = tmp_path / ".claude" / "settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    settings["hooks"].setdefault("PreToolUse", []).append(
        {"matcher": "Bash", "hooks": [{"type": "command", "command": "echo added-later"}]}
    )
    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")

    exit_code, output = _install(tmp_path)

    assert exit_code == 0, output
    final = _settings(tmp_path)
    pre_tool_use_commands = [
        hook["command"] for group in final["hooks"]["PreToolUse"] for hook in group["hooks"]
    ]
    assert "echo added-later" in pre_tool_use_commands
    # Still exactly one Cairn SessionEnd hook, not duplicated.
    session_end_hooks = [hook for group in final["hooks"]["SessionEnd"] for hook in group["hooks"]]
    assert len(session_end_hooks) == 1


def test_errors_cleanly_on_invalid_existing_settings_json(tmp_path: Path) -> None:
    _init(tmp_path)
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    (claude_dir / "settings.json").write_text("{not valid json", encoding="utf-8")

    exit_code, output = _install(tmp_path)

    assert exit_code == 1
    assert "not valid JSON" in output

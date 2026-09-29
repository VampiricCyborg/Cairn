"""`cairn doctor` must execute what it reports on, not read it.

The bug these pin: doctor reported `PASS claude-code  SessionEnd +
SessionStart hooks registered` on a machine where neither hook could run --
the SessionEnd command was a .sh file Windows cannot spawn, and the
SessionStart command was a bare `cairn` that was not on PATH. Both failures
were silent, and doctor's green light was the only thing the user saw.

Rule under test: no check may report PASS on the basis of a file's contents
when the thing it describes can be executed instead, and a check that
genuinely cannot execute reports UNVERIFIED rather than PASS.
"""

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cairn.cli import app

runner = CliRunner()


def _init(tmp_path: Path) -> None:
    result = runner.invoke(app, ["init", str(tmp_path)])
    assert result.exit_code == 0, result.output


def _install(tmp_path: Path) -> None:
    result = runner.invoke(app, ["install", "claude-code", str(tmp_path)])
    assert result.exit_code == 0, result.output


def _norm(text: str) -> str:
    """Collapse column padding: the status/name/detail columns are layout, the
    words are the contract."""

    return re.sub(r"[ 	]+", " ", text)


def _doctor(tmp_path: Path) -> tuple[int, str]:
    result = runner.invoke(app, ["doctor", str(tmp_path)])
    return result.exit_code, _norm(result.output)


def _settings_path(tmp_path: Path) -> Path:
    return tmp_path / ".claude" / "settings.json"


def _load_settings(tmp_path: Path) -> dict:
    return json.loads(_settings_path(tmp_path).read_text(encoding="utf-8"))


def _save_settings(tmp_path: Path, settings: dict) -> None:
    _settings_path(tmp_path).write_text(json.dumps(settings, indent=2), encoding="utf-8")


def _session_end_hook(settings: dict) -> dict:
    return settings["hooks"]["SessionEnd"][0]["hooks"][0]


def _session_start_hook(settings: dict) -> dict:
    return settings["hooks"]["SessionStart"][0]["hooks"][0]


# -- the healthy case ------------------------------------------------------------


def test_healthy_install_passes_and_exits_zero(tmp_path: Path) -> None:
    _init(tmp_path)
    _install(tmp_path)

    exit_code, output = _doctor(tmp_path)

    assert exit_code == 0, output
    assert "PASS claude-code" in output
    # It must say it ran them, not that they are registered.
    assert "SessionEnd wrote a job" in output
    assert "SessionStart returned" in output


def test_executing_the_hooks_does_not_touch_the_real_store(tmp_path: Path) -> None:
    """Probing SessionEnd must not leave a job in the user's queue, and
    probing SessionStart must not drain one: `cairn context --hook` sweeps."""

    _init(tmp_path)
    _install(tmp_path)
    queue = tmp_path / ".cairn" / "queue"
    queue.mkdir(parents=True, exist_ok=True)
    job = queue / "real-job.json"
    job.write_text(json.dumps({"session_id": "real-job", "transcript_path": "x"}), "utf-8")

    exit_code, output = _doctor(tmp_path)

    assert exit_code == 0, output
    assert job.exists(), "doctor drained a real queued job"
    assert sorted(p.name for p in queue.glob("*.json")) == ["real-job.json"]


# -- the failures doctor used to miss --------------------------------------------


def test_a_dot_sh_hook_fails_doctor(tmp_path: Path) -> None:
    """The original bug: a .sh command cannot be spawned on Windows. Doctor
    must run it, fail, and surface the spawn error."""

    _init(tmp_path)
    _install(tmp_path)
    settings = _load_settings(tmp_path)
    script = tmp_path / ".cairn" / "hooks" / "enqueue.sh"
    script.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    hook = _session_end_hook(settings)
    hook["command"] = str(script)
    hook["args"] = []
    _save_settings(tmp_path, settings)

    exit_code, output = _doctor(tmp_path)

    assert exit_code != 0, output
    assert "FAIL claude-code" in output


def test_a_missing_interpreter_fails_doctor(tmp_path: Path) -> None:
    _init(tmp_path)
    _install(tmp_path)
    settings = _load_settings(tmp_path)
    _session_end_hook(settings)["command"] = str(tmp_path / "no-such-python.exe")
    _save_settings(tmp_path, settings)

    exit_code, output = _doctor(tmp_path)

    assert exit_code != 0, output
    assert "FAIL claude-code" in output
    assert "no-such-python" in output


def test_a_hook_that_runs_but_writes_no_job_fails_doctor(tmp_path: Path) -> None:
    """Exit code 0 is not evidence. The capture hook exits 0 on every path by
    design, so doctor must check the job file, not the return code."""

    _init(tmp_path)
    _install(tmp_path)
    # Keep the registration intact and gut the script it points at, so the
    # hook is still recognisably Cairn's and simply does not do its job.
    (tmp_path / ".cairn" / "hooks" / "enqueue.py").write_text(
        "import sys; sys.stdin.read()", encoding="utf-8"
    )

    exit_code, output = _doctor(tmp_path)

    assert exit_code != 0, output
    assert "FAIL claude-code" in output
    assert "no job" in output.lower()


def test_a_session_start_hook_with_bad_output_fails_doctor(tmp_path: Path) -> None:
    _init(tmp_path)
    _install(tmp_path)
    settings = _load_settings(tmp_path)
    hook = _session_start_hook(settings)
    # The trailing cairn/context/--hook arguments keep this recognisable as
    # Cairn's SessionStart hook; `python -c` ignores them beyond sys.argv.
    hook["args"] = ["-c", "print('not json')", "cairn", "context", "--hook"]
    _save_settings(tmp_path, settings)

    exit_code, output = _doctor(tmp_path)

    assert exit_code != 0, output
    assert "FAIL claude-code" in output


def test_an_unspawnable_session_start_command_fails_doctor(tmp_path: Path) -> None:
    """The retired shape ran a bare `cairn`, which only resolves after a tool
    install. Doctor must spawn whatever is registered and fail when it cannot,
    rather than passing because the registration looks right."""

    _init(tmp_path)
    _install(tmp_path)
    settings = _load_settings(tmp_path)
    hook = _session_start_hook(settings)
    hook["command"] = "definitely-not-a-real-binary-xyz"
    hook["args"] = ["-m", "cairn", "context", "--hook"]
    _save_settings(tmp_path, settings)

    exit_code, output = _doctor(tmp_path)

    assert exit_code != 0, output


# -- unverified rather than pass -------------------------------------------------


def test_uninstalled_adapter_is_not_a_pass(tmp_path: Path) -> None:
    _init(tmp_path)

    exit_code, output = _doctor(tmp_path)

    assert "PASS claude-code" not in output
    assert exit_code == 0, "not installed is a warning, not a failure"


def test_opencode_without_the_binary_is_unverified_not_pass(tmp_path: Path) -> None:
    _init(tmp_path)
    result = runner.invoke(app, ["install", "opencode", str(tmp_path)])
    assert result.exit_code == 0, result.output

    exit_code, output = _doctor(tmp_path)

    # The plugin file exists, but nothing proved opencode can load it.
    assert "PASS opencode" not in output
    assert "UNVERIFIED opencode" in output


def test_external_binaries_are_probed_by_running_them(tmp_path: Path) -> None:
    _init(tmp_path)

    _, output = _doctor(tmp_path)

    assert "git" in output


@pytest.mark.parametrize(
    "missing", ["entries/strategy", "entries/gotcha", "entries/fact", "rejected"]
)
def test_committed_store_layout_survives_a_clone(missing: str) -> None:
    """SPEC.md lists these as committed, but git cannot track an empty
    directory, so each needs a .gitkeep or a fresh clone loses the layout."""

    repo_store = Path(__file__).resolve().parent.parent / ".cairn"
    assert (repo_store / missing).is_dir(), f"{missing} missing from the committed store"
    assert (repo_store / missing / ".gitkeep").is_file(), f"{missing}/.gitkeep is not committed"


def test_a_clone_of_the_committed_store_validates(tmp_path: Path) -> None:
    """`git clone` of this repo must yield a store that passes
    `cairn validate --strict`, which is only true while the .gitkeep files
    keep the empty directories SPEC.md lists as committed."""

    import subprocess

    repo = Path(__file__).resolve().parent.parent
    clone = tmp_path / "clone"
    result = subprocess.run(
        ["git", "clone", "--depth", "1", "--no-hardlinks", str(repo), str(clone)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode != 0:
        pytest.skip(f"git clone unavailable here: {result.stderr.strip()[:120]}")

    store = clone / ".cairn"
    for relative in ("entries/strategy", "entries/gotcha", "entries/fact", "staging", "rejected"):
        assert (store / relative).is_dir(), f"{relative} did not survive the clone"

    validated = runner.invoke(app, ["validate", str(clone), "--strict"])
    assert validated.exit_code == 0, validated.output

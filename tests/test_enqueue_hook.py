"""Subprocess tests for cairn/adapters/claude_code/enqueue.sh.

These exercise the actual shell script through `bash`, not a Python
reimplementation of its logic: the requirement being tested is "this script
can never fail a SessionEnd hook," which is a property of the script's
control flow, not of anything Python-side.
"""

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="requires bash")

_SCRIPT = (
    Path(__file__).resolve().parent.parent / "cairn" / "adapters" / "claude_code" / "enqueue.sh"
)

_FAKE_JQ = """\
#!/usr/bin/env bash
# Minimal jq stand-in for tests: only supports the two invocation shapes
# enqueue.sh uses. Not a general jq replacement.
if [ "$1" = "-r" ]; then
  filter="$2"
  field="$(printf '%s' "$filter" | sed -n 's/^\\.\\([a-zA-Z_]*\\).*/\\1/p')"
  input="$(cat)"
  value="$(printf '%s' "$input" \\
    | sed -n "s/.*\\"$field\\"[[:space:]]*:[[:space:]]*\\"\\\\([^\\"]*\\\\)\\".*/\\\\1/p")"
  printf '%s\\n' "$value"
  exit 0
fi
if [ "$1" = "-nc" ]; then
  shift
  declare -A vals
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --arg) vals["$2"]="$3"; shift 3 ;;
      *) shift ;;
    esac
  done
  printf '{"session_id":"%s","transcript_path":"%s","harness":"%s","enqueued_at":"%s"}\\n' \\
    "${vals[s]}" "${vals[t]}" "${vals[h]}" "${vals[at]}"
  exit 0
fi
exit 1
"""


def _run(payload: str, *, path: str) -> subprocess.CompletedProcess[str]:
    """Run enqueue.sh with stdin `payload` and PATH overridden to `path`,
    otherwise inheriting the real environment -- replacing the whole
    environment (as `subprocess.run(env=...)` would) risks bash's own MSYS
    runtime failing to start on Windows for reasons unrelated to what these
    tests are actually checking.
    """

    bash = shutil.which("bash")
    assert bash is not None
    env = {**os.environ, "PATH": path}
    return subprocess.run(
        [bash, str(_SCRIPT)],
        input=payload,
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )


def _write_fake_jq(bin_dir: Path) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    jq_path = bin_dir / "jq"
    jq_path.write_text(_FAKE_JQ, encoding="utf-8")
    jq_path.chmod(jq_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def test_exits_zero_when_jq_is_missing() -> None:
    """No `jq` resolvable anywhere on PATH: the very first guard must catch
    this and exit 0 without needing any other external tool (stdin is read
    with a bash builtin, not external `cat`)."""

    result = _run(
        json.dumps({"session_id": "abc", "transcript_path": "/tmp/t.jsonl", "cwd": "/tmp"}),
        path="",
    )

    assert result.returncode == 0
    assert "jq not found" in result.stderr


def test_exits_zero_on_malformed_stdin(tmp_path: Path) -> None:
    """`jq` is present, but stdin is not valid JSON: session_id/cwd extract
    as empty, and the script must still exit 0 rather than propagate a
    parse failure."""

    fake_bin = tmp_path / "fakebin"
    _write_fake_jq(fake_bin)

    result = _run("this is not json at all", path=str(fake_bin))

    assert result.returncode == 0
    assert "malformed or incomplete event JSON" in result.stderr


def test_exits_zero_on_empty_stdin(tmp_path: Path) -> None:
    fake_bin = tmp_path / "fakebin"
    _write_fake_jq(fake_bin)

    result = _run("", path=str(fake_bin))

    assert result.returncode == 0


def test_happy_path_writes_queue_job(tmp_path: Path) -> None:
    """With a working `jq` and a well-formed event, the script writes the
    job record to `<cwd>/.cairn/queue/<session_id>.json` and still exits 0."""

    fake_bin = tmp_path / "fakebin"
    _write_fake_jq(fake_bin)
    project = tmp_path / "project"
    project.mkdir()

    payload = json.dumps(
        {
            "session_id": "abc123",
            "transcript_path": "/home/user/.claude/projects/x/abc123.jsonl",
            "cwd": str(project),
        }
    )
    # Fake jq shadows any real one; mkdir/date/mv/rm still resolve from the
    # real PATH, since this scenario (unlike the two failure tests above)
    # needs the script to actually reach and complete those steps.
    result = _run(payload, path=f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}")

    assert result.returncode == 0
    job_path = project / ".cairn" / "queue" / "abc123.json"
    assert job_path.is_file()
    job = json.loads(job_path.read_text(encoding="utf-8"))
    assert job["session_id"] == "abc123"
    assert job["harness"] == "claude-code"
    assert job["transcript_path"] == "/home/user/.claude/projects/x/abc123.jsonl"
    assert "enqueued_at" in job

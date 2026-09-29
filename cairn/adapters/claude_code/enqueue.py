#!/usr/bin/env python3
"""Claude Code SessionEnd capture hook: enqueue a Cairn job and get out of the way.

Deliberately stdlib-only, and deliberately does not import `cairn`. Python is
already a hard prerequisite of this project; `bash` and `jq` are not, and the
previous shell version of this hook needed both. On Windows it needed neither
to be *missing* to fail: Claude Code spawns a hook command directly (libuv),
and Windows cannot execute a `.sh` file at all, so the hook died with
`EFTYPE: inappropriate file type or format` before any of its logic ran.

Not importing `cairn` is a performance decision as much as a dependency one:
the whole job is to append one small JSON file inside Claude Code's SessionEnd
budget, and importing the package would drag in pydantic, typer and rich for a
task that needs `json` and `os`.

The contract this must never break: **exit 0 on every path**. A broken Cairn
install must not block, slow, or crash a coding session, so every failure --
unreadable stdin, malformed payload, unwritable directory -- is reported on
stderr and still exits 0.

Two observability surfaces, both append-only and both gitignored:

- `capture-log.jsonl` is written on every run, always. It is the numerator for
  the capture rate; the denominator is the number of session transcripts
  Claude Code wrote, which is why the rate can be computed even for sessions
  where this hook never ran at all. `cairn stats --capture` reads it.
- `hook-trace.jsonl` is written only under `CAIRN_HOOK_DEBUG=1`, at three
  checkpoints -- spawned, stdin read, job written -- so a silent failure can
  be localized to process startup, stdin, or the write rather than guessed at.
  The first checkpoint is the first statement this module executes, so a run
  that leaves no line at all never got far enough to run Python.
"""

import json
import os
import sys
import time

HARNESS = "claude-code"
TRACE_ENV = "CAIRN_HOOK_DEBUG"
TRACE_PATH_ENV = "CAIRN_HOOK_DEBUG_LOG"
#: Optional marker recorded in the capture log so headless runs can be told
#: apart from interactive ones later. Nothing sets it automatically: the
#: SessionEnd payload carries no mode, so an unset value means "unrecorded"
#: rather than "interactive".
MODE_ENV = "CAIRN_CAPTURE_MODE"


def _iso_now() -> str:
    """UTC timestamp, `time` rather than `datetime` to keep the import cost down."""

    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _append_jsonl(path: str, record: dict) -> bool:
    """Append one JSON line. Never raises; returns whether it was written."""

    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    except Exception:  # noqa: BLE001 - observability must never fail the hook
        return False
    return True


def _trace_path() -> str:
    """Where `CAIRN_HOOK_DEBUG=1` traces go.

    Defaults beside the `.cairn/hooks/` directory this script was installed
    into, which is known at checkpoint 1. The payload's `cwd` is not: stdin
    has not been read yet, and that is exactly the point of checkpoint 1.
    """

    override = os.environ.get(TRACE_PATH_ENV)
    if override:
        return override
    hooks_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(hooks_dir), "hook-trace.jsonl")


def _trace(checkpoint: str, **fields: object) -> None:
    if os.environ.get(TRACE_ENV) != "1":
        return
    record: dict = {
        "checkpoint": checkpoint,
        "ts": _iso_now(),
        "monotonic": round(time.monotonic(), 6),
        "pid": os.getpid(),
        "ppid": os.getppid(),
    }
    record.update(fields)
    _append_jsonl(_trace_path(), record)


# Checkpoint 1, and the first statement this module executes. A session that
# leaves no `spawned` line never reached Python at all; one that leaves
# `spawned` and no `stdin_read` died waiting on stdin.
_trace("spawned", argv=sys.argv, cwd=os.getcwd())


def _warn(message: str) -> None:
    try:
        print(f"cairn enqueue.py: {message}", file=sys.stderr)
    except Exception:  # noqa: BLE001 - even a broken stderr must not fail the hook
        pass


def _log_capture(
    cwd: str, timestamp: str, session_id: str, reason: str, enqueued: bool, error: str | None
) -> None:
    """Append the permanent capture-rate record. Always written, never gated."""

    _append_jsonl(
        os.path.join(cwd, ".cairn", "capture-log.jsonl"),
        {
            "ts": timestamp,
            "session_id": session_id,
            "reason": reason,
            "enqueued": enqueued,
            "error": error,
            "mode": os.environ.get(MODE_ENV) or None,
            "pid": os.getpid(),
        },
    )


def enqueue(payload: dict, *, now: str | None = None) -> str | None:
    """Write the queue record for `payload`. Returns the path written, or None.

    Split out from `main` so tests can drive it directly, and so the
    exit-0-always guarantee lives in exactly one place (`main`).
    """

    session_id = str(payload.get("session_id") or "")
    cwd = str(payload.get("cwd") or "")
    reason = str(payload.get("reason") or "")
    timestamp = now or _iso_now()

    if not session_id or not cwd:
        _warn("malformed or incomplete event JSON; skipping capture")
        return None

    # A session id lands in a filename; anything path-like in it would let a
    # malformed payload write outside the queue directory.
    if os.path.basename(session_id) != session_id or session_id in (".", ".."):
        _warn(f"refusing to use {session_id!r} as a filename; skipping capture")
        _log_capture(cwd, timestamp, session_id, reason, False, "unsafe session_id")
        return None

    queue_dir = os.path.join(cwd, ".cairn", "queue")
    try:
        os.makedirs(queue_dir, exist_ok=True)
    except OSError as exc:
        _warn(f"could not create {queue_dir} ({exc}); skipping capture")
        _log_capture(cwd, timestamp, session_id, reason, False, str(exc))
        return None

    record = {
        "session_id": session_id,
        "transcript_path": str(payload.get("transcript_path") or ""),
        "harness": HARNESS,
        "enqueued_at": timestamp,
        # Free provenance: why the session ended. Carried so the sweep, and any
        # later analysis, can tell a normal exit from a /clear or a crash.
        "reason": reason,
    }

    target = os.path.join(queue_dir, f"{session_id}.json")
    tmp = os.path.join(queue_dir, f".{session_id}.json.tmp.{os.getpid()}")
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(record, handle)
            handle.write("\n")
        # Atomic on both POSIX and Windows: the sweep never sees a half-written job.
        os.replace(tmp, target)
    except OSError as exc:
        _warn(f"could not write {target} ({exc}); skipping capture")
        try:
            os.unlink(tmp)
        except OSError:
            pass
        _log_capture(cwd, timestamp, session_id, reason, False, str(exc))
        return None

    # Checkpoint 3.
    _trace("written", session_id=session_id, target=target)
    _log_capture(cwd, timestamp, session_id, reason, True, None)
    return target


def main() -> int:
    try:
        raw = sys.stdin.read()
    except Exception as exc:  # noqa: BLE001 - see module docstring
        _warn(f"could not read stdin ({exc}); skipping capture")
        return 0

    # Checkpoint 2: stdin came back. A run that stops here died in the write.
    _trace("stdin_read", bytes=len(raw))

    try:
        payload = json.loads(raw) if raw.strip() else {}
    except ValueError as exc:
        _warn(f"stdin was not valid JSON ({exc}); skipping capture")
        return 0

    try:
        if isinstance(payload, dict):
            enqueue(payload)
        else:
            _warn("event JSON was not an object; skipping capture")
    except Exception as exc:  # noqa: BLE001 - see module docstring
        _warn(f"unexpected error ({exc}); skipping capture")
    return 0


if __name__ == "__main__":
    sys.exit(main())

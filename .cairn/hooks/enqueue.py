#!/usr/bin/env python3
"""Claude Code SessionEnd capture hook: enqueue a Cairn job and get out of the way.

Deliberately stdlib-only, and deliberately does not import `cairn`. Python is
already a hard prerequisite of this project; `bash` and `jq` are not, and the
previous shell version of this hook needed both. On Windows it needed neither
to be *missing* to fail: Claude Code spawns a hook command directly (libuv),
and Windows cannot execute a `.sh` file at all, so the hook died with
`EFTYPE: inappropriate file type or format` before any of its logic ran.
Registering this file under an absolute interpreter path removes the shell,
the `jq` dependency, and the PATH lookup in one step.

Not importing `cairn` is a performance decision as much as a dependency one:
the whole job is to append one small JSON file inside Claude Code's SessionEnd
budget, and importing the package would drag in pydantic, typer and rich for
a task that needs `json` and `os`.

The contract this must never break: **exit 0 on every path**. A broken Cairn
install must not block, slow, or crash a coding session, so every failure --
unreadable stdin, malformed payload, unwritable directory -- is reported on
stderr and still exits 0.
"""

import json
import os
import sys
import time

HARNESS = "claude-code"


def _iso_now() -> str:
    """UTC timestamp, `time` rather than `datetime` to keep the import cost down."""

    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _warn(message: str) -> None:
    try:
        print(f"cairn enqueue.py: {message}", file=sys.stderr)
    except Exception:  # noqa: BLE001 - even a broken stderr must not fail the hook
        pass


def enqueue(payload: dict[str, object], *, now: str | None = None) -> str | None:
    """Write the queue record for `payload`. Returns the path written, or None.

    Split out from `main` so tests can drive it directly, and so the
    exit-0-always guarantee lives in exactly one place (`main`).
    """

    session_id = str(payload.get("session_id") or "")
    cwd = str(payload.get("cwd") or "")
    if not session_id or not cwd:
        _warn("malformed or incomplete event JSON; skipping capture")
        return None

    # A session id lands in a filename; anything path-like in it would let a
    # malformed payload write outside the queue directory.
    if os.path.basename(session_id) != session_id or session_id in (".", ".."):
        _warn(f"refusing to use {session_id!r} as a filename; skipping capture")
        return None

    queue_dir = os.path.join(cwd, ".cairn", "queue")
    try:
        os.makedirs(queue_dir, exist_ok=True)
    except OSError as exc:
        _warn(f"could not create {queue_dir} ({exc}); skipping capture")
        return None

    record = {
        "session_id": session_id,
        "transcript_path": str(payload.get("transcript_path") or ""),
        "harness": HARNESS,
        "enqueued_at": now or _iso_now(),
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
        return None
    return target


def main() -> int:
    try:
        raw = sys.stdin.read()
    except Exception as exc:  # noqa: BLE001 - see module docstring
        _warn(f"could not read stdin ({exc}); skipping capture")
        return 0

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

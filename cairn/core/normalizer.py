"""Harness transcripts to canonical `SessionTrace`.

`normalize()` is the harness dispatch every caller (the CLI's SessionStart
queue sweep, `cairn reflect`) should go through; the per-harness functions
below it are the actual parsers, each deliberately narrow to the one shape
its harness's capture adapter hands off. See `cairn.core.models.SessionTrace`
for the canonical shape every normalizer converges on.

Claude Code writes one JSON object per line to the path a
`SessionStart`/`SessionEnd` hook receives as `transcript_path` -- `type`
`"user"` or `"assistant"`, a nested `message` whose `content` is either a
plain string or a list of content blocks (`text`, `tool_use`,
`tool_result`). Its transcript never bundles a tool call and its result on
one record the way `SessionTrace.Turn` does (`tool_calls` and
`tool_results` together) -- the call comes in an `assistant` record and its
result in the *next* `user` record. `normalize_claude_code_transcript`'s
main job is reassembling that pair back onto a single `Turn`, keyed by
`tool_use_id`.

opencode has no equivalent flat transcript file (see
`cairn/adapters/opencode/plugin.ts`'s module docstring for why): its
capture adapter instead writes the verbatim JSON response of the
`@opencode-ai/sdk` call `client.session.messages({ path: { id } })` --
confirmed against opencode's published SDK types (checked 2026-09-14,
package version 1.18.30) -- an array of `{info: Message, parts: Part[]}`
records. `normalize_opencode_transcript` parses that shape directly; unlike
Claude Code it needs no call/result reassembly, since a tool call and its
result already live on the same `ToolPart` via `state.status`.
"""

import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from cairn.core.models import Diff, SessionTrace, ToolCall, ToolResult, Turn, TurnRole

logger = logging.getLogger(__name__)

#: Cap on how much of an Edit/Write tool call's content is kept in the
#: synthetic `Diff.patch` this module derives. Not a real unified diff --
#: Claude Code's transcript doesn't carry one -- just enough before/after
#: text to be useful evidence, without inflating the trace on a large file.
_DIFF_PREVIEW_CHARS = 2000


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _text_from_content(content: object) -> str:
    """Plain string content, or the concatenated `text` blocks of a content
    list (a `tool_use`/`tool_result` block contributes no text here)."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block["text"]
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ]
        return "\n".join(parts)
    return ""


def _blocks_of_type(content: object, block_type: str) -> list[dict[str, object]]:
    if not isinstance(content, list):
        return []
    return [
        block for block in content if isinstance(block, dict) and block.get("type") == block_type
    ]


def _diff_for_tool_use(block: dict[str, object]) -> Diff | None:
    """A best-effort `Diff` for a file-editing tool call: not a real unified
    diff, just enough of the before/after to be useful evidence."""

    name = block.get("name")
    tool_input = block.get("input")
    if not isinstance(tool_input, dict):
        return None
    file_path = tool_input.get("file_path")
    if not isinstance(file_path, str):
        return None

    if name == "Edit":
        old, new = tool_input.get("old_string"), tool_input.get("new_string")
        if isinstance(old, str) and isinstance(new, str):
            patch = f"- {old}\n+ {new}"
            return Diff(file=file_path, patch=patch[:_DIFF_PREVIEW_CHARS])
    elif name == "Write":
        content = tool_input.get("content")
        if isinstance(content, str):
            return Diff(file=file_path, patch=f"+ {content[:_DIFF_PREVIEW_CHARS]}")
    return None


def _stringify_result_content(content: object) -> object:
    if isinstance(content, list):
        return _text_from_content(content) or content
    return content


def _assistant_turn(content: object, pending_tool_names: dict[str, str], diffs: list[Diff]) -> Turn:
    tool_calls: list[ToolCall] = []
    for block in _blocks_of_type(content, "tool_use"):
        name = block.get("name")
        if not isinstance(name, str):
            continue
        block_id = block.get("id")
        if isinstance(block_id, str):
            pending_tool_names[block_id] = name
        tool_input = block.get("input")
        tool_calls.append(
            ToolCall(name=name, input=tool_input if isinstance(tool_input, dict) else {})
        )
        diff = _diff_for_tool_use(block)
        if diff is not None:
            diffs.append(diff)

    return Turn(
        role=TurnRole.ASSISTANT,
        content=_text_from_content(content),
        tool_calls=tool_calls,
        tool_results=[],
    )


def _tool_results(content: object, pending_tool_names: dict[str, str]) -> list[ToolResult]:
    results = []
    for block in _blocks_of_type(content, "tool_result"):
        tool_use_id = block.get("tool_use_id")
        name = (
            pending_tool_names.get(tool_use_id, "tool") if isinstance(tool_use_id, str) else "tool"
        )
        results.append(
            ToolResult(
                name=name,
                output=_stringify_result_content(block.get("content")),
                is_error=bool(block.get("is_error", False)),
            )
        )
    return results


def _infer_outcome(turns: list[Turn]) -> str:
    """`"error"` if the last turn carrying a tool result has a failed one,
    `"completed"` otherwise (including when no turn has a tool result at
    all). A narrow heuristic, not a claim about task success."""

    for turn in reversed(turns):
        if turn.tool_results:
            return "error" if any(result.is_error for result in turn.tool_results) else "completed"
    return "completed"


def normalize_claude_code_transcript(path: Path, *, harness: str = "claude-code") -> SessionTrace:
    """Parse the JSONL transcript at `path` into a canonical `SessionTrace`.

    A malformed line is skipped with a warning, not fatal: one corrupted
    line should not prevent normalizing the rest of the session. Raises
    `OSError` if `path` itself cannot be read.
    """

    turns: list[Turn] = []
    diffs: list[Diff] = []
    timestamps: list[datetime] = []
    session_id: str | None = None
    pending_tool_names: dict[str, str] = {}

    with path.open(encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError:
                logger.warning("%s:%d: skipping malformed JSON line", path, line_no)
                continue
            if not isinstance(record, dict):
                continue

            record_type = record.get("type")
            message = record.get("message")
            if record_type not in ("user", "assistant") or not isinstance(message, dict):
                continue

            if session_id is None and isinstance(record.get("sessionId"), str):
                session_id = record["sessionId"]
            timestamp = _parse_timestamp(record.get("timestamp"))
            if timestamp is not None:
                timestamps.append(timestamp)

            content = message.get("content")

            if record_type == "assistant":
                turns.append(_assistant_turn(content, pending_tool_names, diffs))
                continue

            # record_type == "user": either a plain user turn, or the tool
            # results answering the immediately preceding assistant turn.
            results = _tool_results(content, pending_tool_names)
            if results and turns and turns[-1].role is TurnRole.ASSISTANT:
                turns[-1] = turns[-1].model_copy(
                    update={"tool_results": [*turns[-1].tool_results, *results]}
                )
                continue

            turns.append(Turn(role=TurnRole.USER, content=_text_from_content(content)))

    if timestamps:
        started_at, ended_at = min(timestamps), max(timestamps)
    else:
        started_at = ended_at = datetime.now(UTC)

    return SessionTrace(
        session_id=session_id or path.stem,
        harness=harness,
        started_at=started_at,
        ended_at=ended_at,
        turns=turns,
        errors=[],
        diffs=diffs,
        outcome=_infer_outcome(turns),
    )


def _opencode_turn(info: dict[str, object], parts: list[object]) -> Turn | None:
    """One `Turn` from an opencode `{info, parts}` record, or `None` if
    `info.role` isn't `"user"`/`"assistant"` (nothing else appears in
    `client.session.messages`' response per the confirmed `Message` union)."""

    role = info.get("role")
    if role not in ("user", "assistant"):
        return None

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    tool_results: list[ToolResult] = []

    for part in parts:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")

        if part_type == "text":
            text = part.get("text")
            if isinstance(text, str):
                text_parts.append(text)
            continue

        if part_type != "tool":
            continue
        name = part.get("tool")
        state = part.get("state")
        if not isinstance(name, str) or not isinstance(state, dict):
            continue
        tool_input = state.get("input")
        tool_calls.append(
            ToolCall(name=name, input=tool_input if isinstance(tool_input, dict) else {})
        )
        status = state.get("status")
        if status == "completed":
            tool_results.append(ToolResult(name=name, output=state.get("output"), is_error=False))
        elif status == "error":
            tool_results.append(ToolResult(name=name, output=state.get("error"), is_error=True))
        # "pending"/"running" have no result yet -- nothing to record.

    return Turn(
        role=TurnRole.USER if role == "user" else TurnRole.ASSISTANT,
        content="\n".join(text_parts),
        tool_calls=tool_calls,
        tool_results=tool_results,
    )


def normalize_opencode_transcript(path: Path, *, harness: str = "opencode") -> SessionTrace:
    """Parse the JSON array opencode's capture adapter writes -- the
    verbatim response of `client.session.messages({ path: { id } })`, an
    array of `{info: Message, parts: Part[]}` records -- into a canonical
    `SessionTrace`. See this module's docstring for how that shape was
    confirmed and why opencode has no flat transcript file to read instead.

    A malformed record is skipped, not fatal, for the same reason
    `normalize_claude_code_transcript` skips a malformed line. Raises
    `OSError` if `path` can't be read, or `ValueError` if its content
    isn't the expected JSON array.

    opencode has no unified-diff equivalent at this layer: a `patch` part
    (`{type: "patch", hash, files}`) names which files changed but carries
    no diff text, so `diffs` is always empty here rather than fabricated
    from a tool call's `input` -- see SPEC.md's evidence-backed criterion.
    Likewise, an `AssistantMessage.error` is preserved verbatim as its JSON
    string in `errors` rather than picked apart into fields this module
    hasn't confirmed.
    """

    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: not valid JSON") from exc
    if not isinstance(records, list):
        raise ValueError(f"{path}: expected a JSON array of {{info, parts}} records")

    turns: list[Turn] = []
    errors: list[str] = []
    timestamps_ms: list[float] = []
    session_id: str | None = None

    for record in records:
        if not isinstance(record, dict):
            continue
        info = record.get("info")
        parts = record.get("parts")
        if not isinstance(info, dict) or not isinstance(parts, list):
            continue

        if session_id is None and isinstance(info.get("sessionID"), str):
            session_id = info["sessionID"]

        time_field = info.get("time")
        created = time_field.get("created") if isinstance(time_field, dict) else None
        if isinstance(created, int | float):
            timestamps_ms.append(created)

        error = info.get("error")
        if isinstance(error, dict):
            errors.append(json.dumps(error, sort_keys=True))

        turn = _opencode_turn(info, parts)
        if turn is not None:
            turns.append(turn)

    if timestamps_ms:
        started_at = datetime.fromtimestamp(min(timestamps_ms) / 1000, tz=UTC)
        ended_at = datetime.fromtimestamp(max(timestamps_ms) / 1000, tz=UTC)
    else:
        started_at = ended_at = datetime.now(UTC)

    return SessionTrace(
        session_id=session_id or path.stem.removesuffix(".transcript"),
        harness=harness,
        started_at=started_at,
        ended_at=ended_at,
        turns=turns,
        errors=errors,
        diffs=[],
        outcome=_infer_outcome(turns),
    )


_NORMALIZERS: dict[str, Callable[[Path], SessionTrace]] = {
    "claude-code": normalize_claude_code_transcript,
    "opencode": normalize_opencode_transcript,
}


def normalize(transcript_path: Path, harness: str) -> SessionTrace:
    """Dispatch to the normalizer registered for `harness`.

    This is the entry point queue consumers (the CLI's SessionStart sweep,
    `cairn reflect`) should call rather than importing a harness-specific
    function directly, so adding a harness means adding one entry to
    `_NORMALIZERS`, not touching every caller.

    Raises `ValueError` for a harness with no normalizer registered; callers
    are expected to catch this the same way they catch a malformed job
    record or a missing transcript.
    """

    normalizer = _NORMALIZERS.get(harness)
    if normalizer is None:
        raise ValueError(f"no normalizer registered for harness {harness!r}")
    return normalizer(transcript_path)

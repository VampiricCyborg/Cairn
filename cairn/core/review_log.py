"""Append-only decision log at `.cairn/review-log.jsonl`.

Approval rate is the metric that decides whether Cairn's extraction is worth
anything (see README's Evaluation section), and until this log existed it was
not computable: approvals ended up in entry frontmatter, rejections in
`rejected/` tombstones, skips nowhere at all, and candidates the Curator
dropped were printed to stdout and lost. One append-only record per decision
makes all of it countable after the fact.

**Two populations, never blended.** A record is either a *human* decision
(`approve`, `approve_with_edit`, `merge`, `reject`, `skip` -- written by the
review flow at decision time) or an *extraction-side* event (`staged`,
`gate_drop` -- written by `cairn reflect` and the SessionStart sweep). They
answer different questions and have different denominators:

- **Approval rate** counts approvals over candidates *shown to a human*, i.e.
  over `HUMAN_ACTIONS` records only.
- **Gate-drop rate** counts candidates the deterministic gates killed before
  any human saw them, over everything extraction produced, i.e. over
  `EXTRACTION_ACTIONS` records only.

Mixing the two would make a well-tuned Curator look like a reviewer rejecting
things, so `cairn stats --review` reports them as separate blocks and every
consumer filters by action explicitly.

The log is gitignored. It is local telemetry about *this* machine's review
habits, not curated knowledge, so it is deliberately not part of the store
spec and carries no `spec_version`: nothing downstream may depend on its
shape. Writes never raise -- a log that cannot be written must not lose a
decision that already happened on disk.
"""

import json
import logging
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from cairn.core.models import Entry, EntryType

logger = logging.getLogger(__name__)

REVIEW_LOG_NAME = "review-log.jsonl"


class ReviewAction(StrEnum):
    """What happened to one candidate, and who decided it."""

    # Human decisions, written by the review flow.
    APPROVE = "approve"
    APPROVE_WITH_EDIT = "approve_with_edit"
    MERGE = "merge"
    REJECT = "reject"
    SKIP = "skip"

    # Extraction-side events, written by `cairn reflect` and the queue sweep.
    STAGED = "staged"
    GATE_DROP = "gate_drop"


#: Candidates a human actually saw and acted on: the denominator for approval rate.
HUMAN_ACTIONS = frozenset(
    {
        ReviewAction.APPROVE,
        ReviewAction.APPROVE_WITH_EDIT,
        ReviewAction.MERGE,
        ReviewAction.REJECT,
        ReviewAction.SKIP,
    }
)

#: Everything extraction produced: the denominator for gate-drop rate. Kept
#: strictly apart from `HUMAN_ACTIONS` -- a `staged` record says a candidate
#: survived the gates, not that anyone has looked at it.
EXTRACTION_ACTIONS = frozenset({ReviewAction.STAGED, ReviewAction.GATE_DROP})

#: The human decisions that count as accepting a candidate. `merge` is one:
#: the reviewer kept the candidate's evidence by folding it into an existing
#: entry, which is the same judgement as approving it (and is what
#: `ReviewSummary` already counts as approved).
APPROVING_ACTIONS = frozenset(
    {ReviewAction.APPROVE, ReviewAction.APPROVE_WITH_EDIT, ReviewAction.MERGE}
)


class ReviewLogRecord(BaseModel):
    """One line of `review-log.jsonl`."""

    model_config = ConfigDict(extra="forbid")

    ts: datetime
    candidate_id: str
    type: EntryType
    title: str
    action: ReviewAction
    reason: str | None = Field(
        default=None,
        description=(
            "Why, in the reviewer's own words for `reject`, or which gate fired for "
            "`gate_drop`. None for actions that need no reason."
        ),
    )
    session_id: str | None = None
    harness: str | None = None
    model: str | None = Field(
        default=None,
        description=(
            "The model that produced the candidate, where it is known: the extraction "
            "side knows its provider, so `staged` and `gate_drop` records carry it. A "
            "human decision recorded later does not -- `Evidence` has no `model` field "
            "to read it back from, and guessing from whatever `config.toml` says at "
            "review time would misattribute candidates whenever the provider changed. "
            "Join on `candidate_id` against this candidate's `staged` record instead."
        ),
    )
    edited_fields: list[str] = Field(
        default_factory=list,
        description=(
            "For `approve_with_edit`, the frontmatter fields the human changed, plus "
            "`body` if they rewrote the prose. Empty for every other action."
        ),
    )


def log_path(cairn_root: Path) -> Path:
    return cairn_root / REVIEW_LOG_NAME


def make_record(
    entry: Entry,
    action: ReviewAction,
    *,
    reason: str | None = None,
    model: str | None = None,
    edited_fields: list[str] | None = None,
    now: datetime | None = None,
) -> ReviewLogRecord:
    """A record describing `action` taken on `entry`, stamped at `now`."""

    return ReviewLogRecord(
        ts=now or datetime.now(UTC),
        candidate_id=entry.id,
        type=entry.type,
        title=entry.title,
        action=action,
        reason=reason,
        session_id=entry.evidence.session_id,
        harness=entry.evidence.harness,
        model=model,
        edited_fields=edited_fields or [],
    )


def append(cairn_root: Path, record: ReviewLogRecord) -> bool:
    """Append `record` as one JSON line. Returns whether it was written.

    Never raises: by the time this is called the decision has already been
    applied to the store, so a log that cannot be written is worth a warning
    and nothing more. Opened in append mode per record, so two processes
    writing concurrently interleave whole lines rather than corrupting one.
    """

    path = log_path(cairn_root)
    line = json.dumps(record.model_dump(mode="json"), separators=(",", ":")) + "\n"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
    except OSError as exc:
        logger.warning("could not append to %s: %s", path, exc)
        return False
    return True


def read(cairn_root: Path) -> list[ReviewLogRecord]:
    """Every record in the log, in write order.

    A line that is not valid JSON, or not a valid record, is skipped with a
    warning rather than failing the read: a truncated final line (a process
    killed mid-append) must not make the whole history unreadable.
    """

    path = log_path(cairn_root)
    if not path.is_file():
        return []

    records: list[ReviewLogRecord] = []
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("could not read %s: %s", path, exc)
        return []

    for number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(ReviewLogRecord.model_validate_json(line))
        except ValidationError as exc:
            logger.warning("skipping unreadable %s line %d: %s", path.name, number, exc)
    return records

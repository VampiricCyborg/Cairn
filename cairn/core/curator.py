"""Deterministic gates a candidate clears before a human ever sees it.

Per SPEC.md's "Deterministic curation gates" and the reflect pipeline diagram
in README.md, a candidate that passes schema validation (upstream, in the
provider) is checked, in order: is it a re-proposal of something already
rejected (tombstone check), does it restate an approved or already-staged
entry (near-duplicate check), and does it structurally look like it
contradicts an approved entry (contradiction check). None of this uses a
model — it is regex-free, index-free scoring over titles, tags, and scope,
fast enough to run on every candidate before `staging/` ever sees it.

Near-duplicate and contradiction detection do not adjudicate; they surface a
pairing (`proposed_amendment_of` / `proposed_supersession_of`) for
`cairn review` to show a human, who makes the actual call.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from rapidfuzz import fuzz

from cairn.core.models import Entry, EntryStatus
from cairn.core.similarity import DUPLICATE_WORD_OVERLAP, title_overlap
from cairn.core.store import Store

logger = logging.getLogger(__name__)

#: rapidfuzz `token_sort_ratio` (0-100) at or above which two titles are
#: treated as naming the same lesson. `token_sort_ratio` tokenizes both
#: strings, sorts the tokens, and re-joins before comparing, so word-order
#: differences ("migrations must run before fixtures import" vs "fixtures
#: import before migrations run") don't mask a real duplicate the way a
#: plain edit-distance ratio would. 88 is chosen empirically: high enough
#: that two titles sharing only domain vocabulary ("Alembic migrations must
#: run before fixtures" vs "Alembic migrations require a clean test
#: database") stay below it, low enough to catch a reworded, reordered, or
#: lightly-typo'd repeat of the same lesson.
NEAR_DUPLICATE_TITLE_RATIO = 88.0

#: A title scoring below `NEAR_DUPLICATE_TITLE_RATIO` but at or above this
#: floor is still worth a second look via the body: two sessions can derive
#: the same lesson through a differently-phrased title. Below this floor the
#: titles are unrelated enough that a shared opening paragraph is more
#: likely coincidence than an actual duplicate, so the body check never runs.
NEAR_DUPLICATE_TITLE_FLOOR = 72.0

#: Secondary, lower-weight corroboration for a title in the gray zone
#: between the floor and the primary ratio: the two entries' first prose
#: paragraph, compared the same way. Set lower than the title ratio because
#: an opening paragraph is freer prose than a one-line title and naturally
#: varies more even when it describes the same lesson.
NEAR_DUPLICATE_BODY_RATIO = 75.0

#: Overlap coefficient (`|A & B| / min(|A|, |B|)`) over lowercased tag sets,
#: at or above which a same-scope, same-type, differently-titled entry pair
#: is similar enough in subject matter to flag as a possible contradiction
#: for a human to adjudicate. 0.5 means at least half of the smaller tag set
#: is shared — enough to say the two entries are about the same corner of
#: the codebase without requiring identical tagging.
CONTRADICTION_TAG_OVERLAP = 0.5


@dataclass
class CurationResult:
    """What `Curator.stage_candidate` did with one candidate."""

    outcome: Literal["dropped_tombstoned", "amendment", "supersession", "new"]
    written_path: Path | None
    related_entry_id: str | None


def _first_paragraph(body: str) -> str:
    """The first non-heading paragraph of a Markdown `body`, for a light
    prose-level duplicate signal. Blank blocks and pure ATX headings (`##
    What happens`) are skipped; whatever prose follows is the actual claim."""

    for block in body.strip().split("\n\n"):
        text = " ".join(block.split())
        if text and not text.startswith("#"):
            return text
    return ""


def _scope_overlaps(a: list[str], b: list[str]) -> bool:
    """Whether two glob-pattern scopes name at least one identical pattern.

    Scope is compared exactly, with no glob expansion: a contradiction
    candidate needs to be pinned to the same area of the repo, and matching
    well enough to notice that `tests/**` overlaps `tests/unit/**` would
    trade a few missed pairings for a much larger false-positive rate. An
    unscoped entry (`scope: []`) never overlaps anything — it says nothing
    about where it applies, so pairing it with something on scope alone
    would be a guess, not a signal.
    """

    if not a or not b:
        return False
    return bool(set(a) & set(b))


def _tag_overlap(a: list[str], b: list[str]) -> float:
    """Overlap coefficient over lowercased tag sets, 0.0 if either is empty."""

    a_set, b_set = {tag.lower() for tag in a}, {tag.lower() for tag in b}
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / min(len(a_set), len(b_set))


class Curator:
    """Deterministic curation gates for one `.cairn/` `store`."""

    def __init__(self, store: Store) -> None:
        self.store = store

    # -- tombstones -------------------------------------------------------------

    def _load_tombstones(self) -> list[dict[str, Any]]:
        if not self.store.rejected_dir.is_dir():
            return []
        tombstones: list[dict[str, Any]] = []
        for path in sorted(self.store.rejected_dir.glob("*.json")):
            try:
                tombstones.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                logger.warning("skipping unreadable tombstone %s", path)
        return tombstones

    def check_tombstoned(self, candidate: Entry) -> bool:
        """Whether `candidate` is a re-proposal of something already
        rejected: either an exact match on the evidence excerpt hash (the
        same underlying transcript excerpt proposed again), or a
        near-identical title (the same lesson re-derived by a different
        session, which never shares an excerpt hash with the original)."""

        candidate_hash = candidate.evidence.excerpt_sha256
        for tombstone in self._load_tombstones():
            tombstone_hash = tombstone.get("excerpt_sha256")
            if candidate_hash and tombstone_hash and candidate_hash == tombstone_hash:
                return True

            tombstone_title = str(tombstone.get("title") or "")
            if tombstone_title and title_overlap(candidate.title, tombstone_title) >= (
                DUPLICATE_WORD_OVERLAP
            ):
                return True
        return False

    # -- near-duplicate -----------------------------------------------------------

    def find_near_duplicate(
        self,
        candidate: Entry,
        against: list[Entry],
        *,
        candidate_body: str | None = None,
    ) -> Entry | None:
        """The first entry in `against` that names the same lesson as
        `candidate`, or `None`.

        Primary signal is title similarity (`NEAR_DUPLICATE_TITLE_RATIO`).
        When `candidate_body` is given and a title falls in the gray zone
        between `NEAR_DUPLICATE_TITLE_FLOOR` and the primary ratio, the
        first paragraph of each entry's body is compared as a secondary,
        lower-weight corroborating signal (`NEAR_DUPLICATE_BODY_RATIO`) —
        an entry already written to the store has a body `self.store` can
        read back; a not-yet-written candidate does not, hence the
        keyword-only parameter.
        """

        for entry in against:
            title_score = fuzz.token_sort_ratio(candidate.title, entry.title)
            if title_score >= NEAR_DUPLICATE_TITLE_RATIO:
                return entry

            if candidate_body is None or title_score < NEAR_DUPLICATE_TITLE_FLOOR:
                continue
            entry_body = self.store.read_body(entry)
            if entry_body is None:
                continue
            body_score = fuzz.token_sort_ratio(
                _first_paragraph(candidate_body), _first_paragraph(entry_body)
            )
            if body_score >= NEAR_DUPLICATE_BODY_RATIO:
                return entry

        return None

    # -- contradiction --------------------------------------------------------------

    def find_contradiction(self, candidate: Entry, against: list[Entry]) -> Entry | None:
        """The first entry in `against` that structurally looks like it
        might contradict `candidate`: same `type`, overlapping `scope`, a
        title distinct enough not to already be a near-duplicate, and
        enough shared `tags` to say the two are about the same subject.

        This is a structural flag, not semantic contradiction detection —
        no attempt is made to read the body and decide the claims actually
        conflict. `cairn review` shows the pairing; the human decides.
        """

        for entry in against:
            if entry.type is not candidate.type:
                continue
            if not _scope_overlaps(candidate.scope, entry.scope):
                continue
            title_score = fuzz.token_sort_ratio(candidate.title, entry.title)
            if title_score >= NEAR_DUPLICATE_TITLE_RATIO:
                continue
            if _tag_overlap(candidate.tags, entry.tags) >= CONTRADICTION_TAG_OVERLAP:
                return entry

        return None

    # -- staging ----------------------------------------------------------------

    def stage_candidate(self, candidate: Entry, body: str) -> CurationResult:
        """Run the deterministic gates on `candidate`, in order, and write
        it to the store accordingly. See the module docstring for the gate
        order and `CurationResult` for what each outcome means."""

        assert candidate.status is EntryStatus.STAGED, (
            "stage_candidate expects a candidate already shaped like a staged entry; "
            "schema validity is guaranteed by Entry's own construction upstream, "
            "not re-checked here"
        )

        if self.check_tombstoned(candidate):
            logger.info("dropping %s: matches a rejected/ tombstone", candidate.id)
            return CurationResult(
                outcome="dropped_tombstoned", written_path=None, related_entry_id=None
            )

        # Checked against approved *and* currently-staged entries together: a
        # near-duplicate of an approved entry proposes an amendment to it; a
        # near-duplicate of a peer still awaiting review proposes an
        # amendment to whichever of the two the human approves first. Either
        # way this is what closes the staging-dedup gap the reflect e2e
        # tests surfaced — the old check only ever consulted `approved()`.
        known = [*self.store.approved(), *self.store.load_all(status=EntryStatus.STAGED)]
        duplicate = self.find_near_duplicate(candidate, known, candidate_body=body)
        if duplicate is not None:
            marked = candidate.model_copy(update={"proposed_amendment_of": duplicate.id})
            path = self.store.write_entry(marked, body)
            return CurationResult(
                outcome="amendment", written_path=path, related_entry_id=duplicate.id
            )

        contradicted = self.find_contradiction(candidate, self.store.approved())
        if contradicted is not None:
            marked = candidate.model_copy(update={"proposed_supersession_of": contradicted.id})
            path = self.store.write_entry(marked, body)
            return CurationResult(
                outcome="supersession", written_path=path, related_entry_id=contradicted.id
            )

        path = self.store.write_entry(candidate, body)
        return CurationResult(outcome="new", written_path=path, related_entry_id=None)

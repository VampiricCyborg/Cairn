"""Provider protocol shared by all model backends."""

import hashlib
from typing import Protocol

from cairn.core.models import Entry, EntryType, SessionTrace


def make_entry_id(entry_type: EntryType, seed: str) -> str:
    """A stable id, `"{type}-{6 hex chars}"`, derived from `seed` — no
    `uuid4`, so the same seed always produces the same id."""

    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return f"{entry_type.value}-{digest[:6]}"


class Provider(Protocol):
    """A pluggable extraction backend.

    Implementations turn a normalized session trace into candidate `Entry`
    objects paired with their Markdown body, given what the store already
    knows (`known`), so a provider can skip proposing something already
    covered. Returned entries are well-formed and ready to stage:
    `status=EntryStatus.STAGED` and a freshly generated `id` in
    `"{type}-{6 hex chars}"` form (see `make_entry_id`). The caller does not
    parse raw model output.
    """

    def extract(
        self, trace: SessionTrace, known: list[Entry], max_candidates: int
    ) -> list[tuple[Entry, str]]:
        """Return up to `max_candidates` new `(entry, body)` candidates from `trace`."""
        ...

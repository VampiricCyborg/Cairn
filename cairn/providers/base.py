"""Provider protocol shared by all model backends."""

from typing import Protocol

from cairn.core.models import Entry, SessionTrace


class Provider(Protocol):
    """A pluggable extraction backend.

    Implementations turn a normalized session trace into candidate `Entry`
    objects, given what the store already knows (`known`), so a provider can
    skip proposing something already covered. Returned entries are
    well-formed and ready to stage: `status=EntryStatus.STAGED` and a
    freshly generated `id` in `"{type}-{6 hex chars}"` form. The caller does
    not parse raw model output.
    """

    def extract(
        self, trace: SessionTrace, known: list[Entry], max_candidates: int
    ) -> list[Entry]:
        """Return up to `max_candidates` new candidate entries from `trace`."""
        ...

"""Read, write, validate, and render the `.cairn/` store.

`load_entry` parses and validates a single entry file (the P0 exit gate).
`Store` builds on it: it wraps a `.cairn/` root, scans the directory layout
from SPEC.md into `Entry` objects, and writes new entries back out atomically.
"""

import logging
import os
import re
import tempfile
from pathlib import Path

import frontmatter

from cairn.core.models import Entry, EntryStatus, EntryType

logger = logging.getLogger(__name__)

_SLUG_WORD_RE = re.compile(r"[^a-z0-9]+")


class StoreError(Exception):
    """Base error for problems with a `.cairn/` store."""


class StoreNotFoundError(StoreError):
    """Raised when `Store` is pointed at a path that is not a `.cairn/` store."""


def load_entry(path: Path) -> Entry:
    """Parse a single entry Markdown file and validate it as an `Entry`.

    Raises `pydantic.ValidationError` if the frontmatter does not match the
    entry schema.
    """

    post = frontmatter.load(path)
    return Entry.model_validate(post.metadata)


def _slugify(title: str, *, max_words: int = 6) -> str:
    """A short, filesystem-safe slug for `title`, e.g. for use in filenames."""

    words = _SLUG_WORD_RE.sub(" ", title.lower()).split()
    return "-".join(words[:max_words]) or "entry"


class Store:
    """A `.cairn/` store rooted at `root`, per the layout in SPEC.md."""

    def __init__(self, root: Path) -> None:
        root = Path(root)
        if not root.is_dir():
            raise StoreNotFoundError(
                f"{root} is not a `.cairn/` store: directory does not exist. "
                "Run `cairn init` to create one."
            )
        self.root = root

    @property
    def entries_dir(self) -> Path:
        return self.root / "entries"

    @property
    def staging_dir(self) -> Path:
        return self.root / "staging"

    @property
    def rejected_dir(self) -> Path:
        return self.root / "rejected"

    @property
    def queue_dir(self) -> Path:
        return self.root / "queue"

    @property
    def traces_dir(self) -> Path:
        return self.root / "traces"

    def _scan_flat(self, directory: Path) -> list[Entry]:
        """Load every `*.md` entry directly under `directory`, non-recursive."""

        if not directory.is_dir():
            return []
        return [load_entry(path) for path in sorted(directory.glob("*.md"))]

    def _scan_typed(self, directory: Path, expected_type: EntryType) -> list[Entry]:
        """Like `_scan_flat`, but skip (and log) entries whose frontmatter
        `type` doesn't match the type-partitioned directory they live in."""

        if not directory.is_dir():
            return []
        entries = []
        for path in sorted(directory.glob("*.md")):
            entry = load_entry(path)
            if entry.type is not expected_type:
                logger.warning(
                    "skipping %s: frontmatter type %r does not match its directory (expected %r)",
                    path,
                    entry.type.value,
                    expected_type.value,
                )
                continue
            entries.append(entry)
        return entries

    def load_all(self, status: EntryStatus | None = None) -> list[Entry]:
        """Load entries from the store, optionally filtered by `status`.

        `staged` and `rejected` entries live outside `entries/` (in
        `staging/` and `rejected/` respectively), so they are only scanned
        when specifically asked for via `status`; otherwise this scans
        `entries/{strategy,gotcha,fact}/`.
        """

        if status is EntryStatus.STAGED:
            return self._scan_flat(self.staging_dir)
        if status is EntryStatus.REJECTED:
            return self._scan_flat(self.rejected_dir)

        entries: list[Entry] = []
        for entry_type in EntryType:
            entries.extend(self._scan_typed(self.entries_dir / entry_type.value, entry_type))

        if status is not None:
            entries = [entry for entry in entries if entry.status is status]
        return entries

    def approved(self) -> list[Entry]:
        """Convenience wrapper for `load_all(status=EntryStatus.APPROVED)`."""

        return self.load_all(status=EntryStatus.APPROVED)

    def _target_path(self, entry: Entry) -> Path:
        if entry.status is EntryStatus.STAGED:
            directory = self.staging_dir
        elif entry.status is EntryStatus.REJECTED:
            directory = self.rejected_dir
        else:
            directory = self.entries_dir / entry.type.value

        id_suffix = entry.id.split("-", 1)[-1][:4]
        filename = f"{entry.created.date().isoformat()}-{_slugify(entry.title)}-{id_suffix}.md"
        return directory / filename

    def read_body(self, entry: Entry) -> str | None:
        """The Markdown body already written for `entry`, at the path
        `write_entry` would compute for it, or `None` if nothing is written
        there yet."""

        path = self._target_path(entry)
        if not path.is_file():
            return None
        return frontmatter.load(path).content

    def write_entry(self, entry: Entry, body: str) -> Path:
        """Atomically write `entry` (with Markdown `body`) to the store.

        Writes to a temp file in the destination directory, then `os.replace`s
        it into place, so a process killed mid-write can never leave a
        corrupted or half-written file at the final path.
        """

        target = self._target_path(entry)
        directory = target.parent
        directory.mkdir(parents=True, exist_ok=True)

        post = frontmatter.Post(body, **entry.model_dump(mode="json"))
        rendered = frontmatter.dumps(post) + "\n"

        fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=f".{target.name}.", suffix=".tmp")
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(rendered)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, target)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

        return target

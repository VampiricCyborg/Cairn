"""Read, write, validate, and render the `.cairn/` store.

Only entry loading is implemented so far (the P0 exit gate: "a hand-written
entry validates"). Directory scanning and writing come with P1.
"""

from pathlib import Path

import frontmatter

from cairn.core.models import Entry


def load_entry(path: Path) -> Entry:
    """Parse a single entry Markdown file and validate it as an `Entry`.

    Raises `pydantic.ValidationError` if the frontmatter does not match the
    entry schema.
    """

    post = frontmatter.load(path)
    return Entry.model_validate(post.metadata)

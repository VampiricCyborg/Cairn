"""Rich-based interactive review prompt flow (`cairn review`).

Walks `staging/` one candidate at a time, in file order, and applies the
human decision per SPEC.md's entry lifecycle:

- approve (optionally after editing): `status: approved`, `review` filled in,
  written to `entries/<type>/`, staging file removed.
- reject: a tombstone (id, original title, reason) written to `rejected/`,
  staging file removed.
- merge: for now, a reject with reason "merged into <target id>".
- skip: staging file left untouched.
- quit: stop, leaving every remaining candidate in staging.

Tombstones are written as `rejected/<id>.json`, not `.md`: `rejected/*.md` is
scanned and validated as full `Entry` documents by `Store.load_all` and
`cairn validate`, and a tombstone is deliberately not a full entry.
"""

import json
import os
import shlex
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import frontmatter
import typer
import yaml
from pydantic import ValidationError
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text

from cairn.core.models import Entry, EntryStatus, Review
from cairn.core.store import Store

_ACTIONS = {
    "a": "approve",
    "e": "edit",
    "m": "merge",
    "r": "reject",
    "s": "skip",
    "q": "quit",
}
_ACTION_PROMPT = "[a]pprove, [e]dit, [m]erge into existing, [r]eject, [s]kip, [q]uit"


@dataclass
class ReviewSummary:
    approved: int = 0
    rejected: int = 0
    skipped: int = 0
    remaining: int = 0

    def render(self) -> str:
        return (
            f"{self.approved} approved, {self.rejected} rejected, "
            f"{self.skipped} skipped, {self.remaining} remaining in staging"
        )


@dataclass
class _Candidate:
    path: Path
    entry: Entry
    body: str


def _default_editor() -> str:
    return "notepad" if os.name == "nt" else "vi"


def _launch_editor(path: Path) -> None:
    """Open `path` in `$VISUAL`/`$EDITOR` and block until it exits. A seam for
    tests, which replace it with a function that rewrites the file."""

    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or _default_editor()
    command = shlex.split(editor, posix=os.name != "nt")
    subprocess.run([*command, str(path)], check=False)


def _git_user_name(cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "config", "user.name"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    name = result.stdout.strip()
    return name or None


def _render_markdown(entry: Entry, body: str) -> str:
    """Frontmatter + body, in the same shape `Store.write_entry` persists."""

    post = frontmatter.Post(body, **entry.model_dump(mode="json"))
    return frontmatter.dumps(post) + "\n"


def _parse_markdown(text: str) -> tuple[Entry, str]:
    """Inverse of `_render_markdown`. Raises `ValueError` with a readable
    message if the frontmatter is not valid YAML or not a valid entry."""

    try:
        post = frontmatter.loads(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"frontmatter is not valid YAML: {exc}") from exc
    try:
        entry = Entry.model_validate(post.metadata)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc
    return entry, post.content


class ReviewSession:
    """One interactive pass over `staging/`."""

    def __init__(
        self,
        store: Store,
        *,
        console: Console | None = None,
        repo_root: Path | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.store = store
        self.console = console or Console(highlight=False)
        self.repo_root = repo_root or store.root.parent
        self.now = now
        self.summary = ReviewSummary()
        self._reviewer: str | None = None

    # -- loading --------------------------------------------------------------

    def _load_candidates(self) -> list[_Candidate]:
        # `load_all` validates every staged file (and raises on a malformed
        # one); the paths and bodies come from the same sorted scan it uses.
        entries = self.store.load_all(status=EntryStatus.STAGED)
        paths = sorted(self.store.staging_dir.glob("*.md")) if entries else []
        return [
            _Candidate(path=path, entry=entry, body=frontmatter.load(path).content)
            for path, entry in zip(paths, entries, strict=True)
        ]

    # -- display --------------------------------------------------------------

    def _show(self, candidate: _Candidate, index: int, total: int) -> None:
        entry = candidate.entry
        evidence = entry.evidence
        evidence_parts = [evidence.harness]
        if evidence.session_id:
            evidence_parts.append(f"session {evidence.session_id}")
        if evidence.artifacts:
            evidence_parts.append(", ".join(evidence.artifacts))

        header = Text()
        header.append(f"[{entry.type.value}] ", style="bold cyan")
        header.append(entry.title, style="bold")
        meta = Text(
            f"id: {entry.id}   confidence: {entry.confidence.value}   "
            f"scope: {', '.join(entry.scope) or '(none)'}"
        )
        evidence_line = Text("evidence: " + " · ".join(evidence_parts))

        self.console.print(
            Panel(
                Group(header, meta, evidence_line, Text(""), Text(candidate.body.strip())),
                title="cairn review",
                subtitle=f"candidate {index} of {total}",
                expand=True,
            )
        )

    # -- prompts --------------------------------------------------------------

    def _prompt_action(self) -> str:
        while True:
            choice = str(typer.prompt(_ACTION_PROMPT)).strip().lower()
            if choice in _ACTIONS:
                return _ACTIONS[choice]
            typer.echo(f"unrecognized choice {choice!r}")

    def _reviewer_name(self) -> str:
        if self._reviewer is None:
            default = os.environ.get("CAIRN_REVIEWER") or _git_user_name(self.repo_root)
            while True:
                name = str(typer.prompt("reviewer name", default=default or "")).strip()
                if name:
                    break
                typer.echo("reviewer name is required")
            self._reviewer = name
        return self._reviewer

    # -- actions --------------------------------------------------------------

    def _approve(self, candidate: _Candidate, entry: Entry, body: str) -> None:
        approved = entry.model_copy(
            update={
                "status": EntryStatus.APPROVED,
                "review": Review(approved_by=self._reviewer_name(), approved_at=self.now()),
            }
        )
        # Write the approved copy before removing the staged one, so a crash
        # in between leaves a duplicate rather than losing the entry.
        target = self.store.write_entry(approved, body)
        candidate.path.unlink()
        self.summary.approved += 1
        typer.echo(f"approved {approved.id} -> {target}")

    def _reject(self, candidate: _Candidate, reason: str) -> None:
        tombstone = {
            "id": candidate.entry.id,
            "title": candidate.entry.title,
            "reason": reason,
        }
        self.store.rejected_dir.mkdir(parents=True, exist_ok=True)
        target = self.store.rejected_dir / f"{candidate.entry.id}.json"
        target.write_text(json.dumps(tombstone, indent=2) + "\n", encoding="utf-8")
        candidate.path.unlink()
        self.summary.rejected += 1
        typer.echo(f"rejected {candidate.entry.id}: {reason}")

    def _prompt_reject(self, candidate: _Candidate) -> None:
        while True:
            reason = str(typer.prompt("reason")).strip()
            if reason:
                break
            typer.echo("a rejection reason is required")
        self._reject(candidate, reason)

    def _merge(self, candidate: _Candidate) -> bool:
        """Returns False if there is nothing to merge into, so the caller can
        re-prompt for a different action on the same candidate."""

        # TODO: real amendment support is P3's supersession/amendment work, not this command
        targets = [entry for entry in self.store.approved() if entry.type is candidate.entry.type]
        if not targets:
            typer.echo(f"no approved {candidate.entry.type.value} entries to merge into")
            return False

        typer.echo(f"approved {candidate.entry.type.value} entries:")
        for entry in targets:
            typer.echo(f"  {entry.id}  {entry.title}")
        ids = {entry.id for entry in targets}
        while True:
            target_id = str(typer.prompt("merge into id")).strip()
            if target_id in ids:
                break
            typer.echo(f"{target_id!r} is not one of the listed ids")
        self._reject(candidate, f"merged into {target_id}")
        return True

    def _edit(self, candidate: _Candidate) -> bool:
        """Edit, re-validate, and approve. Returns False if the edit was
        abandoned, leaving the staging file untouched."""

        fd, tmp_name = tempfile.mkstemp(prefix=f"cairn-review-{candidate.entry.id}-", suffix=".md")
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(_render_markdown(candidate.entry, candidate.body))

            while True:
                try:
                    _launch_editor(tmp_path)
                except OSError as exc:
                    typer.echo(f"could not launch editor: {exc}")
                    return False

                try:
                    entry, body = _parse_markdown(tmp_path.read_text(encoding="utf-8"))
                except ValueError as exc:
                    typer.echo(f"edited entry is invalid:\n{exc}")
                    while True:
                        choice = str(typer.prompt("[r]etry edit or [a]bandon edit")).strip()
                        if choice.lower() in ("r", "a"):
                            break
                    if choice.lower() == "a":
                        typer.echo("edit abandoned; candidate left in staging")
                        return False
                    continue

                self._approve(candidate, entry, body)
                return True
        finally:
            tmp_path.unlink(missing_ok=True)

    # -- loop -----------------------------------------------------------------

    def run(self) -> ReviewSummary:
        candidates = self._load_candidates()
        if not candidates:
            typer.echo("nothing to review: staging/ is empty")
            return self.summary

        total = len(candidates)
        for index, candidate in enumerate(candidates, start=1):
            self._show(candidate, index, total)
            if self._decide(candidate) == "quit":
                break

        self.summary.remaining = len(list(self.store.staging_dir.glob("*.md")))
        typer.echo(self.summary.render())
        return self.summary

    def _decide(self, candidate: _Candidate) -> str:
        while True:
            action = self._prompt_action()
            if action == "approve":
                self._approve(candidate, candidate.entry, candidate.body)
            elif action == "edit":
                if not self._edit(candidate):
                    continue
            elif action == "merge":
                if not self._merge(candidate):
                    continue
            elif action == "reject":
                self._prompt_reject(candidate)
            elif action == "skip":
                self.summary.skipped += 1
            return action


def run_review(store: Store, *, console: Console | None = None) -> ReviewSummary:
    """Run an interactive review over `store`'s staged candidates."""

    return ReviewSession(store, console=console).run()

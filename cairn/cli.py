"""Typer entrypoint for the `cairn` CLI."""

import copy
import fnmatch
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anthropic
import frontmatter
import typer
from pydantic import ValidationError

from cairn import __version__
from cairn.core import review_log
from cairn.core.config import (
    HumanApprovalRequiredError,
    human_approval_problem,
    load_provider_name,
)
from cairn.core.curator import CurationResult, Curator
from cairn.core.eval import DEFAULT_JUDGE_MODEL, evaluate_fixture, summarize
from cairn.core.models import Entry, EntryStatus, EntryType, SessionTrace
from cairn.core.normalizer import normalize
from cairn.core.review_log import (
    APPROVING_ACTIONS,
    EXTRACTION_ACTIONS,
    HUMAN_ACTIONS,
    ReviewAction,
    ReviewLogRecord,
)
from cairn.core.store import Store, StoreNotFoundError, load_entry
from cairn.providers.anthropic import AnthropicProvider
from cairn.providers.base import Provider, ProviderUnavailableError
from cairn.providers.mock import MockProvider
from cairn.providers.registry import get_provider
from cairn.review.cli_review import run_review

logger = logging.getLogger(__name__)

app = typer.Typer(help="Cairn: harness-agnostic, git-native memory for coding agents.")
install_app = typer.Typer(help="Wire Cairn into a coding agent's hook/skill system.")
app.add_typer(install_app, name="install")

_SCHEMA_SRC_DIR = Path(__file__).resolve().parent / "schema"
_SCHEMA_FILES = ("entry.schema.json", "trace.schema.json")
_ADAPTERS_DIR = Path(__file__).resolve().parent / "adapters"
_CLAUDE_CODE_ADAPTER_DIR = _ADAPTERS_DIR / "claude_code"

#: Claude Code reads both of these, and hook lists MERGE across scopes rather
#: than override. A registration present in both would fire the capture hook
#: twice per session, so Cairn keeps it in exactly one: the local file.
#: settings.json is the shared, committed file; settings.local.json is the
#: per-project personal one Claude Code keeps out of git. The registration is
#: machine-specific -- it carries this machine's absolute interpreter path --
#: so it belongs in the local file and must never be committed.
_SHARED_SETTINGS = ("settings.json",)
_LOCAL_SETTINGS = "settings.local.json"
_LOCAL_SETTINGS_IGNORE = ".claude/settings.local.json"
_OPENCODE_ADAPTER_DIR = _ADAPTERS_DIR / "opencode"

#: The opencode CLI version `cairn/adapters/opencode/plugin.ts` was last
#: verified against (its `session.idle` event shape and `client.session.messages`
#: response shape, per `cairn.core.normalizer.normalize_opencode_transcript`'s
#: docstring). Bump this after re-verifying the adapter against a newer release.
_OPENCODE_MIN_VERSION = "1.18.30"

_CONFIG_TOML_TEMPLATE = """# .cairn/config.toml
spec_version = "0.1.0"

[provider]
name  = "{provider_name}"          # anthropic | openai | ollama | mock
model = "claude-sonnet-4-6"
max_output_tokens = 2000

# Local model via Ollama (no API key, runs offline):
#   name  = "ollama"
#   model = "qwen2.5-coder:14b"
#   base_url = "http://localhost:11434"

# Any OpenAI-compatible endpoint -- Groq shown, also works for OpenAI, Together, etc.
# api_key_env names the environment variable holding the key; it is never written here.
# Groq rotates its model lineup faster than most providers -- reconfirm the current
# free-tier model name against https://console.groq.com/docs/models periodically.
#   name         = "openai"
#   model        = "openai/gpt-oss-20b"
#   base_url     = "https://api.groq.com/openai/v1"
#   api_key_env  = "GROQ_API_KEY"

[reflect]
max_candidates_per_session = 3
min_session_turns          = 4      # skip trivial sessions entirely
salience = ["errors", "retries", "reversals", "test_transitions"]

[inject]
token_budget    = 1500
scope_filter    = true               # only entries matching touched paths
include_types   = ["strategy", "gotcha", "fact"]

[redaction]
deny_globs = [".env*", "**/secrets/**", "**/*.pem", "**/*.key"]
patterns   = ["sk-[A-Za-z0-9]{{20,}}", "ghp_[A-Za-z0-9]{{36}}", "AKIA[0-9A-Z]{{16}}"]

[review]
require_human_approval = true        # v0 refuses to run without this
"""

_GITIGNORE_LINES = [
    ".cairn/queue/",
    ".cairn/traces/",
    ".cairn/review-log.jsonl",
    ".cairn/capture-log.jsonl",
    ".cairn/hook-trace.jsonl",
]


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(False, "--version", help="Show the Cairn version and exit."),
) -> None:
    if version:
        typer.echo(f"cairn {__version__}")
        raise typer.Exit()


def _ensure_gitignore(repo_root: Path, lines: list[str]) -> None:
    """Append the two gitignored `.cairn/` paths to `repo_root/.gitignore`,
    creating a minimal one if it does not already exist."""

    gitignore_path = repo_root / ".gitignore"
    if gitignore_path.exists():
        existing = gitignore_path.read_text(encoding="utf-8")
        existing_lines = set(existing.splitlines())
        missing = [line for line in lines if line not in existing_lines]
        if missing:
            with gitignore_path.open("a", encoding="utf-8") as handle:
                if existing and not existing.endswith("\n"):
                    handle.write("\n")
                for line in missing:
                    handle.write(line + "\n")
    else:
        gitignore_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@app.command()
def init(
    path: Path = typer.Argument(Path("."), help="Repository root to initialize `.cairn/` in."),
) -> None:
    """Create a fresh `.cairn/` store at PATH."""

    repo_root = path.resolve()
    cairn_root = repo_root / ".cairn"

    if cairn_root.exists():
        typer.echo(f"error: {cairn_root} already exists", err=True)
        raise typer.Exit(code=1)

    for entry_type in EntryType:
        (cairn_root / "entries" / entry_type.value).mkdir(parents=True)
    (cairn_root / "staging").mkdir(parents=True)
    (cairn_root / "rejected").mkdir(parents=True)
    schema_dir = cairn_root / "schema"
    schema_dir.mkdir(parents=True)

    (cairn_root / "VERSION").write_text("0.1.0", encoding="utf-8")
    (cairn_root / "config.toml").write_text(
        _CONFIG_TOML_TEMPLATE.format(provider_name="mock"), encoding="utf-8"
    )
    (cairn_root / "CONTEXT.md").write_text("", encoding="utf-8")

    # git cannot track an empty directory, so a store committed without these
    # loses the layout SPEC.md says is committed: a fresh clone would have no
    # entries/<type>/, staging/ or rejected/ at all.
    for keep_dir in (
        *(cairn_root / "entries" / entry_type.value for entry_type in EntryType),
        cairn_root / "staging",
        cairn_root / "rejected",
    ):
        keep_dir.mkdir(parents=True, exist_ok=True)
        (keep_dir / ".gitkeep").touch()

    for schema_file in _SCHEMA_FILES:
        shutil.copyfile(_SCHEMA_SRC_DIR / schema_file, schema_dir / schema_file)

    _ensure_gitignore(repo_root, _GITIGNORE_LINES)

    typer.echo(f"Initialized Cairn store at {cairn_root}")
    typer.echo(f"  VERSION       {cairn_root / 'VERSION'}")
    typer.echo(f"  config.toml   {cairn_root / 'config.toml'}")
    typer.echo(f"  CONTEXT.md    {cairn_root / 'CONTEXT.md'}")
    typer.echo(f"  entries/      {cairn_root / 'entries'}")
    typer.echo(f"  staging/      {cairn_root / 'staging'}")
    typer.echo(f"  rejected/     {cairn_root / 'rejected'}")
    typer.echo(f"  schema/       {cairn_root / 'schema'}")


def _iter_entry_files(store: Store) -> list[Path]:
    """All entry Markdown files under `entries/`, `staging/`, and `rejected/`,
    in a stable order."""

    files: list[Path] = []
    for entry_type in EntryType:
        directory = store.entries_dir / entry_type.value
        if directory.is_dir():
            files.extend(sorted(directory.glob("*.md")))
    for directory in (store.staging_dir, store.rejected_dir):
        if directory.is_dir():
            files.extend(sorted(directory.glob("*.md")))
    return files


def _entry_warnings(entry: Entry) -> list[str]:
    """Non-fatal quality warnings for an otherwise-valid entry. Only checked
    under `--strict`, where any warning also fails the command."""

    warnings = []
    if not entry.scope:
        warnings.append("empty scope: entry will never match `cairn context --scope`")
    return warnings


@app.command()
def validate(
    path: Path = typer.Argument(Path("."), help="Repository root containing `.cairn/`."),
    strict: bool = typer.Option(
        False, "--strict", help="Also fail on quality warnings (e.g. an empty scope)."
    ),
) -> None:
    """Validate every entry in the store's `entries/`, `staging/`, and `rejected/`."""

    cairn_root = path.resolve() / ".cairn"
    try:
        store = Store(cairn_root)
    except StoreNotFoundError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    files = _iter_entry_files(store)
    failures = 0
    warned = 0

    for entry_file in files:
        try:
            entry = load_entry(entry_file)
        except ValidationError as exc:
            failures += 1
            typer.echo(f"FAIL {entry_file}: {exc}")
            continue

        warnings = _entry_warnings(entry)
        if warnings:
            warned += 1
            for warning in warnings:
                typer.echo(f"WARN {entry_file}: {warning}")
        else:
            typer.echo(f"PASS {entry_file}")

    typer.echo(f"{len(files)} entries checked, {failures} failed, {warned} warned")

    if failures or (strict and warned):
        raise typer.Exit(code=1)


def _load_approved_entries(store: Store) -> list[tuple[Entry, str]]:
    """Approved entries with their Markdown body, in scan order. Entries that
    fail validation are skipped rather than aborting the whole render."""

    results: list[tuple[Entry, str]] = []
    for entry_type in EntryType:
        directory = store.entries_dir / entry_type.value
        if not directory.is_dir():
            continue
        for entry_file in sorted(directory.glob("*.md")):
            try:
                post = frontmatter.load(entry_file)
                entry = Entry.model_validate(post.metadata)
            except ValidationError:
                continue
            if entry.type is not entry_type or entry.status is not EntryStatus.APPROVED:
                continue
            results.append((entry, post.content))
    return results


def _scope_matches(entry: Entry, scope: str) -> bool:
    return any(fnmatch.fnmatch(scope, glob) for glob in entry.scope)


def _render_entry(entry: Entry, body: str) -> str:
    return f"## [{entry.type.value}] {entry.title}\n\n{body.strip()}\n"


def _select_within_budget(items: list[tuple[Entry, str]], budget: int) -> list[tuple[Entry, str]]:
    """Greedily keep entries, in order, while the running approximate token
    count (`len(text) // 4`) stays within `budget`. Approximate because it
    counts characters, not real tokens."""

    selected: list[tuple[Entry, str]] = []
    total_chars = 0
    for entry, body in items:
        chunk = _render_entry(entry, body)
        projected_chars = total_chars + len(chunk)
        if (projected_chars // 4) > budget:
            break
        total_chars = projected_chars
        selected.append((entry, body))
    return selected


def _load_config_toml(cairn_root: Path) -> dict[str, Any]:
    config_path = cairn_root / "config.toml"
    if not config_path.is_file():
        return {}
    try:
        with config_path.open("rb") as handle:
            return tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        logger.warning("could not parse %s: %s", config_path, exc)
        return {}


def _resolve_provider(cairn_root: Path) -> Provider:
    """The provider `cairn reflect` and the SessionStart queue sweep use, per
    `config.toml`'s `[provider]` table (see `cairn.providers.registry.get_provider`).

    Raises `ProviderUnavailableError` for a provider that is unconfigured,
    misconfigured, or unreachable -- callers decide whether that means
    failing loudly (`reflect`) or skipping the sweep (the SessionStart hook).
    """

    return get_provider(_load_config_toml(cairn_root))


#: `CurationResult.outcome` -> the `reason` recorded on a `gate_drop`. Mapped
#: explicitly rather than reusing the outcome string, so a new dropping gate
#: has to name itself here and cannot land in the log as `dropped_<something>`.
_GATE_DROP_REASONS = {"dropped_tombstoned": "tombstone"}


def _provider_model(provider: Provider) -> str | None:
    """The model string `provider` is configured with, if it exposes one.

    `Provider` is a structural protocol with `extract` and nothing else, and
    `MockProvider` has no model at all, so this is a best-effort read for the
    review log rather than part of the interface.
    """

    model = getattr(provider, "model", None)
    return model if isinstance(model, str) and model else None


def _extract_and_stage(
    store: Store, provider: Provider, trace: SessionTrace, max_candidates: int
) -> list[tuple[Entry, CurationResult]]:
    """Extract candidates from `trace` with `provider` and stage every one
    through `Curator.stage_candidate`.

    The single path from a provider's output to `staging/`, shared by
    `cairn reflect` and the SessionStart queue sweep so both run the same
    tombstone, near-duplicate and contradiction gates. Nothing here (or in
    the Curator) can write to `entries/`; only `cairn review` does.

    Every candidate is also recorded in the review log as `staged` or
    `gate_drop` — the extraction-side population, kept apart from the human
    decisions `cairn review` writes (see `cairn.core.review_log`). This is
    what makes the gate-drop rate computable at all: without a `staged`
    record there is no denominator for the candidates the gates killed.
    """

    candidates = provider.extract(trace, known=store.approved(), max_candidates=max_candidates)
    curator = Curator(store)
    model = _provider_model(provider)

    results: list[tuple[Entry, CurationResult]] = []
    for entry, body in candidates:
        result = curator.stage_candidate(entry, body)
        dropped = result.written_path is None
        review_log.append(
            store.root,
            review_log.make_record(
                entry,
                ReviewAction.GATE_DROP if dropped else ReviewAction.STAGED,
                reason=_GATE_DROP_REASONS.get(result.outcome, result.outcome) if dropped else None,
                model=model,
            ),
        )
        results.append((entry, result))
    return results


_QUEUE_SWEEP_MAX_CANDIDATES = 3


def _sweep_queue(store: Store, provider: Provider) -> int:
    """Drain `.cairn/queue/`: normalize each queued job's transcript, run it
    through `provider`, stage the results via `Curator`, then remove the
    queue file. Per the README's hook-safety principle, a bad job (a
    missing transcript, a malformed job record, a normalizer error) is
    caught and logged, never allowed to block the rest of the queue or the
    context injection that follows the sweep. The queue file is removed
    either way -- a permanently malformed job would otherwise be retried,
    and fail identically, on every future session start.

    Returns the number of jobs processed without error.
    """

    if not store.queue_dir.is_dir():
        return 0

    processed = 0
    for job_path in sorted(store.queue_dir.glob("*.json")):
        try:
            job = json.loads(job_path.read_text(encoding="utf-8"))
            transcript_path = Path(str(job["transcript_path"]))
            trace = normalize(transcript_path, harness=str(job.get("harness", "claude-code")))
            for entry, result in _extract_and_stage(
                store, provider, trace, _QUEUE_SWEEP_MAX_CANDIDATES
            ):
                logger.info(
                    "queue sweep: %s from %s -> %s", entry.id, job_path.name, result.outcome
                )
            processed += 1
        except Exception as exc:  # noqa: BLE001 - one bad job must never block the rest
            logger.warning("queue sweep: skipping %s: %s", job_path.name, exc)
        finally:
            job_path.unlink(missing_ok=True)

    return processed


@app.command()
def context(
    path: Path = typer.Argument(Path("."), help="Repository root containing `.cairn/`."),
    budget: int = typer.Option(1500, "--budget", help="Approximate token budget."),
    scope: str | None = typer.Option(
        None, "--scope", help="Only include entries whose `scope` glob-matches this path."
    ),
    format: str = typer.Option("text", "--format", help="Output format: text or json."),
    hook: bool = typer.Option(
        False,
        "--hook",
        help="Sweep .cairn/queue/ first, then emit the SessionStart additionalContext JSON shape.",
    ),
) -> None:
    """Render the approved-entry context block that would be injected into a session."""

    if format not in ("text", "json"):
        typer.echo(f"error: unknown format {format!r}, expected 'text' or 'json'", err=True)
        raise typer.Exit(code=1)

    cairn_root = path.resolve() / ".cairn"
    try:
        store = Store(cairn_root)
    except StoreNotFoundError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if hook:
        try:
            provider = _resolve_provider(cairn_root)
        except ProviderUnavailableError as exc:
            logger.warning("SessionStart sweep: provider unavailable (%s); skipping sweep", exc)
        else:
            _sweep_queue(store, provider)

    items = _load_approved_entries(store)
    if scope is not None:
        items = [(entry, body) for entry, body in items if _scope_matches(entry, scope)]

    selected = _select_within_budget(items, budget)

    if hook:
        block = (
            "# Project knowledge (Cairn)\n\n"
            + "\n".join(_render_entry(entry, body) for entry, body in selected)
            if selected
            else ""
        )
        payload = {
            "hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": block}
        }
        typer.echo(json.dumps(payload))
        return

    if format == "json":
        payload_list = [
            {"id": entry.id, "type": entry.type.value, "title": entry.title}
            for entry, _ in selected
        ]
        typer.echo(json.dumps(payload_list))
        return

    if not selected:
        typer.echo("no approved entries")
        return

    block = "# Project knowledge (Cairn)\n\n" + "\n".join(
        _render_entry(entry, body) for entry, body in selected
    )
    typer.echo(block)


def _describe_curation(entry: Entry, result: CurationResult) -> str:
    """One `cairn reflect` output line for what the Curator did with `entry`."""

    if result.outcome == "dropped_tombstoned":
        return f"dropped {entry.id}: {entry.title} (matches a rejected tombstone)"
    line = f"staged {entry.id}: {entry.title}"
    if result.outcome == "amendment":
        line += f" (proposes amendment of {result.related_entry_id})"
    elif result.outcome == "supersession":
        line += f" (may supersede {result.related_entry_id})"
    return line


@app.command()
def reflect(
    path: Path = typer.Argument(Path("."), help="Repository root containing `.cairn/`."),
    trace: Path = typer.Option(..., "--trace", help="Path to a SessionTrace JSON file."),
    max_candidates: int = typer.Option(
        3, "--max-candidates", help="Maximum candidate entries to extract."
    ),
) -> None:
    """Extract candidate entries from a session trace and stage them."""

    cairn_root = path.resolve() / ".cairn"
    try:
        store = Store(cairn_root)
    except StoreNotFoundError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    try:
        raw = trace.read_text(encoding="utf-8")
        session_trace = SessionTrace.model_validate(json.loads(raw))
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        typer.echo(f"error: could not load trace from {trace}: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    try:
        provider = _resolve_provider(cairn_root)
        results = _extract_and_stage(store, provider, session_trace, max_candidates)
    except ProviderUnavailableError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if not results:
        typer.echo("no candidates extracted")
        return

    for entry, result in results:
        typer.echo(_describe_curation(entry, result))

    staged = sum(1 for _, result in results if result.written_path is not None)
    dropped = len(results) - staged
    summary = f"{staged} entries staged"
    if dropped:
        summary += f", {dropped} dropped"
    typer.echo(summary)


@app.command()
def review(
    path: Path = typer.Argument(Path("."), help="Repository root containing `.cairn/`."),
) -> None:
    """Interactively approve, edit, merge, reject, or skip staged candidates."""

    cairn_root = path.resolve() / ".cairn"
    try:
        store = Store(cairn_root)
    except StoreNotFoundError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    try:
        run_review(store)
    except HumanApprovalRequiredError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except ValidationError as exc:
        typer.echo(f"error: a staged entry is invalid; run `cairn validate`: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _rate(numerator: int, denominator: int, noun: str) -> str:
    if denominator == 0:
        return f"n/a (0/0 {noun})"
    return f"{numerator / denominator:.1%} ({numerator}/{denominator} {noun})"


def _histogram(counts: Counter[str], indent: str = "  ") -> list[str]:
    """Counts, highest first, ties broken alphabetically so output is stable."""

    return [
        f"{indent}{count:>4}  {label}"
        for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _render_review_stats(records: list[ReviewLogRecord]) -> list[str]:
    """`cairn stats --review` output.

    The two populations in the log are reported as separate blocks with
    separate denominators, and never summed: approval rate is about the
    reviewer's judgement of what they were shown, gate-drop rate is about the
    Curator's, and a single blended number would let a well-tuned Curator
    read as a reviewer rejecting things. See `cairn.core.review_log`.
    """

    by_action = Counter(record.action for record in records)
    human = [record for record in records if record.action in HUMAN_ACTIONS]
    extraction = [record for record in records if record.action in EXTRACTION_ACTIONS]

    shown = len(human)
    approvals = sum(by_action[action] for action in APPROVING_ACTIONS)
    decided = shown - by_action[ReviewAction.SKIP]

    lines = [
        f"shown to a human       {shown:>5}",
        f"  approved             {by_action[ReviewAction.APPROVE]:>5}",
        f"  approved with edit   {by_action[ReviewAction.APPROVE_WITH_EDIT]:>5}",
        f"  merged               {by_action[ReviewAction.MERGE]:>5}",
        f"  rejected             {by_action[ReviewAction.REJECT]:>5}",
        f"  skipped              {by_action[ReviewAction.SKIP]:>5}",
        f"approval rate          {_rate(approvals, shown, 'shown')}",
        f"  excluding skipped    {_rate(approvals, decided, 'decided')}",
    ]

    rejections = Counter(
        record.reason or "(no reason recorded)"
        for record in human
        if record.action is ReviewAction.REJECT
    )
    if rejections:
        lines += ["", "rejection reasons", *_histogram(rejections)]

    drops_total = by_action[ReviewAction.GATE_DROP]
    lines += [
        "",
        "gates — a separate population, never folded into approval rate",
        f"extracted              {len(extraction):>5}",
        f"  staged for review    {by_action[ReviewAction.STAGED]:>5}",
        f"  dropped by a gate    {by_action[ReviewAction.GATE_DROP]:>5}",
        f"gate-drop rate         {_rate(drops_total, len(extraction), 'extracted')}",
    ]

    drops = Counter(
        record.reason or "(no gate recorded)"
        for record in extraction
        if record.action is ReviewAction.GATE_DROP
    )
    if drops:
        lines += _histogram(drops)

    return lines


#: Claude Code stores each session transcript under a slug of the project path,
#: e.g. `C:\\Users\\me\\proj` -> `C--Users-me-proj`. Those files are the only
#: record of sessions whose SessionEnd hook never ran, which makes them the
#: denominator for the capture rate -- the capture log alone can only count the
#: runs that happened.
#: How long after its last write a transcript is assumed to belong to a
#: finished session. A capture hook fires within a few hundred ms of the
#: end, so two minutes is generous while still excluding in-flight sessions.
_IN_FLIGHT_GRACE_SECONDS = 120

_CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects"


def _claude_project_slug(repo_root: Path) -> str:
    return re.sub(r"[:/\\]", "-", str(repo_root))


def _session_transcripts(repo_root: Path) -> list[Path]:
    directory = _CLAUDE_PROJECTS_DIR / _claude_project_slug(repo_root)
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.jsonl"))


def _read_capture_log(cairn_root: Path) -> list[dict[str, Any]]:
    path = cairn_root / "capture-log.jsonl"
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            logger.warning("skipping unreadable capture-log line in %s", path)
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _capture_window_start(records: list[dict[str, Any]], cairn_root: Path) -> float | None:
    """Epoch seconds from which capture could have happened at all.

    The first capture-log entry, or -- with an empty log -- the installed
    hook'''s mtime. Anything earlier is a session that ended before this machine
    had a working capture hook.
    """

    stamps = [str(record.get("ts") or "") for record in records]
    parsed = []
    for stamp in stamps:
        try:
            parsed.append(datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC))
        except ValueError:
            continue
    if parsed:
        return min(parsed).timestamp()
    hook = cairn_root / "hooks" / "enqueue.py"
    return hook.stat().st_mtime if hook.is_file() else None


def _render_capture_stats(repo_root: Path, cairn_root: Path) -> list[str]:
    """`cairn stats --capture` output: how often SessionEnd actually captured.

    The denominator is session transcripts, not capture-log lines, precisely
    because the failure mode being measured is the hook not running at all --
    counting only the runs that logged something would report 100% forever.
    """

    records = _read_capture_log(cairn_root)
    transcripts = _session_transcripts(repo_root)
    logged = {str(record.get("session_id")) for record in records}
    enqueued = {str(record.get("session_id")) for record in records if record.get("enqueued")}

    # Only sessions this store could plausibly have captured. Transcripts
    # predating capture logging are history, not misses, so the window starts
    # at the first capture-log entry. The bias runs the wrong way: misses
    # before the very first success fall outside the window, so a reported
    # rate is a ceiling rather than a floor.
    #
    # A transcript still being appended to belongs to a session that has not
    # ended, so its SessionEnd hook has not run and cannot have failed. The
    # grace period keeps the session asking the question out of its own answer.
    cutoff = _capture_window_start(records, cairn_root)
    settled = time.time() - _IN_FLIGHT_GRACE_SECONDS
    considered = {
        path.stem
        for path in transcripts
        if (cutoff is None or path.stat().st_mtime >= cutoff) and path.stat().st_mtime <= settled
    }
    captured = enqueued & considered
    missed = sorted(considered - logged)
    ran_but_failed = sorted((logged - enqueued) & considered)

    lines = [
        f"sessions (transcripts)   {len(considered):>5}",
        f"  hook ran               {len(logged & considered):>5}",
        f"  job enqueued           {len(captured):>5}",
        f"  hook never ran         {len(missed):>5}",
        f"  ran but wrote nothing  {len(ran_but_failed):>5}",
        f"capture rate             {_rate(len(captured), len(considered), 'sessions')}",
    ]

    by_mode: Counter[str] = Counter(str(record.get("mode") or "unrecorded") for record in records)
    if by_mode:
        lines += ["", "captured by mode", *_histogram(by_mode)]
    by_reason: Counter[str] = Counter(str(record.get("reason") or "(none)") for record in records)
    if by_reason:
        lines += ["", "SessionEnd reason", *_histogram(by_reason)]
    return lines


@app.command()
def stats(
    path: Path = typer.Argument(Path("."), help="Repository root containing `.cairn/`."),
    review: bool = typer.Option(
        False, "--review", help="Summarize `.cairn/review-log.jsonl`: approval rate and gate drops."
    ),
    capture: bool = typer.Option(
        False,
        "--capture",
        help="Summarize `.cairn/capture-log.jsonl`: how often SessionEnd actually captured.",
    ),
) -> None:
    """Summarize the review or capture logs. Other stats are not built yet."""

    if not review and not capture:
        typer.echo(
            "error: `cairn stats` currently implements only `--review` and `--capture`; "
            "store, context and health stats are not built yet",
            err=True,
        )
        raise typer.Exit(code=1)

    repo_root = path.resolve()
    cairn_root = repo_root / ".cairn"
    try:
        Store(cairn_root)
    except StoreNotFoundError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if capture:
        typer.echo(f"capture log  {cairn_root / 'capture-log.jsonl'}")
        typer.echo("")
        for line in _render_capture_stats(repo_root, cairn_root):
            typer.echo(line)
        if not review:
            return
        typer.echo("")

    records = review_log.read(cairn_root)
    log_file = review_log.log_path(cairn_root)
    if not records:
        typer.echo(f"no review decisions logged yet ({log_file} is empty or absent)")
        typer.echo("`cairn reflect` records what the gates did; `cairn review` records your calls")
        return

    typer.echo(f"review log  {log_file}  ·  {len(records)} records")
    typer.echo("")
    for line in _render_review_stats(records):
        typer.echo(line)


_EVAL_MAX_CANDIDATES = 3
_EVAL_PROXY_NOTE = (
    "precision/non-redundant are judge- and match-scored, not human-verified — see README's "
    "Evaluation section for why this is a proxy, not ground truth."
)


def _anthropic_client() -> anthropic.Anthropic:
    """The client `cairn eval` judges with (and, without `--mock`, extracts
    with). A seam for tests, which replace it so no API call is made."""

    return anthropic.Anthropic()


def _discover_fixture_pairs(suite: Path) -> list[tuple[Path, Path]]:
    """`(session_trace_N.json, gold_N.json)` pairs in `suite`, ordered by `N`.
    Raises `FileNotFoundError` if a trace has no matching gold file."""

    def order(path: Path) -> tuple[int, str]:
        suffix = path.stem.removeprefix("session_trace_")
        return (int(suffix) if suffix.isdigit() else 2**31, suffix)

    pairs = []
    for trace_path in sorted(suite.glob("session_trace_*.json"), key=order):
        suffix = trace_path.stem.removeprefix("session_trace_")
        gold_path = suite / f"gold_{suffix}.json"
        if not gold_path.is_file():
            raise FileNotFoundError(f"{trace_path.name} has no matching {gold_path.name}")
        pairs.append((trace_path, gold_path))
    return pairs


def _load_gold(path: Path) -> list[dict[str, Any]]:
    gold = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(gold, list) or not all(
        isinstance(item, dict) and isinstance(item.get("title"), str) and item["title"].strip()
        for item in gold
    ):
        raise ValueError("expected a JSON list of objects, each with a non-empty `title`")
    return gold


def _format_metric(metric: dict[str, Any]) -> str:
    if metric["rate"] is None:
        return "n/a (0/0)"
    return f"{metric['rate']:.1%} ({metric['numerator']}/{metric['denominator']})"


@app.command(name="eval")
def eval_suite(
    suite: Path = typer.Option(
        ..., "--suite", help="Directory of session_trace_N.json / gold_N.json pairs."
    ),
    report: Path = typer.Option(
        Path("eval-report.json"), "--report", help="Where to write the per-candidate JSON report."
    ),
    mock: bool = typer.Option(
        False,
        "--mock",
        help="Extract candidates with the offline MockProvider. The judge still calls the "
        "Anthropic API.",
    ),
) -> None:
    """Score extracted candidates against a suite of gold fixtures."""

    try:
        pairs = _discover_fixture_pairs(suite)
    except FileNotFoundError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if not pairs:
        typer.echo(f"error: no session_trace_*.json fixtures in {suite}", err=True)
        raise typer.Exit(code=1)

    client = _anthropic_client()
    provider: Provider
    if mock:
        provider = MockProvider()
        modes = {"extraction": "mock", "judge": "anthropic", "judge_model": DEFAULT_JUDGE_MODEL}
        typer.echo(
            "warning: mixed modes. Extraction uses the offline MockProvider, but the judge "
            f"still calls the Anthropic API ({DEFAULT_JUDGE_MODEL})."
        )
    else:
        anthropic_provider = AnthropicProvider(client=client)
        provider = anthropic_provider
        modes = {
            "extraction": "anthropic",
            "extraction_model": anthropic_provider.model,
            "judge": "anthropic",
            "judge_model": DEFAULT_JUDGE_MODEL,
        }
        typer.echo(
            f"mode: extraction ({anthropic_provider.model}) and judge ({DEFAULT_JUDGE_MODEL}) "
            "both call the Anthropic API."
        )

    emitted: list[Entry] = []
    fixtures: list[dict[str, Any]] = []
    for trace_path, gold_path in pairs:
        try:
            trace = SessionTrace.model_validate(json.loads(trace_path.read_text(encoding="utf-8")))
            gold = _load_gold(gold_path)
        except (OSError, ValueError) as exc:
            typer.echo(f"error: could not load fixture {trace_path.name}: {exc}", err=True)
            raise typer.Exit(code=1) from exc

        try:
            candidates = provider.extract(
                trace, known=list(emitted), max_candidates=_EVAL_MAX_CANDIDATES
            )
        except anthropic.APIError as exc:
            typer.echo(f"error: extraction failed for {trace_path.name}: {exc}", err=True)
            raise typer.Exit(code=1) from exc

        result = evaluate_fixture(candidates, gold, client, known=emitted)
        fixtures.append(
            {
                "trace": trace_path.name,
                "gold": gold_path.name,
                "session_id": trace.session_id,
                **result,
            }
        )
        emitted.extend(entry for entry, _ in candidates)
        typer.echo(
            f"{trace_path.name}: {len(candidates)} candidates, "
            f"{len(result['matched_gold_titles'])}/{result['gold_count']} gold entries matched"
        )

    summary = summarize(fixtures)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps({"modes": modes, "summary": summary, "fixtures": fixtures}, indent=2) + "\n",
        encoding="utf-8",
    )

    typer.echo("")
    typer.echo(f"schema validity  {_format_metric(summary['schema_validity'])}")
    typer.echo(f"precision        {_format_metric(summary['precision'])}")
    typer.echo(f"recall           {_format_metric(summary['recall'])}")
    typer.echo(f"duplicate rate   {_format_metric(summary['duplicate_rate'])}")
    if summary["judge_errors"]:
        typer.echo(
            f"warning: {summary['judge_errors']} judge call(s) failed; those candidates count "
            "as not passing all seven criteria (see judge_error in the report)"
        )
    typer.echo(_EVAL_PROXY_NOTE)
    typer.echo(f"wrote report to {report}")


# -- install claude-code -------------------------------------------------------------


#: Hook script basenames that identify Cairn's SessionEnd capture hook. The
#: retired `.sh` is still matched so `cairn install claude-code` upgrades an
#: existing registration in place, rather than leaving a dead bash hook
#: registered beside the working Python one.
_ENQUEUE_HOOK_SUFFIXES = (".cairn/hooks/enqueue.py", ".cairn/hooks/enqueue.sh")


def _is_cairn_session_end_hook(hook: dict[str, Any]) -> bool:
    """Whether `hook` is Cairn's SessionEnd capture hook.

    The script path is the `command` in the retired bash shape and the first
    `arg` in the current `<python> <script>` one, so both are checked.
    """

    values = [str(hook.get("command", ""))]
    args = hook.get("args")
    if isinstance(args, list):
        values.extend(str(arg) for arg in args)
    return any(
        value.replace("\\", "/").endswith(suffix)
        for value in values
        for suffix in _ENQUEUE_HOOK_SUFFIXES
    )


def _is_cairn_session_start_hook(hook: dict[str, Any]) -> bool:
    """Whether `hook` is Cairn's SessionStart injection hook, in either the
    current `<python> -m cairn context --hook` shape or the retired bare
    `cairn context --hook` one."""

    args = hook.get("args")
    if not isinstance(args, list):
        return False
    arg_strings = [str(arg) for arg in args]
    if "--hook" not in arg_strings:
        return False
    if str(hook.get("command", "")) == "cairn":
        return True
    return "cairn" in arg_strings and "context" in arg_strings


_CAIRN_HOOK_MATCHERS: dict[str, Callable[[dict[str, Any]], bool]] = {
    "SessionEnd": _is_cairn_session_end_hook,
    "SessionStart": _is_cairn_session_start_hook,
}


def _merge_hooks(existing: dict[str, Any], template: dict[str, Any]) -> dict[str, Any]:
    """Merge `template`'s `hooks.<event>` matcher-groups into `existing`'s
    settings.

    An already-registered Cairn hook (matched by `_CAIRN_HOOK_MATCHERS`,
    not by position) is replaced in place, so running install twice is
    idempotent rather than appending a duplicate. Otherwise the template's
    matcher-group is appended alongside whatever is already registered for
    that event. Every other key -- other events, other tools' matcher-groups,
    unrelated top-level settings -- passes through untouched. Neither input
    is mutated; a deep copy of `existing` is returned.
    """

    merged = copy.deepcopy(existing)
    hooks = merged.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        hooks = {}
        merged["hooks"] = hooks

    for event, matcher in _CAIRN_HOOK_MATCHERS.items():
        template_groups = template.get("hooks", {}).get(event, [])
        if not template_groups:
            continue
        our_group = template_groups[0]
        our_hook = our_group["hooks"][0]

        existing_groups = hooks.setdefault(event, [])
        replaced = False
        for group in existing_groups:
            group_hooks = group.get("hooks") if isinstance(group, dict) else None
            if not isinstance(group_hooks, list):
                continue
            for index, existing_hook in enumerate(group_hooks):
                if isinstance(existing_hook, dict) and matcher(existing_hook):
                    group_hooks[index] = copy.deepcopy(our_hook)
                    replaced = True
                    break
            if replaced:
                break

        if not replaced:
            existing_groups.append(copy.deepcopy(our_group))

    return merged


#: Placeholders in `adapters/claude_code/hooks.json`, resolved at install time.
_HOOK_PYTHON_PLACEHOLDER = "__CAIRN_PYTHON__"
_HOOK_ENQUEUE_PLACEHOLDER = "__CAIRN_ENQUEUE_PY__"


def _render_hook_template(template: Any, substitutions: dict[str, str]) -> Any:
    """Substitute the template's placeholder strings, dropping `_comment` keys.

    Substitution happens on the *parsed* JSON rather than its text so that a
    Windows interpreter path full of backslashes needs no JSON escaping.
    """

    if isinstance(template, str):
        return substitutions.get(template, template)
    if isinstance(template, list):
        return [_render_hook_template(item, substitutions) for item in template]
    if isinstance(template, dict):
        return {
            key: _render_hook_template(value, substitutions)
            for key, value in template.items()
            if key != "_comment"
        }
    return template


def _strip_cairn_hooks(settings: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """`settings` with any Cairn hook registration removed, and how many went.

    Used to clean a stale registration out of the shared, committed
    settings.json when install moves it to the local file: leaving it in both
    would fire the capture hook twice per session, because Claude Code merges
    hook lists across scopes instead of letting one override the other. Every
    other tool's hooks, and every unrelated setting, are left untouched.
    """

    cleaned = copy.deepcopy(settings)
    hooks = cleaned.get("hooks")
    if not isinstance(hooks, dict):
        return cleaned, 0

    removed = 0
    for event, matcher in _CAIRN_HOOK_MATCHERS.items():
        groups = hooks.get(event)
        if not isinstance(groups, list):
            continue
        kept_groups = []
        for group in groups:
            if not isinstance(group, dict) or not isinstance(group.get("hooks"), list):
                kept_groups.append(group)
                continue
            kept_hooks = [
                hook for hook in group["hooks"] if not (isinstance(hook, dict) and matcher(hook))
            ]
            removed += len(group["hooks"]) - len(kept_hooks)
            if kept_hooks:
                kept_groups.append({**group, "hooks": kept_hooks})
        if kept_groups:
            hooks[event] = kept_groups
        else:
            hooks.pop(event, None)
    if not hooks:
        cleaned.pop("hooks", None)
    return cleaned, removed


def _load_settings(path: Path) -> dict[str, Any]:
    """Parse a Claude Code settings file, or raise typer.Exit with a clear
    message. A missing file is an empty document, not an error."""

    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        typer.echo(f"error: {path} is not valid JSON: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if not isinstance(loaded, dict):
        typer.echo(f"error: {path} does not contain a JSON object", err=True)
        raise typer.Exit(code=1)
    return loaded


@install_app.command(name="claude-code")
def install_claude_code(
    path: Path = typer.Argument(
        Path("."), help="Repository root to install the Claude Code adapter into."
    ),
) -> None:
    """Register Cairn's SessionEnd/SessionStart hooks and skill in Claude Code.

    Merges into `.claude/settings.json` rather than overwriting it, and is
    idempotent: running this twice does not duplicate hook entries.
    """

    repo_root = path.resolve()
    cairn_root = repo_root / ".cairn"
    if not cairn_root.is_dir():
        typer.echo(f"error: {cairn_root} does not exist; run `cairn init` first", err=True)
        raise typer.Exit(code=1)

    hooks_dir = cairn_root / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    enqueue_dest = hooks_dir / "enqueue.py"
    shutil.copyfile(_CLAUDE_CODE_ADAPTER_DIR / "enqueue.py", enqueue_dest)

    # Absolute, resolved now: Claude Code spawns a hook command directly, so a
    # bare `cairn` only works if it happens to be on PATH (a `uv tool install`,
    # not a `uv sync`), and the interpreter running this install is the one
    # that definitely has cairn importable.
    template = _render_hook_template(
        json.loads((_CLAUDE_CODE_ADAPTER_DIR / "hooks.json").read_text(encoding="utf-8")),
        {
            _HOOK_PYTHON_PLACEHOLDER: sys.executable,
            _HOOK_ENQUEUE_PLACEHOLDER: str(enqueue_dest),
        },
    )

    claude_dir = repo_root / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    settings_path = claude_dir / _LOCAL_SETTINGS

    merged = _merge_hooks(_load_settings(settings_path), template)
    settings_path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")

    # The registration carries this machine's absolute interpreter path, so it
    # must not reach the shared file. A stale one left there would not override
    # this one -- Claude Code merges hook lists across scopes -- it would fire
    # the capture hook a second time per session.
    cleaned_files: list[str] = []
    for shared_name in _SHARED_SETTINGS:
        shared_path = claude_dir / shared_name
        if not shared_path.is_file():
            continue
        cleaned, removed = _strip_cairn_hooks(_load_settings(shared_path))
        if removed:
            shared_path.write_text(json.dumps(cleaned, indent=2) + "\n", encoding="utf-8")
            cleaned_files.append(f"{shared_path} ({removed} hook(s))")

    _ensure_gitignore(repo_root, [_LOCAL_SETTINGS_IGNORE])

    skill_dir = claude_dir / "skills" / "cairn"
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_dest = skill_dir / "SKILL.md"
    shutil.copyfile(_CLAUDE_CODE_ADAPTER_DIR / "SKILL.md", skill_dest)

    typer.echo(f"Installed Claude Code adapter at {repo_root}")
    typer.echo(f"  hooks         {settings_path}")
    for cleaned_file in cleaned_files:
        typer.echo(f"  removed from  {cleaned_file} — hooks merge across scopes, not override")
    typer.echo(f"  enqueue hook  {enqueue_dest}")
    typer.echo(f"  interpreter   {sys.executable}")
    typer.echo(f"  skill         {skill_dest}")


# -- install opencode --------------------------------------------------------------


@install_app.command(name="opencode")
def install_opencode(
    path: Path = typer.Argument(
        Path("."), help="Repository root to install the opencode adapter into."
    ),
) -> None:
    """Link Cairn's session-idle capture plugin into opencode.

    Unlike Claude Code's hooks, which need a JSON registration merged into
    `.claude/settings.json`, opencode auto-discovers local plugins dropped
    into `.opencode/plugins/` (https://opencode.ai/docs/plugins) -- no
    config file to merge. Copying the same plugin source on top of itself
    is naturally idempotent.
    """

    repo_root = path.resolve()
    cairn_root = repo_root / ".cairn"
    if not cairn_root.is_dir():
        typer.echo(f"error: {cairn_root} does not exist; run `cairn init` first", err=True)
        raise typer.Exit(code=1)

    plugins_dir = repo_root / ".opencode" / "plugins"
    plugins_dir.mkdir(parents=True, exist_ok=True)
    plugin_dest = plugins_dir / "cairn.ts"
    shutil.copyfile(_OPENCODE_ADAPTER_DIR / "plugin.ts", plugin_dest)

    typer.echo(f"Installed opencode adapter at {repo_root}")
    typer.echo(f"  plugin        {plugin_dest}")


# -- doctor ----------------------------------------------------------------------
#
# Doctor executes; it does not read. Every check here that describes something
# runnable runs it, because the failure this tool exists to catch is exactly the
# one inspection cannot see: a hook that is correctly registered and cannot
# execute. A check that genuinely cannot be run reports UNVERIFIED, never PASS,
# so "not checked" is never displayed as "works".

PASS, WARN, FAIL, UNVERIFIED = "PASS", "WARN", "FAIL", "UNVERIFIED"

#: The session id doctor's SessionEnd probe claims. Distinctive so a job file
#: that ever escapes into a real queue is recognisable as a probe artifact.
_PROBE_SESSION_ID = "cairn-doctor-probe"
_PROBE_TIMEOUT_SECONDS = 30


@dataclass
class Check:
    """One doctor row. `status` drives both the label and the exit code."""

    status: str
    name: str
    detail: str

    def render(self) -> str:
        return f"{self.status:<10} {self.name:<13} {self.detail}"


def _read_spec_version(cairn_root: Path) -> str:
    version_file = cairn_root / "VERSION"
    if version_file.is_file():
        return version_file.read_text(encoding="utf-8").strip() or "unknown"
    return "unknown"


def _probe_binary(name: str, *args: str) -> tuple[bool, str]:
    """Run `name` and report whether it is actually usable.

    Presence on PATH is not the question: `shutil.which` answers that and still
    misses a binary that is present and cannot execute. Running it is the only
    evidence that counts.
    """

    try:
        result = subprocess.run(
            [name, *args], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    output = (result.stdout or result.stderr or "").strip().splitlines()
    return True, output[0] if output else f"exit {result.returncode}"


def _doctor_store_check(cairn_root: Path) -> Check:
    try:
        store = Store(cairn_root)
    except StoreNotFoundError:
        return Check(FAIL, "store", f"{cairn_root} not found — run `cairn init`")
    approved = len(store.approved())
    staged = len(store.load_all(status=EntryStatus.STAGED))
    return Check(
        PASS,
        "store",
        f".cairn/ present, schema {_read_spec_version(cairn_root)}, "
        f"{approved} entries, {staged} staged",
    )


def _doctor_git_check(repo_root: Path) -> Check:
    runnable, detail = _probe_binary("git", "--version")
    if not runnable:
        return Check(FAIL, "git", f"git is not runnable ({detail}); Cairn shells out to it")
    if not (repo_root / ".git").exists():
        return Check(WARN, "git", f"{detail}, but no .git directory here")
    gitignore = repo_root / ".gitignore"
    ignored = (
        gitignore.is_file()
        and ".cairn/queue/" in gitignore.read_text(encoding="utf-8").splitlines()
    )
    if not ignored:
        return Check(WARN, "git", f"{detail}, .cairn/queue not gitignored — run `cairn init`")
    return Check(PASS, "git", f"{detail}, repository detected, .cairn/queue ignored")


def _doctor_provider_check(cairn_root: Path) -> Check:
    name = load_provider_name(cairn_root)
    if name != "anthropic":
        return Check(PASS, "provider", name)
    if os.environ.get("ANTHROPIC_API_KEY"):
        return Check(PASS, "provider", "anthropic, key found in env")
    return Check(WARN, "provider", "anthropic, ANTHROPIC_API_KEY not set")


def _doctor_review_gate_check(cairn_root: Path) -> Check:
    """`[review] require_human_approval`. A failure here is not cosmetic:
    `cairn review` refuses outright while the gate is off, so nothing can reach
    `entries/` at all."""

    problem = human_approval_problem(cairn_root)
    if problem is None:
        return Check(PASS, "review", "human approval required before anything reaches entries/")
    return Check(FAIL, "review", f"{problem} — `cairn review` will refuse")


def _hook_argv(hook: dict[str, Any]) -> list[str]:
    """The hook's command line as Claude Code spawns it: command then args."""

    argv = [str(hook.get("command", ""))]
    args = hook.get("args")
    if isinstance(args, list):
        argv.extend(str(arg) for arg in args)
    return argv


def _probe_session_end_hook(hook: dict[str, Any]) -> tuple[bool, str]:
    """Spawn the registered SessionEnd command exactly as Claude Code does --
    argv, with the event JSON on stdin -- and check a job record appears.

    The synthesized payload points `cwd` at a throwaway directory, so the probe
    writes its job there and the real queue is never touched.

    Exit status is deliberately not the pass condition. The capture hook is
    required to exit 0 on every path, including every failure path, so the only
    evidence it worked is the file it was supposed to write.
    """

    argv = _hook_argv(hook)
    with tempfile.TemporaryDirectory(prefix="cairn-doctor-") as tmp:
        payload = json.dumps(
            {
                "session_id": _PROBE_SESSION_ID,
                "transcript_path": str(Path(tmp) / "probe-transcript.jsonl"),
                "cwd": tmp,
                "hook_event_name": "SessionEnd",
                "reason": "other",
            }
        )
        try:
            result = subprocess.run(
                argv,
                input=payload,
                capture_output=True,
                text=True,
                timeout=_PROBE_TIMEOUT_SECONDS,
                check=False,
            )
        except OSError as exc:
            # The EFTYPE class of failure lands here: registered correctly,
            # impossible to spawn. This is what doctor used to miss entirely.
            return False, f"SessionEnd could not be spawned ({' '.join(argv)}): {exc}"
        except subprocess.TimeoutExpired:
            return False, f"SessionEnd timed out after {_PROBE_TIMEOUT_SECONDS}s"

        job = Path(tmp) / ".cairn" / "queue" / f"{_PROBE_SESSION_ID}.json"
        if not job.is_file():
            stderr = (result.stderr or "").strip().splitlines()
            hint = f" — {stderr[-1]}" if stderr else ""
            return False, f"SessionEnd ran (exit {result.returncode}) but wrote no job file{hint}"
        try:
            record = json.loads(job.read_text(encoding="utf-8"))
        except ValueError as exc:
            return False, f"SessionEnd wrote a job file that is not valid JSON: {exc}"

    missing = [
        field
        for field in ("session_id", "transcript_path", "harness", "enqueued_at")
        if not record.get(field)
    ]
    if missing:
        return False, f"SessionEnd job record is missing {', '.join(missing)}"
    if record.get("session_id") != _PROBE_SESSION_ID:
        return False, "SessionEnd job record has the wrong session_id"
    return True, "SessionEnd wrote a job record"


def _probe_session_start_hook(hook: dict[str, Any]) -> tuple[bool, str]:
    """Run the registered SessionStart command and check it emits the
    `additionalContext` shape Claude Code injects.

    Run against a throwaway store, because this command sweeps the queue:
    probing it inside the real store would consume the user's pending capture
    jobs as a side effect of asking whether it works.
    """

    argv = _hook_argv(hook)
    with tempfile.TemporaryDirectory(prefix="cairn-doctor-") as tmp:
        (Path(tmp) / ".cairn" / "entries").mkdir(parents=True, exist_ok=True)
        try:
            result = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=_PROBE_TIMEOUT_SECONDS,
                cwd=tmp,
                check=False,
            )
        except OSError as exc:
            return False, f"SessionStart could not be spawned ({' '.join(argv)}): {exc}"
        except subprocess.TimeoutExpired:
            return False, f"SessionStart timed out after {_PROBE_TIMEOUT_SECONDS}s"

    if result.returncode != 0:
        stderr = (result.stderr or "").strip().splitlines()
        hint = f" — {stderr[-1]}" if stderr else ""
        return False, f"SessionStart exited {result.returncode}{hint}"
    try:
        payload = json.loads(result.stdout)
    except ValueError:
        printed = (result.stdout or "").strip().splitlines()
        head = printed[0][:60] if printed else "(no output)"
        return False, f"SessionStart did not emit JSON: {head}"
    specific = payload.get("hookSpecificOutput") if isinstance(payload, dict) else None
    if not isinstance(specific, dict) or "additionalContext" not in specific:
        return False, "SessionStart JSON has no hookSpecificOutput.additionalContext"
    return True, "SessionStart returned additionalContext"


def _registered_cairn_hooks(settings: dict[str, Any]) -> dict[str, dict[str, Any] | None]:
    """The registered Cairn hook for each event, or None where none matches."""

    hooks = settings.get("hooks")
    hooks = hooks if isinstance(hooks, dict) else {}
    return {
        event: next(
            (
                hook
                for group in hooks.get(event) or []
                if isinstance(group, dict)
                for hook in group.get("hooks", [])
                if isinstance(hook, dict) and matcher(hook)
            ),
            None,
        )
        for event, matcher in _CAIRN_HOOK_MATCHERS.items()
    }


def _doctor_claude_code_check(repo_root: Path) -> Check:
    """Which settings file the registration came from, and whether it runs.

    Both scopes are checked because Claude Code merges hook lists across them:
    a registration in both files is not a redundancy, it is the capture hook
    firing twice per session.
    """

    claude_dir = repo_root / ".claude"
    install_hint = "run `cairn install claude-code`"

    found: dict[str, dict[str, dict[str, Any] | None]] = {}
    for name in (_LOCAL_SETTINGS, *_SHARED_SETTINGS):
        path = claude_dir / name
        if not path.is_file():
            continue
        try:
            settings = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return Check(FAIL, "claude-code", f"{path} is not valid JSON")
        if not isinstance(settings, dict):
            return Check(FAIL, "claude-code", f"{path} does not contain a JSON object")
        hooks = _registered_cairn_hooks(settings)
        if any(hook is not None for hook in hooks.values()):
            found[name] = hooks

    if not found:
        return Check(WARN, "claude-code", f"not installed — {install_hint}")
    if len(found) > 1:
        return Check(
            FAIL,
            "claude-code",
            f"registered in {' and '.join(sorted(found))} — hook lists merge across scopes, "
            f"so capture would run twice per session; {install_hint} to clean up",
        )

    source, registered = next(iter(found.items()))
    missing = [event for event, hook in registered.items() if hook is None]
    if missing:
        return Check(
            WARN, "claude-code", f"{', '.join(missing)} not registered in {source} — {install_hint}"
        )

    probes = [
        _probe_session_end_hook(registered["SessionEnd"] or {}),
        _probe_session_start_hook(registered["SessionStart"] or {}),
    ]
    failures = [detail for ok, detail in probes if not ok]
    if failures:
        return Check(FAIL, "claude-code", "; ".join(failures))

    # Both commands run and produce what they should -- but that is
    # spawnability, not capture. Measured on this machine, roughly one headless
    # session in twelve ends without its SessionEnd hook ever starting: the
    # process leaves no trace at all, which is Claude Code's hook lifecycle
    # rather than anything this hook does. Reporting PASS here would restate
    # the original bug in a subtler form -- a green row for a path that
    # silently drops sessions. The real rate is measurable, so doctor points
    # at it instead of asserting.
    return Check(
        UNVERIFIED,
        "claude-code",
        f"from {source}: "
        + "; ".join(detail for _, detail in probes)
        + " — both spawn correctly, but capture under a real session lifecycle is "
        "lossy and unproven here; run `cairn stats --capture` for the measured rate",
    )


def _parse_version(text: str) -> tuple[int, ...] | None:
    """Best-effort dotted-integer version parse, e.g. `"1.18.30"` ->
    `(1, 18, 30)`. `None` for anything without a leading digit run."""

    match = re.search(r"(\d+(?:\.\d+)+)", text.strip())
    if not match:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def _doctor_opencode_check(repo_root: Path) -> Check:
    plugin = repo_root / ".opencode" / "plugins" / "cairn.ts"
    if not plugin.is_file():
        return Check(WARN, "opencode", "no plugin found — run `cairn install opencode`")

    runnable, detail = _probe_binary("opencode", "--version")
    if not runnable:
        # The plugin file is in place, but nothing here has shown opencode can
        # load it. That is not the same as working.
        return Check(
            UNVERIFIED,
            "opencode",
            f"plugin linked, but opencode is not runnable here ({detail}) — capture unproven",
        )

    installed = _parse_version(detail)
    minimum = _parse_version(_OPENCODE_MIN_VERSION)
    if installed and minimum and installed < minimum:
        return Check(
            WARN,
            "opencode",
            f"plugin linked, opencode {'.'.join(map(str, installed))} is older than "
            f"{_OPENCODE_MIN_VERSION}, the version this adapter was verified against",
        )
    return Check(
        UNVERIFIED,
        "opencode",
        f"plugin linked, opencode present ({detail}) — its session.idle hook cannot be "
        "exercised from here; only a real opencode session proves capture",
    )


def _doctor_agents_md_check(repo_root: Path) -> Check:
    agents_md = repo_root / "AGENTS.md"
    if not agents_md.is_file():
        return Check(WARN, "agents-md", "no AGENTS.md found — run: cairn install agents-md")
    if "cairn:begin" in agents_md.read_text(encoding="utf-8"):
        return Check(PASS, "agents-md", "pointer block present")
    return Check(
        WARN, "agents-md", "AGENTS.md found, no Cairn pointer block — run: cairn install agents-md"
    )


@app.command()
def doctor(
    path: Path = typer.Argument(Path("."), help="Repository root containing `.cairn/`."),
) -> None:
    """Check that the store, git integration, provider, and adapters work.

    Exits non-zero if any check FAILs. UNVERIFIED is not a failure: it means the
    check could not be run here, which is reported rather than assumed away.
    """

    repo_root = path.resolve()
    cairn_root = repo_root / ".cairn"

    typer.echo(f"cairn {__version__}  ·  spec {_read_spec_version(cairn_root)}")
    checks = [
        _doctor_store_check(cairn_root),
        _doctor_git_check(repo_root),
        _doctor_provider_check(cairn_root),
        _doctor_review_gate_check(cairn_root),
        _doctor_claude_code_check(repo_root),
        _doctor_opencode_check(repo_root),
        _doctor_agents_md_check(repo_root),
    ]
    for check in checks:
        typer.echo(check.render())

    failures = [check for check in checks if check.status == FAIL]
    if failures:
        typer.echo("")
        typer.echo(f"{len(failures)} check(s) failed: {', '.join(c.name for c in failures)}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()

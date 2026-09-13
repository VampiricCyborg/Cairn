"""Typer entrypoint for the `cairn` CLI."""

import fnmatch
import json
import shutil
from pathlib import Path
from typing import Any

import anthropic
import frontmatter
import typer
from pydantic import ValidationError

from cairn import __version__
from cairn.core.eval import DEFAULT_JUDGE_MODEL, evaluate_fixture, summarize
from cairn.core.models import Entry, EntryStatus, EntryType, SessionTrace
from cairn.core.store import Store, StoreNotFoundError, load_entry
from cairn.providers.anthropic import AnthropicProvider
from cairn.providers.base import Provider
from cairn.providers.mock import MockProvider
from cairn.review.cli_review import run_review

app = typer.Typer(help="Cairn: harness-agnostic, git-native memory for coding agents.")

_SCHEMA_SRC_DIR = Path(__file__).resolve().parent / "schema"
_SCHEMA_FILES = ("entry.schema.json", "trace.schema.json")

_CONFIG_TOML_TEMPLATE = """# .cairn/config.toml
spec_version = "0.1.0"

[provider]
name  = "{provider_name}"          # anthropic | openai | ollama | mock
model = "claude-sonnet-4-6"
max_output_tokens = 2000

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

_GITIGNORE_LINES = [".cairn/queue/", ".cairn/traces/"]


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(False, "--version", help="Show the Cairn version and exit."),
) -> None:
    if version:
        typer.echo(f"cairn {__version__}")
        raise typer.Exit()


def _ensure_gitignore(repo_root: Path) -> None:
    """Append the two gitignored `.cairn/` paths to `repo_root/.gitignore`,
    creating a minimal one if it does not already exist."""

    gitignore_path = repo_root / ".gitignore"
    if gitignore_path.exists():
        existing = gitignore_path.read_text(encoding="utf-8")
        existing_lines = set(existing.splitlines())
        missing = [line for line in _GITIGNORE_LINES if line not in existing_lines]
        if missing:
            with gitignore_path.open("a", encoding="utf-8") as handle:
                if existing and not existing.endswith("\n"):
                    handle.write("\n")
                for line in missing:
                    handle.write(line + "\n")
    else:
        gitignore_path.write_text("\n".join(_GITIGNORE_LINES) + "\n", encoding="utf-8")


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

    for schema_file in _SCHEMA_FILES:
        shutil.copyfile(_SCHEMA_SRC_DIR / schema_file, schema_dir / schema_file)

    _ensure_gitignore(repo_root)

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


@app.command()
def context(
    path: Path = typer.Argument(Path("."), help="Repository root containing `.cairn/`."),
    budget: int = typer.Option(1500, "--budget", help="Approximate token budget."),
    scope: str | None = typer.Option(
        None, "--scope", help="Only include entries whose `scope` glob-matches this path."
    ),
    format: str = typer.Option("text", "--format", help="Output format: text or json."),
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

    items = _load_approved_entries(store)
    if scope is not None:
        items = [(entry, body) for entry, body in items if _scope_matches(entry, scope)]

    selected = _select_within_budget(items, budget)

    if format == "json":
        payload = [
            {"id": entry.id, "type": entry.type.value, "title": entry.title}
            for entry, _ in selected
        ]
        typer.echo(json.dumps(payload))
        return

    if not selected:
        typer.echo("no approved entries")
        return

    block = "# Project knowledge (Cairn)\n\n" + "\n".join(
        _render_entry(entry, body) for entry, body in selected
    )
    typer.echo(block)


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

    # TODO: read provider.name from config.toml once anthropic.py exists
    provider = MockProvider()
    candidates = provider.extract(
        session_trace, known=store.approved(), max_candidates=max_candidates
    )

    if not candidates:
        typer.echo("no candidates extracted")
        return

    for entry, body in candidates:
        store.write_entry(entry, body)
        typer.echo(f"staged {entry.id}: {entry.title}")

    typer.echo(f"{len(candidates)} entries staged")


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
    except ValidationError as exc:
        typer.echo(f"error: a staged entry is invalid; run `cairn validate`: {exc}", err=True)
        raise typer.Exit(code=1) from exc


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


if __name__ == "__main__":
    app()

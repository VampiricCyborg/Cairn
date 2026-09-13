"""Typer entrypoint for the `cairn` CLI."""

import copy
import fnmatch
import json
import logging
import os
import shutil
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anthropic
import frontmatter
import typer
from pydantic import ValidationError

from cairn import __version__
from cairn.core.config import load_provider_name
from cairn.core.curator import Curator
from cairn.core.eval import DEFAULT_JUDGE_MODEL, evaluate_fixture, summarize
from cairn.core.models import Entry, EntryStatus, EntryType, SessionTrace
from cairn.core.normalizer import normalize_claude_code_transcript
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
#   name         = "openai"
#   model        = "llama-3.1-8b-instant"
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

    curator = Curator(store)
    processed = 0
    for job_path in sorted(store.queue_dir.glob("*.json")):
        try:
            job = json.loads(job_path.read_text(encoding="utf-8"))
            transcript_path = Path(str(job["transcript_path"]))
            trace = normalize_claude_code_transcript(transcript_path)
            candidates = provider.extract(
                trace, known=store.approved(), max_candidates=_QUEUE_SWEEP_MAX_CANDIDATES
            )
            for entry, body in candidates:
                result = curator.stage_candidate(entry, body)
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
        candidates = provider.extract(
            session_trace, known=store.approved(), max_candidates=max_candidates
        )
    except ProviderUnavailableError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

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


# -- install claude-code -------------------------------------------------------------


def _is_cairn_session_end_hook(hook: dict[str, Any]) -> bool:
    command = str(hook.get("command", "")).replace("\\", "/")
    return command.endswith(".cairn/hooks/enqueue.sh")


def _is_cairn_session_start_hook(hook: dict[str, Any]) -> bool:
    args = hook.get("args")
    return hook.get("command") == "cairn" and isinstance(args, list) and "--hook" in args


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

    template = json.loads((_CLAUDE_CODE_ADAPTER_DIR / "hooks.json").read_text(encoding="utf-8"))

    claude_dir = repo_root / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    settings_path = claude_dir / "settings.json"

    existing: dict[str, Any] = {}
    if settings_path.is_file():
        try:
            loaded = json.loads(settings_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            typer.echo(f"error: {settings_path} is not valid JSON: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        if not isinstance(loaded, dict):
            typer.echo(f"error: {settings_path} does not contain a JSON object", err=True)
            raise typer.Exit(code=1)
        existing = loaded

    merged = _merge_hooks(existing, template)
    settings_path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")

    hooks_dir = cairn_root / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    enqueue_dest = hooks_dir / "enqueue.sh"
    shutil.copyfile(_CLAUDE_CODE_ADAPTER_DIR / "enqueue.sh", enqueue_dest)
    enqueue_dest.chmod(0o755)

    skill_dir = claude_dir / "skills" / "cairn"
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_dest = skill_dir / "SKILL.md"
    shutil.copyfile(_CLAUDE_CODE_ADAPTER_DIR / "SKILL.md", skill_dest)

    typer.echo(f"Installed Claude Code adapter at {repo_root}")
    typer.echo(f"  hooks         {settings_path}")
    typer.echo(f"  enqueue hook  {enqueue_dest}")
    typer.echo(f"  skill         {skill_dest}")


# -- doctor ----------------------------------------------------------------------


def _read_spec_version(cairn_root: Path) -> str:
    version_file = cairn_root / "VERSION"
    if version_file.is_file():
        return version_file.read_text(encoding="utf-8").strip() or "unknown"
    return "unknown"


def _doctor_store_line(cairn_root: Path) -> str:
    try:
        store = Store(cairn_root)
    except StoreNotFoundError:
        return f"FAIL store            {cairn_root} not found — run `cairn init`"
    approved = len(store.approved())
    staged = len(store.load_all(status=EntryStatus.STAGED))
    return (
        f"PASS store            .cairn/ present, schema {_read_spec_version(cairn_root)}, "
        f"{approved} entries, {staged} staged"
    )


def _doctor_git_line(repo_root: Path) -> str:
    if not (repo_root / ".git").exists():
        return "WARN git              no .git directory found here"
    gitignore = repo_root / ".gitignore"
    ignored = (
        gitignore.is_file()
        and ".cairn/queue/" in gitignore.read_text(encoding="utf-8").splitlines()
    )
    if ignored:
        return "PASS git              repository detected, .cairn/queue ignored"
    return (
        "WARN git              repository detected, .cairn/queue not gitignored — run `cairn init`"
    )


def _doctor_provider_line(cairn_root: Path) -> str:
    name = load_provider_name(cairn_root)
    if name != "anthropic":
        return f"PASS provider         {name}"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "PASS provider         anthropic, key found in env"
    return "WARN provider         anthropic, ANTHROPIC_API_KEY not set"


def _doctor_claude_code_line(repo_root: Path) -> str:
    settings_path = repo_root / ".claude" / "settings.json"
    if not settings_path.is_file():
        return "WARN claude-code      not installed — run `cairn install claude-code`"
    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return f"FAIL claude-code      {settings_path} is not valid JSON"
    if not isinstance(settings, dict):
        return f"FAIL claude-code      {settings_path} does not contain a JSON object"

    hooks = settings.get("hooks")
    hooks = hooks if isinstance(hooks, dict) else {}

    def _has_cairn_hook(event: str, matcher: Callable[[dict[str, Any]], bool]) -> bool:
        groups = hooks.get(event) or []
        return any(
            matcher(hook)
            for group in groups
            if isinstance(group, dict)
            for hook in group.get("hooks", [])
            if isinstance(hook, dict)
        )

    has_end = _has_cairn_hook("SessionEnd", _is_cairn_session_end_hook)
    has_start = _has_cairn_hook("SessionStart", _is_cairn_session_start_hook)

    install_hint = "run `cairn install claude-code`"
    if has_end and has_start:
        return "PASS claude-code      SessionEnd + SessionStart hooks registered"
    if has_end:
        return f"WARN claude-code      SessionEnd registered, SessionStart missing — {install_hint}"
    if has_start:
        return f"WARN claude-code      SessionStart registered, SessionEnd missing — {install_hint}"
    return f"WARN claude-code      hooks not registered — {install_hint}"


def _doctor_opencode_line(repo_root: Path) -> str:
    plugin = repo_root / ".opencode" / "plugins" / "cairn.ts"
    if plugin.is_file():
        return f"PASS opencode         plugin linked at {plugin}"
    return "WARN opencode         no plugin found — run `cairn install opencode`"


def _doctor_agents_md_line(repo_root: Path) -> str:
    agents_md = repo_root / "AGENTS.md"
    if not agents_md.is_file():
        return "WARN agents-md        no AGENTS.md found — run: cairn install agents-md"
    if "cairn:begin" in agents_md.read_text(encoding="utf-8"):
        return "PASS agents-md        pointer block present"
    return (
        "WARN agents-md        AGENTS.md found, no Cairn pointer block — "
        "run: cairn install agents-md"
    )


@app.command()
def doctor(
    path: Path = typer.Argument(Path("."), help="Repository root containing `.cairn/`."),
) -> None:
    """Check that the store, git integration, provider, and adapters are wired up correctly."""

    repo_root = path.resolve()
    cairn_root = repo_root / ".cairn"

    typer.echo(f"cairn {__version__}  ·  spec {_read_spec_version(cairn_root)}")
    typer.echo(_doctor_store_line(cairn_root))
    typer.echo(_doctor_git_line(repo_root))
    typer.echo(_doctor_provider_line(cairn_root))
    typer.echo(_doctor_claude_code_line(repo_root))
    typer.echo(_doctor_opencode_line(repo_root))
    typer.echo(_doctor_agents_md_line(repo_root))


if __name__ == "__main__":
    app()

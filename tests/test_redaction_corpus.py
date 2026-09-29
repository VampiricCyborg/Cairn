"""The redactor's precision and recall against a labeled, checked-in corpus.

CI fails on ANY MUST_NOT_REDACT hit and ANY MUST_REDACT miss. Both directions
are hard failures on purpose:

- a MUST_REDACT miss is a secret reaching a model provider, and potentially a
  committed entry;
- a MUST_NOT_REDACT hit destroys the file paths, errors and code fragments an
  entry needs to satisfy "evidence-backed", which is the failure that was
  actually observed on this repository's own transcripts.

Loosening the entropy pass is the one change here that can leak, which is why
the named pattern set must get stronger in the same commit that narrows it --
`tests/fixtures/redaction/must_redact.txt` is what holds that line.
"""

import re
from pathlib import Path

import pytest

from cairn.core.redactor import Redactor

_CORPUS = Path(__file__).resolve().parent / "fixtures" / "redaction"


def _lines(name: str) -> list[str]:
    text = (_CORPUS / name).read_text(encoding="utf-8")
    return [
        line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")
    ]


def _must_not_redact() -> list[str]:
    return _lines("must_not_redact.txt")


#: Filler alphabets for the corpus placeholders. Deterministic, so a failure
#: reproduces exactly, and diverse enough that the one entropy-only fixture
#: genuinely scores as high-entropy.
_FILLERS = {
    "f": "AbCdEf0123GhIjKl4567MnOpQr89StUvWxYz",
    "u": "ABCDEF0123GHIJKL4567MNOPQR89STUVWXYZ",
    "l": "abcdef0123ghijkl4567mnopqr89stuvwxyz",
}
_PLACEHOLDER_RE = re.compile(r"\{([ful])(\d+)\}")


def _expand(text: str) -> str:
    """Expand `{fN}` / `{uN}` / `{lN}` into deterministic filler.

    The corpus stores placeholders rather than literal secret-shaped strings.
    The first version of the file used literals and GitHub push protection
    rejected the push, having matched its GitLab and Slack detectors. Keeping
    the literals out of git means nothing in history has to be evaluated by a
    scanner, or by a human reading the diff.
    """

    def replace(match: re.Match[str]) -> str:
        alphabet = _FILLERS[match.group(1)]
        length = int(match.group(2))
        return "".join(alphabet[index % len(alphabet)] for index in range(length))

    return _PLACEHOLDER_RE.sub(replace, text)


def _must_redact() -> list[tuple[str, str]]:
    pairs = []
    for line in _lines("must_redact.txt"):
        secret, _, context = line.partition("\t")
        pairs.append((_expand(secret), _expand(context or secret)))
    return pairs


def _redactor() -> Redactor:
    """A redactor configured exactly as `cairn init` leaves a fresh store:
    the built-in patterns, the default deny-globs, no extra config."""

    return Redactor(
        deny_globs=[".env*", "**/secrets/**", "**/*.pem", "**/*.key"],
        patterns=[],
    )


@pytest.mark.parametrize("line", _must_not_redact(), ids=range(len(_must_not_redact())))
def test_must_not_redact(line: str) -> None:
    assert _redactor().redact_text(line) == line, (
        "the redactor destroyed evidence a reflector needs; if this line really "
        "does contain a secret, move it to must_redact.txt"
    )


@pytest.mark.parametrize(
    ("secret", "context"), _must_redact(), ids=[s[:24] for s, _ in _must_redact()]
)
def test_must_redact(secret: str, context: str) -> None:
    redacted = _redactor().redact_text(context)

    assert secret not in redacted, "a secret survived redaction"
    assert "[REDACTED:" in redacted


def test_corpus_precision_and_recall_are_both_perfect() -> None:
    """The per-line tests localize a regression; this one states the headline
    numbers CI is actually enforcing, so a partial pass cannot read as success.
    """

    redactor = _redactor()
    safe = _must_not_redact()
    secrets = _must_redact()

    false_positives = [line for line in safe if redactor.redact_text(line) != line]
    misses = [secret for secret, context in secrets if secret in redactor.redact_text(context)]

    caught = len(secrets) - len(misses)
    precision = caught / (caught + len(false_positives)) if caught + len(false_positives) else 1.0
    recall = caught / len(secrets) if secrets else 1.0

    assert not false_positives, (
        f"{len(false_positives)} MUST_NOT_REDACT hits: {false_positives[:3]}"
    )
    assert not misses, f"{len(misses)} MUST_REDACT misses: {misses[:3]}"
    assert precision == 1.0 and recall == 1.0


def test_the_corpus_holds_no_real_looking_anthropic_key() -> None:
    """A guard on the corpus itself: it is committed, so a real key added here
    would be a leak dressed as a test fixture."""

    text = (_CORPUS / "must_redact.txt").read_text(encoding="utf-8")
    assert "sk-ant-api03-" in text, "expected the fake Anthropic-shaped sample"
    # No literal secret bytes are committed: every random-looking run is a
    # placeholder the loader expands, so the file itself is scanner-clean.
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        for token in line.split():
            body = token.split("-")[-1].split("_")[-1].split(".")[-1]
            mixed = len({character.isdigit() for character in body}) > 1 and not body.islower()
            assert not (len(body) >= 20 and body.isalnum() and mixed), (
                f"literal secret-shaped token committed: {token!r}"
            )


def test_every_builtin_pattern_compiles_and_is_free_of_control_characters() -> None:
    """A guard on a bug that actually happened while writing these patterns.

    `\b` written into a non-raw string becomes a literal backspace (0x08). The
    regex still compiles and still looks correct when printed, but the word
    boundary is gone and the pattern silently matches nothing -- which in a
    redactor means a secret format quietly stops being caught.
    """

    import re

    from cairn.core.redactor import DEFAULT_PATTERNS

    for name, pattern in DEFAULT_PATTERNS:
        re.compile(pattern)
        control = {character for character in pattern if ord(character) < 32}
        assert not control, f"{name} carries control characters {control!r}; a mangled escape"


def test_every_builtin_pattern_matches_at_least_one_corpus_secret() -> None:
    """A pattern nothing exercises is a pattern nobody knows is broken."""

    import re

    from cairn.core.redactor import DEFAULT_PATTERNS

    contexts = [context for _, context in _must_redact()]
    unexercised = [
        name
        for name, pattern in DEFAULT_PATTERNS
        if not any(re.search(pattern, context) for context in contexts)
    ]
    assert not unexercised, f"no corpus line exercises: {unexercised}"

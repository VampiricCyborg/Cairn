"""Secret and PII redaction over normalized session traces.

A precision ladder, in order. Entropy is the last rung, not the first pass.

1. **Deny-globs** remove content outright. A `Diff` whose `file` matches a
   deny-glob (`.env*`, `**/secrets/**`) is dropped from the trace entirely
   rather than redacted in place -- that content must never reach a model
   provider in any form, not even as a mostly-blanked-out patch.
2. **Named patterns** -- `DEFAULT_PATTERNS` below, plus anything from
   `config.toml` -- match known secret formats exactly. This is where the real
   coverage lives: it is high-precision, it needs no threshold, and a new
   provider format is a one-line addition.
3. **Entropy**, only over what survives a structural allowlist.

Why the ladder, rather than leaning on entropy: per-character Shannon entropy
measures *character diversity*, which is not secrecy. Measured on this
repository's own transcripts, the previous entropy-first pass scored a Windows
path at 4.52 bits/char, a URL at 4.14 and an f-string fragment at 4.65 -- all
redacted, all harmless -- while a realistically-shaped AWS access key id
(`AKIA` + 16 uppercase/digits) scored **3.68** and survived. It was destroying
the file paths and code fragments an entry needs to be evidence-backed, and
missing a real secret format, because the two populations are not separated by
the quantity it measures. Named patterns catch the key id; entropy never
should have been asked to.

Entropy now fires only on a token that is long, drawn from a secret-shaped
alphabet, mixed across character classes, and not recognisable as a path, code
identifier, URL, UUID, or hash of a common length. The `tests/fixtures/redaction/`
corpus pins both directions: CI fails on any MUST_NOT_REDACT hit and any
MUST_REDACT miss.
"""

import math
import re
from collections import Counter

from cairn.core.models import SessionTrace

#: Known secret formats, always applied regardless of `config.toml`. Narrowing
#: the entropy check without widening these would trade false positives for
#: leaks, so this set carries the coverage the entropy pass used to be asked
#: for. Each entry is anchored on a literal vendor prefix or an explicit
#: assignment, so a match is evidence of a secret rather than of randomness.
DEFAULT_PATTERNS: tuple[tuple[str, str], ...] = (
    # Anthropic / OpenAI-style keys.
    ("anthropic-key", r"sk-ant-[A-Za-z0-9_-]{16,}"),
    ("openai-project-key", r"sk-proj-[A-Za-z0-9_-]{16,}"),
    ("sk-key", r"sk-[A-Za-z0-9]{20,}"),
    # GitHub: personal, OAuth, user-to-server, server-to-server, refresh.
    ("github-token", r"gh[pousr]_[A-Za-z0-9]{20,}"),
    ("github-pat", r"github_pat_[A-Za-z0-9_]{22,}"),
    # AWS access key id, and the secret that usually sits beside it.
    ("aws-key-id", r"(?:AKIA|ASIA|ABIA|ACCA)[0-9A-Z]{16}"),
    ("aws-secret", r"(?i)aws_secret_access_key\s*[:=]\s*\S{20,}"),
    # GitLab, Slack, Google, npm, Stripe, SendGrid, Hugging Face, Postman.
    ("gitlab-token", r"glpat-[A-Za-z0-9_-]{16,}"),
    ("slack-token", r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    ("google-key", r"AIza[0-9A-Za-z_-]{35}"),
    ("npm-token", r"npm_[A-Za-z0-9]{36}"),
    ("stripe-key", r"(?:sk|pk|rk)_(?:live|test)_[A-Za-z0-9]{16,}"),
    ("sendgrid-key", r"SG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}"),
    ("huggingface-token", r"hf_[A-Za-z0-9]{30,}"),
    ("postman-key", r"PMAK-[A-Za-z0-9]{20,}"),
    # Private keys and JWTs.
    ("private-key", r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----"),
    ("jwt", r"eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
    # Credentials embedded in a URL, e.g. postgres://user:password@host/db.
    ("url-credentials", r"[a-z][a-z0-9+.-]*://[^\s:@/]+:[^\s@/]+@"),
    # An explicit assignment is strong evidence whatever the value looks like:
    # this is what catches a secret with no recognisable shape.
    (
        "assignment",
        r"(?i)\b(?:api[_-]?key|secret|token|passwd|password|pwd|auth)"
        r"\s*[:=]\s*[\"']?[A-Za-z0-9_\-./+=]{12,}",
    ),
    ("bearer", r"(?i)\bbearer\s+[A-Za-z0-9._-]{20,}"),
)

#: Minimum length before the entropy pass will consider a token at all. Raised
#: from 20: below this a token is far more likely to be an identifier or a
#: short hash than a credential, and anything genuinely secret and this short
#: needs a named pattern rather than a guess.
ENTROPY_MIN_LENGTH = 24

#: Entropy only ever looks at tokens drawn from this alphabet. A token carrying
#: brackets, quotes, backslashes or other code punctuation is a code fragment
#: or a Windows path, not a bare credential -- this single rule is what stops
#: `{by_action[ReviewAction.STAGED]:>5}` and `C:\Users\...\test_x.py` from
#: being scored at all.
_SECRET_ALPHABET_RE = re.compile(r"^[A-Za-z0-9_\-+/=.:~@]+$")

#: Structural allowlist: shapes that are never a secret, whatever they score.
_STRUCTURAL_SAFE_RES = (
    # URL or scheme-prefixed reference (credentials in one are caught above).
    re.compile(r"^[a-z][a-z0-9+.-]*://"),
    # Dotted or snake_case code identifier, e.g. sqlalchemy.exc.ProgrammingError
    # or test_store_write_surface. A separator is required: without one, a bare
    # run of letters and digits is indistinguishable from a random credential,
    # and exempting that shape would wave every alphanumeric secret through.
    re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:[._][A-Za-z0-9_]+)+$"),
    # UUID, e.g. a session id.
    re.compile(r"^[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}$"),
    # Hex at the lengths git and the hashing world actually use: short SHA,
    # SHA-1, SHA-256. Excerpt hashes and commit SHAs are provenance, not secrets.
    re.compile(r"^[0-9a-fA-F]{7,8}$|^[0-9a-fA-F]{32}$|^[0-9a-fA-F]{40}$|^[0-9a-fA-F]{64}$"),
    # Dotted version or numeric run, e.g. 3.13.1, 2.52.0.windows.1.
    re.compile(r"^[0-9][0-9A-Za-z.+-]*$"),
)

#: A path segment: what every part of a path-shaped token has to look like.
#: Base64 payloads fail this because of `+` and `=`.
_PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_.@~ -]*$")
_PATH_START_RE = re.compile(r"^(?:[A-Za-z]:[\\/]|[.~]{0,2}[\\/])")
_FILE_EXTENSION_RE = re.compile(r"\.[A-Za-z][A-Za-z0-9]{0,9}$")

#: Punctuation that is part of the surrounding prose rather than the token.
_TRIM_CHARS = "\"'`(),;:[]{}<>.!?"

_LITERAL_PREFIX_RE = re.compile(r"^[A-Za-z0-9_.-]+")
_TOKEN_RE = re.compile(r"\S+")
_HIGH_ENTROPY_NAME = "high-entropy"

_Span = tuple[int, int, str]


def _is_path_shaped(token: str) -> bool:
    """Whether `token` looks like a filesystem path rather than a credential.

    Requires a separator plus either a recognisable path start (drive letter,
    `./`, `../`, `~/`, `/`) or a file extension, and requires every segment to
    be made of path-ish characters -- so a base64 blob containing `/` does not
    qualify.
    """

    if not any(sep in token for sep in "/\\"):
        return False
    segments = [segment for segment in re.split(r"[\\/]+", token) if segment]
    if len(segments) < 2 or not all(_PATH_SEGMENT_RE.match(seg) for seg in segments):
        return False
    return bool(_PATH_START_RE.match(token) or _FILE_EXTENSION_RE.search(token))


def _character_classes(token: str) -> int:
    """How many of lower / upper / digit the token uses."""

    return sum(
        (
            any(character.islower() for character in token),
            any(character.isupper() for character in token),
            any(character.isdigit() for character in token),
        )
    )


def _is_structurally_safe(token: str) -> bool:
    """Whether `token` is a shape that is never a secret, so entropy must not
    be allowed to have an opinion about it."""

    if not _SECRET_ALPHABET_RE.match(token) or _is_path_shaped(token):
        return True
    if not any(regex.match(token) for regex in _STRUCTURAL_SAFE_RES):
        return False
    # An identifier-shaped token that is mostly digits is not really an
    # identifier; `_STRUCTURAL_SAFE_RES` is shape-only, so the density check
    # lives here rather than being bolted onto the regex.
    digits = sum(character.isdigit() for character in token)
    return digits / len(token) < 0.3


def _looks_like_a_secret(token: str) -> bool:
    """Whether the entropy pass is even allowed to score `token`.

    Every rung below has to hold: long enough, secret-shaped alphabet, not a
    recognisable structure, and mixed across character classes. Entropy is
    asked last, and only about tokens nothing else could explain.
    """

    return (
        len(token) >= ENTROPY_MIN_LENGTH
        and not _is_structurally_safe(token)
        and _character_classes(token) >= 2
    )


def _glob_to_regex(glob: str) -> re.Pattern[str]:
    """Translate a gitignore-style glob (`**` included) into a regex.

    `*` and `?` never cross a `/`; `**/` matches zero or more leading path
    segments; a trailing `/**` matches the segment separator plus anything
    after it. This is a small hand-rolled translator rather than
    `fnmatch`/`pathlib.match` because neither treats `**` as "zero or more
    directories" the way the config format (and users) expect.
    """

    glob = glob.replace("\\", "/")
    parts: list[str] = []
    i, n = 0, len(glob)
    while i < n:
        if glob[i : i + 3] == "**/":
            parts.append("(?:.*/)?")
            i += 3
        elif glob[i : i + 3] == "/**" and i + 3 == n:
            parts.append("/.*")
            i += 3
        elif glob[i : i + 2] == "**":
            parts.append(".*")
            i += 2
        elif glob[i] == "*":
            parts.append("[^/]*")
            i += 1
        elif glob[i] == "?":
            parts.append("[^/]")
            i += 1
        else:
            parts.append(re.escape(glob[i]))
            i += 1
    return re.compile("^" + "".join(parts) + "$")


def _pattern_name(pattern: str, used: set[str]) -> str:
    """A short, collision-free name for `pattern`, derived from its literal
    prefix (e.g. `sk-[A-Za-z0-9]{20,}` -> `sk`) so the redaction placeholder
    never has to include any of the actual matched text."""

    match = _LITERAL_PREFIX_RE.match(pattern)
    name = (match.group(0) if match else pattern).rstrip("-_.") or "pattern"
    if name in used:
        suffix = 2
        while f"{name}{suffix}" in used:
            suffix += 1
        name = f"{name}{suffix}"
    used.add(name)
    return name


def _shannon_entropy(token: str) -> float:
    """Shannon entropy of `token`, in bits per character."""

    if not token:
        return 0.0
    length = len(token)
    counts = Counter(token)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


class Redactor:
    """Redacts secrets from a `SessionTrace` before it reaches a model
    provider. Compiles `deny_globs` and `patterns` once at construction."""

    def __init__(
        self,
        deny_globs: list[str],
        patterns: list[str],
        entropy_threshold: float = 4.0,
    ) -> None:
        self.entropy_threshold = entropy_threshold
        self._deny_globs = [(("/" in glob), _glob_to_regex(glob)) for glob in deny_globs]

        # Built-ins first, then config: a store that configures nothing still
        # gets the known-format coverage, and a user pattern can only add.
        # Built-ins first (explicitly named, since a name derived from a
        # regex prefix is unreadable for these), then config patterns, whose
        # names are still derived from their literal prefix.
        used_names: set[str] = set()
        compiled = [(name, re.compile(regex)) for name, regex in DEFAULT_PATTERNS]
        used_names.update(name for name, _ in DEFAULT_PATTERNS)
        known = {regex for _, regex in DEFAULT_PATTERNS}
        compiled.extend(
            (_pattern_name(pattern, used_names), re.compile(pattern))
            for pattern in dict.fromkeys(patterns)
            if pattern not in known
        )
        self._patterns = compiled

    def _file_is_denied(self, file: str) -> bool:
        path = file.replace("\\", "/")
        basename = path.rsplit("/", 1)[-1]
        return any(
            regex.match(path if is_full_path else basename)
            for is_full_path, regex in self._deny_globs
        )

    def _named_pattern_spans(self, text: str) -> list[_Span]:
        spans = sorted(
            (
                (match.start(), match.end(), name)
                for name, regex in self._patterns
                for match in regex.finditer(text)
                if match.start() != match.end()
            ),
            key=lambda span: (span[0], span[1]),
        )
        selected: list[_Span] = []
        last_end = -1
        for start, end, name in spans:
            if start < last_end:
                continue  # overlaps an already-selected, earlier-starting match
            selected.append((start, end, name))
            last_end = end
        return selected

    def _entropy_spans(self, text: str, already: list[_Span]) -> list[_Span]:
        spans: list[_Span] = []
        for match in _TOKEN_RE.finditer(text):
            start, end = match.start(), match.end()
            overlaps_existing = any(
                start < existing_end and existing_start < end
                for existing_start, existing_end, _ in already
            )
            overlaps_new = any(start < s_end and s_start < end for s_start, s_end, _ in spans)
            if overlaps_existing or overlaps_new:
                continue
            token = match.group().strip(_TRIM_CHARS)
            if not token:
                continue
            if not _looks_like_a_secret(token):
                continue
            if _shannon_entropy(token) >= self.entropy_threshold:
                # Re-anchor on the trimmed token so surrounding prose
                # punctuation is not swallowed into the placeholder.
                offset = match.group().index(token)
                spans.append((start + offset, start + offset + len(token), _HIGH_ENTROPY_NAME))
        return spans

    def redact_text(self, text: str) -> str:
        """Replace every secret-shaped substring of `text` with a
        `[REDACTED:<name>]` placeholder: named `patterns` first, then an
        entropy check over whatever those patterns didn't already catch."""

        named_spans = self._named_pattern_spans(text)
        all_spans = sorted(named_spans + self._entropy_spans(text, named_spans))

        pieces: list[str] = []
        cursor = 0
        for start, end, name in all_spans:
            if start < cursor:
                continue
            pieces.append(text[cursor:start])
            pieces.append(f"[REDACTED:{name}]")
            cursor = end
        pieces.append(text[cursor:])
        return "".join(pieces)

    def redact_trace(self, trace: SessionTrace) -> SessionTrace:
        """Return a new `SessionTrace` with denied diffs removed and every
        remaining text field redacted. `trace` is never mutated."""

        diffs = [
            diff.model_copy(update={"patch": self.redact_text(diff.patch)})
            for diff in trace.diffs
            if not self._file_is_denied(diff.file)
        ]
        turns = [
            turn.model_copy(
                update={
                    "content": self.redact_text(turn.content),
                    "tool_results": [
                        result.model_copy(
                            update={"output": self._redact_tool_output(result.output)}
                        )
                        for result in turn.tool_results
                    ],
                }
            )
            for turn in trace.turns
        ]
        errors = [self.redact_text(error) for error in trace.errors]

        return trace.model_copy(update={"diffs": diffs, "turns": turns, "errors": errors})

    def _redact_tool_output(self, output: object) -> object:
        """Stringify and redact `output`, per SPEC.md's redaction pass over
        `ToolResult.output`. `None` (the default, meaning "no output") is
        left as `None` rather than turned into the string `"None"`."""

        if output is None:
            return None
        return self.redact_text(str(output))

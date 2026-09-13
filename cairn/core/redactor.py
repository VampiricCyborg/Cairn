"""Secret and PII redaction over normalized session traces.

Two layers, per SPEC.md and the README's "Security and privacy" section:

1. Deny-globs remove content outright. A `Diff` whose `file` matches a
   deny-glob (e.g. `.env*`, `**/secrets/**`) is dropped from the trace
   entirely rather than redacted in place — that content must never reach
   a model provider in any form, not even as a mostly-blanked-out patch.
2. Everything else is scanned with the configured secret `patterns`
   (regexes such as `sk-[A-Za-z0-9]{20,}`, each matched occurrence replaced
   with `[REDACTED:<name>]`, the name derived from the pattern's literal
   prefix so no partial secret text leaks into the placeholder), plus a
   secondary entropy check: any 20+ character run of non-whitespace text
   not already caught by a named pattern is Shannon-entropy-scored per
   character, and redacted as `[REDACTED:high-entropy]` above
   `entropy_threshold` bits/char.

The entropy check is deliberately loose. It trades some false positives —
redacting long hashes, base64 build artifacts, or unusually distinct
identifiers — for not missing secret formats the named patterns don't know
about. That tradeoff is intended: catching an unknown secret format matters
more than the occasional over-redacted hash.
"""

import math
import re
from collections import Counter

from cairn.core.models import SessionTrace

_LITERAL_PREFIX_RE = re.compile(r"^[A-Za-z0-9_.-]+")
_TOKEN_RE = re.compile(r"\S{20,}")
_HIGH_ENTROPY_NAME = "high-entropy"

_Span = tuple[int, int, str]


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

        used_names: set[str] = set()
        self._patterns = [
            (_pattern_name(pattern, used_names), re.compile(pattern)) for pattern in patterns
        ]

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
            if _shannon_entropy(match.group()) >= self.entropy_threshold:
                spans.append((start, end, _HIGH_ENTROPY_NAME))
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

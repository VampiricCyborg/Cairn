"""Title word-overlap scoring.

Shared by `MockProvider`'s near-duplicate check and `cairn eval`'s
candidate-to-gold matching, so both agree on when two titles name the same
lesson. It is deliberately crude: real near-duplicate detection
(rapidfuzz-backed) is the Curator's job.
"""

import string

#: Overlap score at or above which two titles are treated as the same lesson.
DUPLICATE_WORD_OVERLAP = 0.6


def title_words(text: str) -> set[str]:
    """Lowercased whitespace-separated words, with leading and trailing
    punctuation stripped so that `` `alembic` `` and ``alembic,`` count as
    the same word. Inner punctuation is kept (`tests/conftest.py` stays one
    word)."""

    words = (word.strip(string.punctuation) for word in text.lower().split())
    return {word for word in words if word}


def title_overlap(a: str, b: str) -> float:
    """How strongly two titles overlap, from 0.0 to 1.0.

    1.0 if either lowercased title contains the other. Otherwise the overlap
    coefficient of their word sets, `|A & B| / min(|A|, |B|)`, or 0.0 if
    either has no words.
    """

    a_lower, b_lower = a.lower(), b.lower()
    if a_lower in b_lower or b_lower in a_lower:
        return 1.0

    a_words, b_words = title_words(a), title_words(b)
    if not a_words or not b_words:
        return 0.0
    return len(a_words & b_words) / min(len(a_words), len(b_words))

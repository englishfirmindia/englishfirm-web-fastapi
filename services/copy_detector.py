"""Verbatim-copy detector for the writing scorers (SWT, WE, SST).

Deterministic safety net for the PTE rubric clause "0 = ... or a near-verbatim
copy of the source passage". The LLM is sometimes too lenient on copy-paste
submissions, so we layer a rule-based check on top: if a substantial fraction
of the student's words sit inside long verbatim n-grams from the passage, we
floor Content to 0 regardless of what the LLM said. The existing off-topic
floor then drops PTE to 10.

Pure function. No external state, no I/O. Never raises.

Tunables:
  _NGRAM_SIZE        — n-gram length. 6 is a sweet spot: small enough to catch
                       copied phrases, big enough to skip stock collocations
                       ("the use of electricity", "around the world").
  _COVERAGE_FLOOR    — fraction of student words that must sit inside a
                       matching n-gram to flag a copy. 0.70 mirrors how PTE
                       human markers treat "near-verbatim".
  _MIN_USER_WORDS    — skip the check on very short submissions (no signal).
"""

from __future__ import annotations

import re
from typing import Iterable, Tuple

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*")

_NGRAM_SIZE      = 6
_COVERAGE_FLOOR  = 0.70
_MIN_USER_WORDS  = 5


def _normalize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace. Returns word list."""
    if not text:
        return []
    return _WORD_RE.findall(text.lower())


def _ngrams(words: list[str], n: int) -> Iterable[Tuple[str, ...]]:
    for i in range(len(words) - n + 1):
        yield tuple(words[i:i + n])


def detect_verbatim_copy(passage: str, user_text: str) -> Tuple[bool, dict]:
    """Returns (is_copy, details).

    details = {
      "coverage_pct":    float in [0,1],
      "user_word_count": int,
      "ngram_size":      int,
      "threshold":       float,
    }

    `is_copy` is True iff ≥ _COVERAGE_FLOOR of the user's words sit inside a
    verbatim n-gram (size _NGRAM_SIZE) from the passage.
    """
    base_details = {
        "coverage_pct": 0.0,
        "user_word_count": 0,
        "ngram_size": _NGRAM_SIZE,
        "threshold": _COVERAGE_FLOOR,
    }

    p_words = _normalize(passage)
    u_words = _normalize(user_text)
    base_details["user_word_count"] = len(u_words)

    if len(u_words) < _MIN_USER_WORDS or len(p_words) < _NGRAM_SIZE:
        return False, base_details

    passage_ngrams = set(_ngrams(p_words, _NGRAM_SIZE))
    if not passage_ngrams:
        return False, base_details

    covered = [False] * len(u_words)
    for i, gram in enumerate(_ngrams(u_words, _NGRAM_SIZE)):
        if gram in passage_ngrams:
            for j in range(_NGRAM_SIZE):
                covered[i + j] = True

    coverage = sum(covered) / len(u_words) if u_words else 0.0
    base_details["coverage_pct"] = round(coverage, 3)
    return coverage >= _COVERAGE_FLOOR, base_details

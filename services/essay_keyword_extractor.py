"""Deterministic essay-prompt keyword extraction + presence check.

Used by the Essay (WE) scoring path as a hard content gate: if ANY keyword
from the prompt is missing from the student's essay, content is capped at
2.0 (matches the off-prompt / weak-engagement floor PTE applies).

Pure functions — no I/O, no LLM. ~1 ms per essay. Never raises.
"""

from __future__ import annotations

import re
from typing import List, Tuple


# Curated stopword list — common English function words + essay-prompt
# instruction verbs + generic essay-scaffold nouns. Tuned for PTE prompts.
# Anything in this set is NEVER counted as a topical keyword.
_ESSAY_STOPWORDS = {
    # Articles / determiners
    "a", "an", "the", "this", "that", "these", "those", "some", "any",
    "each", "every", "all", "both", "either", "neither", "one", "two",
    # Conjunctions / prepositions
    "and", "or", "but", "so", "yet", "of", "to", "in", "on", "at", "for",
    "with", "by", "as", "from", "into", "about", "between", "through",
    "than", "then", "if", "while", "whilst", "when", "where", "why",
    "what", "which", "who", "whom", "whose", "how", "however",
    "because", "since", "though", "although",
    # Pronouns / auxiliaries
    "you", "your", "yours", "we", "our", "ours", "they", "their", "theirs",
    "them", "it", "its", "he", "his", "him", "she", "her", "hers", "i",
    "me", "my", "mine", "us",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "have", "has", "had", "having",
    "do", "does", "did", "doing",
    "will", "would", "shall", "should", "can", "could", "may", "might",
    "must",
    # Common essay-prompt instruction verbs (rarely echoed verbatim)
    "discuss", "examine", "explain", "describe", "argue", "argues",
    "consider", "analyse", "analyze", "evaluate", "compare", "contrast",
    "give", "gives", "provide", "provides", "suggest", "suggests",
    "outline", "state", "states", "identify", "identifies", "list",
    "agree", "disagree", "support", "oppose",
    # Generic essay-scaffold nouns
    "view", "views", "opinion", "opinions", "side", "sides", "essay",
    "topic", "topics", "point", "points", "reason", "reasons",
    "question", "questions", "answer", "answers",
    # Generics
    "many", "much", "few", "more", "less", "most", "least", "very",
    "people", "person", "persons", "thing", "things", "way", "ways",
    "kind", "kinds", "type", "types", "example", "examples",
    # Demonstratives / hedges
    "now", "today", "yesterday", "tomorrow", "here", "there",
    "yes", "no", "not", "only", "also", "too", "just", "even",
    # "Other(s)" common in essay prompts
    "other", "others",
}


def extract_keywords(prompt: str) -> List[str]:
    """Deterministic keyword extraction from an essay prompt.

    Rules (applied in order):
      • Lowercase, strip punctuation
      • Drop stopwords (curated list above)
      • Drop tokens shorter than 4 chars
      • Dedupe, preserve first-occurrence order

    Returns list of distinct keyword tokens.
    """
    if not prompt or not prompt.strip():
        return []
    tokens = re.findall(r"[a-z]+", prompt.lower())
    out: List[str] = []
    seen: set = set()
    for t in tokens:
        if t in _ESSAY_STOPWORDS:
            continue
        if len(t) < 4:
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _stem(word: str) -> str:
    """Tiny suffix-stripping stemmer — matches study/studies/studying.

    Pure heuristic; not a full Porter stemmer. Strips one of the common
    inflectional suffixes when the residual is at least 3 chars.
    """
    for suffix in ("ing", "ies", "ied", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[: -len(suffix)]
    return word


def check_keyword_presence(keywords: List[str], body: str) -> Tuple[int, int, List[str]]:
    """Check which keywords appear in the essay body.

    Uses light stemming so "study" matches "studies" / "studying" / "studied".
    Single-token match (multi-word phrases are checked per-token).

    Returns (present_count, total_keywords, missing_keywords).
    """
    total = len(keywords)
    if total == 0:
        return (0, 0, [])
    body_tokens = re.findall(r"[a-z]+", (body or "").lower())
    body_raw = set(body_tokens)
    body_stems = {_stem(t) for t in body_tokens}

    missing: List[str] = []
    present = 0
    for kw in keywords:
        kw_lower = kw.lower()
        if kw_lower in body_raw:
            present += 1
            continue
        if _stem(kw_lower) in body_stems:
            present += 1
            continue
        missing.append(kw)
    return (present, total, missing)

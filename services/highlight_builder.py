"""Build a flat list of character-range highlights for the student response
based on:
  - Heuristic findings (extra spaces, missing initial cap, improper ALL-CAPS,
    missing terminal period). These already carry positions from
    services.grammar_heuristic.
  - LLM mistake quotes for grammar and spelling. These are verbatim substrings
    we substring-match into the body to recover positions.

Output shape (each element):
    {"start": int, "end": int,
     "type":     "spelling" | "grammar",
     "category": "spelling" | "grammar" | "spacing" | "capitalisation" | "punctuation",
     "kind": str, "hint": str, "word": str | None,
     "correction": str | None, "reason": str | None}

`type` is the legacy 2-class field kept for backwards compatibility with old
clients (both render red/orange). `category` is the 5-class field new clients
use to colour each error class distinctly. `kind` is the specific finding
(`spelling_typo`, `extra_space`, `missing_initial_cap`, `improper_caps`,
`missing_terminal`, `llm_grammar`). `hint` is a short human-readable string
for the tooltip; for heuristic findings it carries an explicit "X → Y"
correction.

Pure function. No external state, no I/O.
"""
import difflib
import re
from typing import Iterable, Optional, Tuple


def build_highlights(
    body: str,
    heuristic_findings: dict,
    spelling_quotes: Iterable = (),
    grammar_quotes: Iterable = (),
) -> list:
    """Returns a list of highlights sorted by start offset, with overlapping
    ranges deduplicated (later, less-specific highlights are dropped).

    `spelling_quotes` and `grammar_quotes` accept EITHER:
      - plain strings (legacy)               → no correction shown in hint
      - dicts {quote, correction, reason}    → hint becomes "Grammar: 'X' → 'Y' (reason)"
    Each list may mix both shapes.
    """
    if not body:
        return []

    highlights: list = []

    # ── Heuristic positions ──────────────────────────────────────────────
    for start, end in (heuristic_findings.get("extra_space_ranges") or []):
        # Capture a word-window around the offending whitespace so the
        # CorrectionsPanel pill has something to render — `word: None`
        # silently filters the entry out in _corrections_by_category, and
        # painting a highlight over pure whitespace is visually mute.
        # Quote: e.g. "Games   say" ; Correction: "Games say".
        quote, correction = _spacing_snippet(body, start, end)
        highlights.append({
            "start": start, "end": end,
            "type": "grammar", "category": "spacing", "kind": "extra_space",
            "hint": "Spacing: extra space → single space",
            "word": quote, "correction": correction, "reason": "double space",
        })
    if heuristic_findings.get("missing_initial_cap"):
        pos = heuristic_findings.get("initial_cap_position")
        if isinstance(pos, int):
            ch = body[pos:pos + 1] if pos < len(body) else ""
            fix = ch.upper() if ch else ""
            highlights.append({
                "start": pos, "end": pos + 1,
                "type": "grammar", "category": "capitalisation",
                "kind": "missing_initial_cap",
                "hint": (
                    f"Capitalisation: '{ch}' → '{fix}'"
                    if ch else "Capitalisation: capitalise sentence start"
                ),
                "word": ch or None,
                "correction": fix or None,
                "reason": "sentence must start with a capital letter",
            })
    for start, end, word in (heuristic_findings.get("improper_caps_ranges") or []):
        fix = word.capitalize() if word else None
        highlights.append({
            "start": start, "end": end,
            "type": "grammar", "category": "capitalisation",
            "kind": "improper_caps",
            "hint": f"Capitalisation: '{word}' → '{fix}'" if fix else "Improper capitalisation",
            "word": word, "correction": fix,
            "reason": "avoid ALL-CAPS in formal writing",
        })
    if heuristic_findings.get("missing_terminal"):
        pos = heuristic_findings.get("terminal_position")
        if isinstance(pos, int):
            # Zero-width caret at the missing-terminator location. Frontend
            # renders this as an insert-marker.
            highlights.append({
                "start": pos, "end": pos,
                "type": "grammar", "category": "punctuation",
                "kind": "missing_terminal",
                "hint": "Punctuation: add '.', '!' or '?' at the end",
                "word": None, "correction": ".",
                "reason": "missing terminal punctuation",
            })
    # Rule 5 — space before punctuation. Underline the offending span +
    # carry the suggested replacement so the corrections panel can show
    # "ages , → ages,".
    for start, end, original, suggested in (
        heuristic_findings.get("space_before_comma") or ()
    ):
        highlights.append({
            "start": start, "end": end,
            "type": "grammar", "category": "punctuation",
            "kind": "space_before_punct",
            "hint": f"Punctuation: '{original}' → '{suggested}'",
            "word": original, "correction": suggested,
            "reason": "no space before punctuation",
        })
    # Rule 6 — missing space after punctuation. Same shape.
    for start, end, original, suggested in (
        heuristic_findings.get("missing_space_after_punct") or ()
    ):
        highlights.append({
            "start": start, "end": end,
            "type": "grammar", "category": "punctuation",
            "kind": "missing_space_after_punct",
            "hint": f"Punctuation: '{original}' → '{suggested}'",
            "word": original, "correction": suggested,
            "reason": "missing space after punctuation",
        })

    # ── LLM-quoted spelling typos ────────────────────────────────────────
    used_ranges: set = set()
    for item in (spelling_quotes or ()):
        quote, correction, reason = _normalize_mistake(item)
        if not quote:
            continue
        rng = _first_match(body, quote, used_ranges, correction)
        if rng is None:
            continue
        used_ranges.add(rng)
        start, end = rng
        highlights.append({
            "start": start, "end": end,
            "type": "spelling", "category": "spelling", "kind": "spelling_typo",
            "hint": _build_hint("Spelling", quote, correction, reason),
            "word": quote,
            "correction": correction,
            "reason": reason,
        })

    # ── LLM-quoted grammar mistakes ──────────────────────────────────────
    for item in (grammar_quotes or ()):
        quote, correction, reason = _normalize_mistake(item)
        if not quote:
            continue
        rng = _first_match(body, quote, used_ranges, correction)
        if rng is None:
            continue
        used_ranges.add(rng)
        start, end = rng
        highlights.append({
            "start": start, "end": end,
            "type": "grammar", "category": "grammar", "kind": "llm_grammar",
            "hint": _build_hint("Grammar", quote, correction, reason),
            "word": quote,
            "correction": correction,
            "reason": reason,
        })

    highlights.sort(key=lambda h: (h["start"], h["end"]))
    return _dedupe_overlaps(highlights)


def _spacing_snippet(body: str, start: int, end: int) -> Tuple[Optional[str], Optional[str]]:
    """Build a (quote, correction) pair around an extra-space range so the
    spacing correction has something to render in the CorrectionsPanel.

    Strategy: snap the range outward to the nearest word boundaries on each
    side. Quote includes the bad whitespace ("Games   say"); correction
    collapses internal runs of whitespace to a single space ("Games say").

    Falls back to (None, None) when the range can't be widened (e.g. body
    edges, malformed input) — the highlight then keeps the legacy
    word: None shape and is silently dropped by _corrections_by_category,
    same as before the fix. Never raises.
    """
    if not body:
        return None, None
    n = len(body)
    if start < 0 or end > n or start >= end:
        return None, None

    # Snap left: walk back past the whitespace, then back through the
    # preceding word until another whitespace (or BOS) is hit.
    left = start
    while left > 0 and body[left - 1].isspace():
        left -= 1
    while left > 0 and not body[left - 1].isspace():
        left -= 1

    # Snap right: walk forward past the whitespace, then forward through
    # the following word until another whitespace (or EOS) is hit.
    right = end
    while right < n and body[right].isspace():
        right += 1
    while right < n and not body[right].isspace():
        right += 1

    quote = body[left:right]
    if not quote or quote.isspace():
        return None, None
    correction = re.sub(r"\s+", " ", quote).strip()
    if quote == correction:
        # Range somehow didn't have extra whitespace after the snap (e.g.
        # the bad range was at a word edge). Bail and keep legacy shape.
        return None, None
    return quote, correction


def _normalize_mistake(item):
    """Accept either a plain string or a {quote, correction, reason} dict
    (and a {word, correction, offsets} dict for the hybrid spelling
    checker). Returns (quote, correction, reason)."""
    if isinstance(item, str):
        return item, None, None
    if isinstance(item, dict):
        quote = item.get("quote") or item.get("word")
        correction = item.get("correction")
        reason = item.get("reason")
        return (
            str(quote).strip() if quote else None,
            str(correction).strip() if correction else None,
            str(reason).strip() if reason else None,
        )
    return None, None, None


def _build_hint(prefix: str, quote: str, correction: Optional[str], reason: Optional[str]) -> str:
    """Build the tooltip text shown on hover. Includes the correction when
    the LLM (or spelling checker) provided one."""
    base = f"{prefix}: '{quote}'"
    if correction:
        base += f" → '{correction}'"
    if reason:
        base += f" ({reason})"
    return base


def _first_match(
    body: str,
    needle: str,
    used: set,
    correction: Optional[str] = None,
) -> Optional[Tuple[int, int]]:
    """Locate `needle` in `body`, preferring whole-word matches.

    Tier 1 — case-sensitive word-boundary. The LLM almost always quotes the
    typo verbatim including its (incorrect) casing, so a case-sensitive whole-
    word hit is the gold standard. Short quotes like 'th', 'an', 'is' get
    correctly anchored to the standalone word instead of latching onto the
    first occurrence inside an unrelated longer word.

    Tier 2 — case-INsensitive word-boundary. Handles LLM quotes that flipped
    the casing (e.g. it returns 'english' but the body has 'English' lower-
    cased mid-sentence).

    Tier 3 — substring fallback (the legacy behaviour). Used when the quote
    contains punctuation/whitespace such that word-boundary anchors can't
    apply (e.g. multi-word phrase, leading punctuation).

    Whenever a tier returns more than one candidate, the candidates are
    ranked by surrounding-word similarity to `correction`: a match whose
    surrounding word is already identical to the correction is almost
    certainly the corrected form (not the typo) and is pushed to the back.
    Otherwise, closeness-to-correction is treated as typo-likelihood (a
    typo is usually a small edit away from the correction).
    """
    if not needle or not body:
        return None
    needle = needle.strip()
    if not needle:
        return None

    # Tier 1: case-sensitive word boundary
    candidates = [r for r in _wb_matches(body, needle, ci=False) if not _contained(r, used)]
    if candidates:
        return _rank(candidates, body, correction)[0]

    # Tier 2: case-insensitive word boundary
    candidates = [r for r in _wb_matches(body, needle, ci=True) if not _contained(r, used)]
    if candidates:
        return _rank(candidates, body, correction)[0]

    # Tier 3: substring fallback — case-sensitive first, then insensitive.
    idx = body.find(needle)
    while idx != -1:
        rng = (idx, idx + len(needle))
        if not _contained(rng, used):
            return rng
        idx = body.find(needle, idx + 1)
    lower_body = body.lower()
    lower_needle = needle.lower()
    idx = lower_body.find(lower_needle)
    while idx != -1:
        rng = (idx, idx + len(needle))
        if not _contained(rng, used):
            return rng
        idx = lower_body.find(lower_needle, idx + 1)
    return None


def _wb_matches(body: str, needle: str, *, ci: bool) -> list:
    """All word-boundary positions of `needle` in `body`. Boundary anchors
    are only added on alphanumeric edges of the needle — quotes that start
    or end with punctuation drop through to substring fallback in Tier 3."""
    left = r"\b" if needle[0].isalnum() or needle[0] == "_" else ""
    right = r"\b" if needle[-1].isalnum() or needle[-1] == "_" else ""
    pattern = left + re.escape(needle) + right
    flags = re.IGNORECASE if ci else 0
    try:
        return [(m.start(), m.end()) for m in re.finditer(pattern, body, flags=flags)]
    except re.error:
        return []


def _rank(candidates: list, body: str, correction: Optional[str]) -> list:
    """Sort candidates by typo-likelihood (lower score = more typo-like).
    Stable for a single candidate; meaningful only when multiple matches
    compete for the same quote."""
    if len(candidates) == 1:
        return candidates
    return sorted(candidates, key=lambda r: _typo_likelihood(body, r, correction))


def _typo_likelihood(body: str, rng: Tuple[int, int], correction: Optional[str]) -> float:
    """Lower = more likely to be the actual typo location.

    If `correction` isn't provided we can't differentiate, so all candidates
    get the same score (zero) and the iteration order wins — equivalent to
    the old "first match" behaviour.

    If the surrounding word at `rng` already IS the correction (case-
    insensitive), this candidate can't be the typo and gets pushed to the
    back. Otherwise we use 1 - SequenceMatcher.ratio() so candidates closer
    to `correction` (small edit distance ⇒ likely typo) rank earlier."""
    if not correction:
        return 0.0
    surrounding = _surrounding_word(body, rng)
    if surrounding.lower() == correction.lower():
        return 99.0
    return 1.0 - difflib.SequenceMatcher(None, surrounding.lower(), correction.lower()).ratio()


def _surrounding_word(body: str, rng: Tuple[int, int]) -> str:
    """Extend `rng` outwards to the surrounding word's boundaries in `body`."""
    s, e = rng
    while s > 0 and (body[s - 1].isalpha() or body[s - 1] in "'-"):
        s -= 1
    while e < len(body) and (body[e].isalpha() or body[e] in "'-"):
        e += 1
    return body[s:e]


def _contained(rng: Tuple[int, int], used: set) -> bool:
    s, e = rng
    for us, ue in used:
        if s >= us and e <= ue:
            return True
    return False


def _dedupe_overlaps(highlights: list) -> list:
    """Drop later highlights whose range is strictly inside an earlier one
    OF THE SAME CATEGORY. Cross-category nesting is preserved so an LLM
    grammar phrase containing a misspelled word doesn't suppress the
    spelling annotation on that word (Spelling and Grammar are rendered
    with different colours and have separate correction pills — the user
    needs both)."""
    out: list = []
    for h in highlights:
        contained = False
        for prev in out:
            if h is prev:
                continue
            same_category = h.get("category") == prev.get("category")
            if not same_category:
                continue
            if h["start"] >= prev["start"] and h["end"] <= prev["end"]:
                # Identical or strictly nested within the same category.
                if (h["start"], h["end"]) == (prev["start"], prev["end"]):
                    contained = True
                    break
                if h["start"] > prev["start"] or h["end"] < prev["end"]:
                    contained = True
                    break
        if not contained:
            out.append(h)
    return out

"""Build a flat list of character-range highlights for the student response
based on:
  - Heuristic findings (extra spaces, missing initial cap, improper ALL-CAPS,
    missing terminal period). These already carry positions from
    services.grammar_heuristic.
  - LLM mistake quotes for grammar and spelling. These are verbatim substrings
    we substring-match into the body to recover positions.

Output shape (each element):
    {"start": int, "end": int, "type": "spelling" | "grammar",
     "kind": str, "hint": str, "word": str | None}

`type` is what the frontend uses to choose the colour (currently both render
red). `kind` is the specific finding (`spelling_typo`, `extra_space`,
`missing_initial_cap`, `improper_caps`, `missing_terminal`, `llm_grammar`).
`hint` is a short human-readable string for the tooltip.

Pure function. No external state, no I/O.
"""
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
        highlights.append({
            "start": start, "end": end,
            "type": "grammar", "kind": "extra_space",
            "hint": "Extra space", "word": None,
        })
    if heuristic_findings.get("missing_initial_cap"):
        pos = heuristic_findings.get("initial_cap_position")
        if isinstance(pos, int):
            highlights.append({
                "start": pos, "end": pos + 1,
                "type": "grammar", "kind": "missing_initial_cap",
                "hint": "Sentence should start with a capital letter",
                "word": body[pos:pos + 1] if pos < len(body) else None,
            })
    for start, end, word in (heuristic_findings.get("improper_caps_ranges") or []):
        highlights.append({
            "start": start, "end": end,
            "type": "grammar", "kind": "improper_caps",
            "hint": f"Improper capitalisation: {word}", "word": word,
        })
    if heuristic_findings.get("missing_terminal"):
        pos = heuristic_findings.get("terminal_position")
        if isinstance(pos, int):
            # Zero-width caret at the missing-terminator location. Frontend
            # can render this as a small red dot or insert-marker.
            highlights.append({
                "start": pos, "end": pos,
                "type": "grammar", "kind": "missing_terminal",
                "hint": "Missing terminal punctuation (. ! or ?)",
                "word": None,
            })

    # ── LLM-quoted spelling typos ────────────────────────────────────────
    used_ranges: set = set()
    for item in (spelling_quotes or ()):
        quote, correction, reason = _normalize_mistake(item)
        if not quote:
            continue
        rng = _first_match(body, quote, used_ranges)
        if rng is None:
            continue
        used_ranges.add(rng)
        start, end = rng
        highlights.append({
            "start": start, "end": end,
            "type": "spelling", "kind": "spelling_typo",
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
        rng = _first_match(body, quote, used_ranges)
        if rng is None:
            continue
        used_ranges.add(rng)
        start, end = rng
        highlights.append({
            "start": start, "end": end,
            "type": "grammar", "kind": "llm_grammar",
            "hint": _build_hint("Grammar", quote, correction, reason),
            "word": quote,
            "correction": correction,
            "reason": reason,
        })

    highlights.sort(key=lambda h: (h["start"], h["end"]))
    return _dedupe_overlaps(highlights)


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


def _first_match(body: str, needle: str, used: set) -> Optional[Tuple[int, int]]:
    """Return (start, end) of the first occurrence of `needle` in `body` that
    doesn't fall entirely inside an already-used range. Case-sensitive first;
    falls back to case-insensitive if no exact match. Returns None if not found."""
    if not needle or not body:
        return None
    needle = needle.strip()
    if not needle:
        return None
    # Exact match
    idx = body.find(needle)
    while idx != -1:
        rng = (idx, idx + len(needle))
        if not _contained(rng, used):
            return rng
        idx = body.find(needle, idx + 1)
    # Case-insensitive fallback
    lower_body = body.lower()
    lower_needle = needle.lower()
    idx = lower_body.find(lower_needle)
    while idx != -1:
        rng = (idx, idx + len(needle))
        if not _contained(rng, used):
            return rng
        idx = lower_body.find(lower_needle, idx + 1)
    return None


def _contained(rng: Tuple[int, int], used: set) -> bool:
    s, e = rng
    for us, ue in used:
        if s >= us and e <= ue:
            return True
    return False


def _dedupe_overlaps(highlights: list) -> list:
    """Drop later highlights whose range is strictly inside an earlier one.
    Keeps the first (lowest-start) annotation when ranges nest."""
    out: list = []
    for h in highlights:
        contained = False
        for prev in out:
            if h["start"] >= prev["start"] and h["end"] <= prev["end"] and h is not prev:
                # Skip if identical to or fully inside an existing range
                if (h["start"], h["end"]) == (prev["start"], prev["end"]):
                    contained = True
                    break
                if h["start"] > prev["start"] or h["end"] < prev["end"]:
                    contained = True
                    break
        if not contained:
            out.append(h)
    return out

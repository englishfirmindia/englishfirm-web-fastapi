"""Deterministic grammar heuristic — layered on top of the LLM grammar
score via min(heuristic, llm). Catches deterministic failure modes the
LLM tends to overlook (extra spaces, missing terminal period, lowercase
sentence start) without false-positive blizzards on legitimate proper
nouns.

Rules (each starts the heuristic at max=2 and deducts):
  1. Extra spaces:        per-extra-space deduction. "a   b" (3 spaces) → −2.
  2. Missing initial cap: −1 if first non-whitespace char is not uppercase.
  3. Improper ALL-CAPS:   −1 per word that is ALL UPPERCASE, ≥3 letters,
                          mid-sentence (not at a sentence start), and not in
                          the abbreviation allowlist. Option A — narrowest
                          detector; will not false-positive on Sochi, Olympic,
                          UNEP, etc. because those aren't ALL-CAPS or are
                          allowlisted.
  4. Missing terminal:    −1 if the response does not end with . ! or ?

Final heuristic score = max(0, 2 − total_deductions).
Caller takes min(heuristic_score, llm_grammar_score) and reports both for
trainer audit.

This module is intentionally pure / stateless / no external deps.
"""
import re
from typing import Tuple

# Words that are legitimately ALL-CAPS (anywhere in the sentence, including
# the first word) and should NOT count as improper. Extend this list if real
# false positives surface on submissions.
_ABBREV_ALLOWLIST = frozenset({
    # Pronouns / countries / unions
    "I", "USA", "UK", "US", "UN", "EU", "UAE", "USSR", "PRC", "DPRK",
    # International bodies
    "UNEP", "NASA", "OECD", "WHO", "WTO", "IMF", "ASEAN", "APEC", "NATO",
    "BRICS", "G20", "G7", "FIFA", "IOC", "WWF",
    # Media / press
    "BBC", "CNN", "MSNBC", "CNBC", "ABC", "NBC", "CBS",
    # Education / standards
    "PTE", "IELTS", "TOEFL", "MIT", "UCLA", "MBA", "ISO", "IEEE", "ANSI",
    "WCAG",
    # Tech / business
    "AI", "GDP", "CEO", "CTO", "COO", "CFO", "IBM", "AWS", "BMW", "HSBC",
    "KFC", "AT&T",
    "API", "CPU", "GPU", "RAM", "ROM", "URL", "URI", "JSON", "HTML", "CSS",
    "SQL", "HTTP", "HTTPS", "FTP", "USB", "PDF", "GPS", "DNA", "RNA",
})

_MAX = 2


def grammar_heuristic(text: str) -> Tuple[int, dict]:
    """Return (score, findings).

    score    — integer 0–2 (max minus deductions, floored at 0).
    findings — dict with per-rule details PLUS character positions for the
               highlight builder. Position fields:
                 extra_space_ranges      — list of (start, end) byte offsets
                                           covering each run of 2+ spaces
                                           (the whole run, not just the extras)
                 improper_caps_ranges    — list of (start, end, word) for each
                                           flagged ALL-CAPS word
                 initial_cap_position    — int byte offset of the first
                                           non-whitespace char, or None
                 terminal_position       — int byte offset where the missing
                                           terminal punctuation should be
                                           inserted (end of body.rstrip()),
                                           or None
    """
    findings: dict = {
        "extra_spaces": 0,
        "extra_space_ranges": [],
        "missing_initial_cap": False,
        "initial_cap_position": None,
        "improper_caps": [],
        "improper_caps_ranges": [],
        "missing_terminal": False,
        "terminal_position": None,
    }
    body = text or ""
    if not body.strip():
        return 0, findings

    # Rule 1 — count extras beyond a single space within any run of 2+ spaces.
    for m in re.finditer(r" {2,}", body):
        findings["extra_spaces"] += len(m.group()) - 1
        findings["extra_space_ranges"].append((m.start(), m.end()))

    # Rule 2 — first non-whitespace character must be uppercase.
    stripped = body.lstrip()
    if stripped:
        first_pos = len(body) - len(stripped)
        if not stripped[0].isupper():
            findings["missing_initial_cap"] = True
            findings["initial_cap_position"] = first_pos

    # Rule 3 — improper ALL-CAPS anywhere in the response (Option B).
    # Sentence-start exemption removed: ALL-CAPS at sentence start (e.g.
    # "MAJOR athletic events…") is just as wrong as mid-sentence ALL-CAPS
    # in formal writing. Legitimate abbreviations are allowlisted above.
    for m in re.finditer(r"\b[A-Z]{3,}\b", body):
        if m.group() in _ABBREV_ALLOWLIST:
            continue
        findings["improper_caps"].append(m.group())
        findings["improper_caps_ranges"].append((m.start(), m.end(), m.group()))

    # Rule 4 — terminal punctuation.
    rstripped = body.rstrip()
    if rstripped and rstripped[-1] not in ".!?":
        findings["missing_terminal"] = True
        findings["terminal_position"] = len(rstripped)

    deduction = (
        findings["extra_spaces"]
        + (1 if findings["missing_initial_cap"] else 0)
        + len(findings["improper_caps"])
        + (1 if findings["missing_terminal"] else 0)
    )
    return max(0, _MAX - deduction), findings


def format_findings(findings: dict) -> str:
    """Compact one-line summary for inclusion in grammar.reasoning strings."""
    parts = []
    if findings.get("extra_spaces"):
        parts.append(f"extra_spaces={findings['extra_spaces']}")
    if findings.get("missing_initial_cap"):
        parts.append("missing_initial_cap")
    if findings.get("improper_caps"):
        parts.append(f"improper_caps={','.join(findings['improper_caps'])}")
    if findings.get("missing_terminal"):
        parts.append("missing_terminal_punctuation")
    return "; ".join(parts) if parts else "no findings"

"""Hybrid spelling checker for the writing scorers (SWT, WE, SST).

Pipeline (in order):
  1. pyspellchecker — deterministic dictionary lookup, catches every word
     not in its corpus. High recall, ~5 ms, zero cost.
  2. Passage filter — drop any candidate that appears in the source passage
     (case-insensitive). Passage words are professionally edited, so if the
     student spelled a "weird" word the same way as the passage, it's
     correct vocabulary parroting, not a typo.
  3. Claude Haiku 4.5 judge — only invoked when the filter leaves survivors.
     Filters out proper nouns, valid rare words, British/American variants,
     and tokenizer artifacts.

Returns a normalised result dict the scorer wires into the breakdown +
highlights. Pure function, no side effects, never raises.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import List, Tuple

from anthropic import Anthropic
from spellchecker import SpellChecker

log = logging.getLogger(__name__)

_anth_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
_spell = SpellChecker(language="en")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")

_JUDGE_SYSTEM = """You are a spelling judge for an English language test.

Inputs:
1. PASSAGE — clean reference text (may be empty).
2. SUMMARY — the student's writing.
3. CANDIDATES — words flagged by an automatic spell-checker.

For each candidate, decide if it is actually a spelling mistake in the summary.

A word IS a real spelling mistake if and only if it is intended as a standard English word AND spelled wrong.

A word is NOT a spelling mistake if any of these apply:
- Proper noun (person, place, brand, organisation), even if it appears lowercased.
- Valid English word not in the dictionary (rare, technical, compound).
- Valid British or American variant (e.g. programme, colour, organisation).
- Tokenizer artifact (e.g. "didn" or "t" from "didn't"; hyphenated compounds split oddly).
- Appears in the PASSAGE spelled the same way.

Return strict JSON only, no prose:
{"verdicts":[{"word":"X","is_misspelled":true,"correction":"Y"} OR {"word":"X","is_misspelled":false,"correction":null}, ...]}"""


def _find_offsets(word: str, body: str) -> List[Tuple[int, int]]:
    """Whole-word, case-insensitive offsets of ``word`` in ``body``."""
    try:
        pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
        return [(m.start(), m.end()) for m in pattern.finditer(body)]
    except re.error:
        return []


def _call_claude_judge(passage: str, body: str, candidates: List[str]) -> List[dict]:
    user_msg = (
        f"PASSAGE:\n{passage or '(none)'}\n\n"
        f"SUMMARY:\n{body}\n\n"
        f"CANDIDATES:\n" + "\n".join(candidates) +
        "\n\nReply with ONLY the JSON object."
    )
    resp = _anth_client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=2000,
        system=_JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = (resp.content[0].text if resp.content else "{}").strip()
    if raw.startswith("```"):
        # tolerate fenced JSON
        parts = raw.split("```")
        raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw
    parsed = json.loads(raw)
    return parsed.get("verdicts", []) or []


def check_spelling(text: str, passage: str = "") -> dict:
    """Run pyspellchecker + passage filter + Claude judge on ``text``.

    Returns:
        {
          "mistakes":  [{"word", "correction", "offsets": [(start,end),...]}, ...],
          "candidates_raw":      [str, ...],   # what pyspell flagged (lowercased)
          "candidates_filtered": [str, ...],   # after passage filter
          "scorer":   "hybrid" | "pyspell-only" | "no-op",
          "latency_ms": int,
          "warning_code": optional str if Claude was unreachable
        }
    Never raises.
    """
    t0 = time.time()
    body = text or ""
    if not body.strip():
        return {
            "mistakes": [], "candidates_raw": [], "candidates_filtered": [],
            "scorer": "no-op", "latency_ms": 0,
        }

    words = _WORD_RE.findall(body)
    word_set = set(w.lower() for w in words)
    try:
        raw_unknown = sorted(_spell.unknown(word_set))
    except Exception as exc:  # extremely defensive — should never happen
        log.warning("[SPELL] pyspellchecker error: %s", exc)
        return {
            "mistakes": [], "candidates_raw": [], "candidates_filtered": [],
            "scorer": "no-op", "latency_ms": int((time.time() - t0) * 1000),
            "warning_code": "pyspellchecker_unavailable",
        }

    passage_words = set(w.lower() for w in _WORD_RE.findall(passage or ""))
    filtered = [c for c in raw_unknown if c not in passage_words]

    if not filtered:
        return {
            "mistakes": [], "candidates_raw": raw_unknown, "candidates_filtered": [],
            "scorer": "hybrid", "latency_ms": int((time.time() - t0) * 1000),
        }

    try:
        verdicts = _call_claude_judge(passage or "", body, filtered)
    except Exception as exc:
        # Claude unreachable → treat all filtered candidates as misspellings
        # (recall is preserved, precision drops; trainer-visible warning).
        log.warning("[SPELL] Claude judge unavailable, falling back to raw pyspell: %s", exc)
        return {
            "mistakes": [
                {"word": c, "correction": None, "offsets": _find_offsets(c, body)}
                for c in filtered
            ],
            "candidates_raw": raw_unknown,
            "candidates_filtered": filtered,
            "scorer": "pyspell-only",
            "latency_ms": int((time.time() - t0) * 1000),
            "warning_code": "spelling_llm_unavailable",
        }

    mistakes = []
    for v in verdicts:
        if not v.get("is_misspelled"):
            continue
        word = v.get("word")
        if not word:
            continue
        mistakes.append({
            "word": word,
            "correction": v.get("correction"),
            "offsets": _find_offsets(word, body),
        })

    return {
        "mistakes": mistakes,
        "candidates_raw": raw_unknown,
        "candidates_filtered": filtered,
        "scorer": "hybrid",
        "latency_ms": int((time.time() - t0) * 1000),
    }


def format_spelling_reasoning(mistakes: List[dict]) -> str:
    """Clean human-readable summary for the trainer-facing reasoning string."""
    if not mistakes:
        return "No spelling mistakes."
    if len(mistakes) == 1:
        m = mistakes[0]
        corr = f" → {m['correction']}" if m.get("correction") else ""
        return f"1 spelling mistake: {m['word']}{corr}."
    parts = []
    for m in mistakes:
        corr = f" → {m['correction']}" if m.get("correction") else ""
        parts.append(f"{m['word']}{corr}")
    return f"{len(mistakes)} spelling mistakes: " + ", ".join(parts) + "."

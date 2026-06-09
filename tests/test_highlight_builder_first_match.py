"""Regression tests for `services.highlight_builder._first_match` — pins the
Option D fix shipped 2026-06-09 (Nimisha report #108).

Background: the legacy `_first_match` did naive `body.find(needle)` to map LLM
mistake quotes to character positions. For short quotes like 'th' the first
substring match was inside an unrelated longer word ('that', 'with', 'the')
rather than the standalone token the LLM was actually flagging.

Concrete case from RDS attempt_answers.id=5178 (user 3, qid 11421, SWT):
  LLM returned grammar.mistakes = [{quote: 'th', correction: 'the'}]
  Body contained "...choices that align with..." and later "...joy in th
  everyday moments...". Legacy code anchored the highlight to character 254
  (inside 'that'). The actual typo lives at character 410.

These tests pin the fix:
  1. Short quote inside multiple longer words ⇒ standalone match wins
  2. Case-only mistake (lowercase 'english' typo, uppercase 'English' present)
     ⇒ the lowercase typo is picked even though the capitalised form appears
     first
  3. Multi-word LLM quotes still use substring fallback (no regression)
  4. Punctuation-bracketed quotes still use substring fallback
  5. Empty / missing inputs return None as before
  6. End-to-end build_highlights returns the corrected position via the
     public entry point
"""
from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret")

from services.highlight_builder import _first_match, build_highlights


# Nimisha's actual response (attempt_answers.id=5178, qid=11421)
NIMISHA_BODY = (
    "Simplicity is not about deprivation or austerity; it's about focusing on "
    "what truly matters and eliminating unnecessary clutter from our lives, "
    "and simplicity does not require a radical overhaul of our lives; it's "
    "about making small, intentional choices that align with our values and "
    "priorities, so it invites us to declutter, be mindful, and practice "
    "gratitude, ultimately allowing us to find greater joy in th everyday "
    "moments and appreciate the beauty in life's simplicity."
)


def test_short_quote_anchors_to_standalone_word_not_substring():
    """The reported bug. 'th' must map to the standalone token at pos 410,
    not the 'th' inside 'that'/'with'/'the'."""
    rng = _first_match(NIMISHA_BODY, "th", used=set(), correction="the")
    assert rng is not None
    s, e = rng
    assert NIMISHA_BODY[s:e] == "th"
    # Must be standalone — i.e. surrounding chars are NOT alphabetic
    assert s == 0 or not NIMISHA_BODY[s - 1].isalpha()
    assert e == len(NIMISHA_BODY) or not NIMISHA_BODY[e].isalpha()
    # And specifically the right position
    assert (s, e) == (410, 412)


def test_short_quote_an_does_not_match_inside_another():
    body = "another animal entered an enclosure"
    # 'an' inside 'another' starts at 0; the standalone 'an' starts at 23.
    rng = _first_match(body, "an", used=set(), correction="a")
    assert rng is not None
    s, e = rng
    assert body[s:e] == "an"
    assert s == 23


def test_single_letter_quote_anchors_to_standalone():
    body = "I think I am here"
    rng = _first_match(body, "I", used=set(), correction="I")
    # First standalone 'I' is at position 0.
    assert rng == (0, 1)


def test_case_only_typo_lowercase_wins_over_capitalised_form():
    """LLM returns quote='english', correction='English'. Body has the
    capitalised 'English' earlier and the lowercase typo 'english' later.
    Case-sensitive WB matches only the lowercase form ⇒ that's what's
    picked, even though it's further along the string."""
    body = "American English is widely used. The variety of english differs."
    rng = _first_match(body, "english", used=set(), correction="English")
    assert rng is not None
    s, e = rng
    # Lowercase 'english' starts at index 47 in the body above.
    assert body[s:e] == "english"
    assert s == body.index("english")  # the lowercase occurrence
    # And it isn't the 'English' at offset 9.
    assert s != 9


def test_case_insensitive_fallback_when_no_case_sensitive_match():
    body = "She said HELLO to me."
    # LLM returns the lowercased form; case-sensitive WB has zero hits, so
    # we fall back to case-insensitive WB and still find it.
    rng = _first_match(body, "hello", used=set(), correction="hello")
    assert rng is not None
    s, e = rng
    assert body[s:e].lower() == "hello"


def test_multi_word_quote_still_resolves_via_substring_fallback():
    """Multi-word LLM quotes contain spaces, so word boundaries naturally
    anchor them. They must still resolve correctly."""
    body = "The cat sat on the mat and the dog barked."
    rng = _first_match(body, "sat on the", used=set(), correction="sat upon the")
    assert rng is not None
    s, e = rng
    assert body[s:e] == "sat on the"


def test_punctuation_bracketed_quote_falls_through_to_substring():
    """If the LLM quotes punctuation around a token (e.g. ',however') the
    word-boundary anchors can't apply on the comma side; substring fallback
    must still find it."""
    body = "I went home,however I came back."
    rng = _first_match(body, ",however", used=set(), correction=", however")
    assert rng is not None
    s, e = rng
    assert body[s:e] == ",however"


def test_used_ranges_are_skipped():
    """If a candidate is already claimed by an earlier highlight, _first_match
    must move past it."""
    body = "the cat the dog the bird"
    used = {(0, 3)}  # claim the first 'the'
    rng = _first_match(body, "the", used=used, correction="a")
    assert rng is not None
    assert rng[0] > 3  # picked the second one


def test_empty_needle_returns_none():
    assert _first_match("a body", "", used=set()) is None
    assert _first_match("a body", "   ", used=set()) is None


def test_empty_body_returns_none():
    assert _first_match("", "needle", used=set()) is None


def test_no_match_returns_none():
    assert _first_match("the cat sat", "xyz", used=set()) is None


# ── End-to-end via build_highlights ─────────────────────────────────────────


def test_build_highlights_end_to_end_for_nimisha_case():
    """The reported case at the public entry point: LLM grammar mistake
    'th'→'the' must land at the standalone token (pos 410), not inside
    'that' (pos 254)."""
    grammar = [{"quote": "th", "correction": "the", "reason": "Spelling error"}]
    h = build_highlights(NIMISHA_BODY, {}, spelling_quotes=[], grammar_quotes=grammar)
    assert len(h) == 1
    item = h[0]
    assert item["word"] == "th"
    assert (item["start"], item["end"]) == (410, 412)
    assert NIMISHA_BODY[item["start"]:item["end"]] == "th"


def test_build_highlights_does_not_drop_correctly_positioned_quotes():
    """Sanity check — a normal multi-character typo with a unique body
    position still lands where the legacy code would have put it."""
    body = "Manatees are mantees of the sea."
    spelling = [{"word": "mantees", "correction": "manatees"}]
    h = build_highlights(body, {}, spelling_quotes=spelling, grammar_quotes=[])
    assert len(h) == 1
    assert body[h[0]["start"]:h[0]["end"]] == "mantees"

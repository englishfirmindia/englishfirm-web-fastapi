"""Regression tests for `services.grammar_heuristic.grammar_heuristic` —
pins behaviour after the 2026-06-19 extension of Rule 6.

Background
----------
Rule 6 ("missing space after punctuation") shipped with regex `[,;:]` only,
so "First.Second" / "Hello!World" / "Really?Yes" were silently accepted by
the heuristic even though the docstring claimed all five terminator chars
were covered. This shipped today extends the regex to `[,;:.!?]` so all
five marks deduct a point when followed by a letter with no space.

The heuristic is shared by SWT, WE, and SST scoring (`ai_scorer.py`) and
applies uniformly across practice / sectional / mock — so these tests
double as the regression net for every writing question type in every mode.
"""
import pytest

from services.grammar_heuristic import grammar_heuristic, format_findings


# ── Rule 6 NEW behaviour: missing space after . ! ? ──────────────────────


def test_missing_space_after_period_is_flagged():
    """Pins the 2026-06-19 fix — the user-reported case."""
    score, f = grammar_heuristic("First sentence.Second sentence.")
    assert len(f["missing_space_after_punct"]) == 1, (
        "Period followed by letter must deduct 1 — this is the whole "
        "point of the 2026-06-19 ship"
    )
    s, e, original, suggested = f["missing_space_after_punct"][0]
    assert original == ".S"
    assert suggested == ". S"
    # Score: max(0, 2 - 1) = 1
    assert score == 1


def test_missing_space_after_exclamation_is_flagged():
    score, f = grammar_heuristic("Hello!World it is.")
    assert len(f["missing_space_after_punct"]) == 1
    assert f["missing_space_after_punct"][0][2] == "!W"


def test_missing_space_after_question_mark_is_flagged():
    score, f = grammar_heuristic("Really?Yes it is.")
    assert len(f["missing_space_after_punct"]) == 1
    assert f["missing_space_after_punct"][0][2] == "?Y"


def test_multiple_missing_spaces_stack_deductions():
    """Each missing-space site deducts 1, capped at score=0."""
    score, f = grammar_heuristic("One.Two.Three!Four?Five.")
    # 4 sites: .T .T !F ?F (the trailing . before end-of-string is fine
    # — handled by Rule 4 terminal check, not Rule 6).
    assert len(f["missing_space_after_punct"]) == 4
    assert score == 0   # 2 - 4 = -2 → floored at 0


# ── Rule 6 existing behaviour preserved (no regression) ──────────────────


def test_missing_space_after_comma_still_flagged():
    """Existing comma rule must not regress."""
    score, f = grammar_heuristic("emotion,support is key.")
    assert len(f["missing_space_after_punct"]) == 1
    assert f["missing_space_after_punct"][0][2] == ",s"


def test_missing_space_after_semicolon_still_flagged():
    score, f = grammar_heuristic("First;second is wrong.")
    assert len(f["missing_space_after_punct"]) == 1
    assert f["missing_space_after_punct"][0][2] == ";s"


def test_missing_space_after_colon_still_flagged():
    score, f = grammar_heuristic("Note:value is wrong.")
    assert len(f["missing_space_after_punct"]) == 1
    assert f["missing_space_after_punct"][0][2] == ":v"


# ── False-positive guards: numerals MUST NOT trip ────────────────────────


def test_decimal_numbers_not_flagged():
    """1.5 / 1,000 must NEVER be flagged — `[A-Za-z]` lookahead excludes
    digits. Critical because PTE essays cite statistics frequently."""
    score, f = grammar_heuristic("The figure is 1.5 million.")
    assert len(f["missing_space_after_punct"]) == 0


def test_thousands_separator_not_flagged():
    score, f = grammar_heuristic("Over 10,000 people attended.")
    assert len(f["missing_space_after_punct"]) == 0


def test_complex_numeric_passage_not_flagged():
    """A passage full of numerals should produce zero punctuation
    findings even though it has many `,` and `.` characters."""
    score, f = grammar_heuristic(
        "The GDP grew from $1.5 trillion to $10,000 billion by 2025."
    )
    assert len(f["missing_space_after_punct"]) == 0


def test_punctuation_at_end_of_string_not_flagged():
    """The terminal period of a well-formed sentence must not trip Rule 6
    (since there's no letter after it). Rule 4 handles missing-terminal
    case separately."""
    score, f = grammar_heuristic("This sentence ends properly.")
    assert len(f["missing_space_after_punct"]) == 0
    # And no other deductions on a clean sentence.
    assert score == 2


def test_punctuation_followed_by_space_not_flagged():
    """The happy path — a properly spaced sentence."""
    score, f = grammar_heuristic(
        "First sentence. Second sentence. Third sentence!"
    )
    assert len(f["missing_space_after_punct"]) == 0
    assert score == 2


def test_punctuation_followed_by_digit_not_flagged():
    """Section/chapter references like 'see Section.3' would be weird,
    but the rule should only fire on letters — digit lookahead exemption."""
    score, f = grammar_heuristic("In year 2025.7 we saw growth.")
    assert len(f["missing_space_after_punct"]) == 0


# ── Score-floor semantics ────────────────────────────────────────────────


def test_score_floors_at_zero_under_many_violations():
    """Score must never go negative — even a sentence with five missing
    spaces should report score = 0, not -3."""
    score, f = grammar_heuristic("A.B.C.D.E.F.G")
    # 6 missing-space-after-period findings (each . between letters)
    assert len(f["missing_space_after_punct"]) == 6
    # Plus missing-terminal-punctuation
    assert f["missing_terminal"] is True
    assert score == 0


def test_clean_text_scores_full_marks():
    """A grammatically clean passage must score exactly 2 with no
    findings — the floor for everything else."""
    score, f = grammar_heuristic(
        "Technology has reshaped daily life. Many tasks now happen online. "
        "This shift creates both opportunities and challenges."
    )
    assert score == 2
    assert f["missing_space_after_punct"] == []
    assert f["space_before_comma"] == []
    assert f["extra_spaces"] == 0
    assert f["missing_terminal"] is False
    assert f["missing_initial_cap"] is False


def test_empty_text_returns_zero():
    """Empty or whitespace-only input scores 0, not 2 — the student
    submitted nothing."""
    assert grammar_heuristic("")[0] == 0
    assert grammar_heuristic("   ")[0] == 0


# ── Findings shape — corrections panel + highlight builder rely on it ────


def test_finding_tuple_shape_unchanged():
    """The highlight_builder + corrections panel rely on the 4-tuple
    shape (start, end, original, suggested). Pin so we never accidentally
    extend the tuple and break callers."""
    _, f = grammar_heuristic("First.Second.")
    finding = f["missing_space_after_punct"][0]
    assert len(finding) == 4
    s, e, original, suggested = finding
    assert isinstance(s, int) and isinstance(e, int)
    assert isinstance(original, str) and isinstance(suggested, str)
    assert e > s
    # Suggested fix should be original + an inserted space
    assert " " in suggested
    assert suggested.replace(" ", "") == original


def test_format_findings_includes_punct_count():
    """The compact one-liner that lands in grammar.reasoning must report
    the punctuation count when present, so trainer audit can see why
    the heuristic floored the LLM."""
    _, f = grammar_heuristic("Hi!World")
    summary = format_findings(f)
    assert "missing_space_after_punct=1" in summary


# ── Interaction with other rules — no double-counting ────────────────────


def test_period_at_end_of_text_only_counts_for_terminal_rule():
    """A well-terminated period at end of string fires Rule 4? No — Rule
    4 checks LAST char is `.!?`. A clean ".\nfoo" should hit Rule 6 (no
    space after dot, letter after) ONLY, not Rule 4."""
    _, f = grammar_heuristic("Foo.\nbar.")
    # \n is whitespace, so the . before \n has no letter immediately
    # after → no Rule 6 hit. Final `.` is the terminal → Rule 4 happy.
    assert len(f["missing_space_after_punct"]) == 0
    assert f["missing_terminal"] is False


def test_period_followed_by_letter_no_terminal_punct_double_hits():
    """Two violations in one short string: 'a.b' has no space after dot
    AND ends without proper terminal."""
    _, f = grammar_heuristic("a.b")
    assert len(f["missing_space_after_punct"]) == 1
    assert f["missing_terminal"] is True
    # Plus missing initial cap (lowercase 'a')
    assert f["missing_initial_cap"] is True


# ── Realistic PTE essay sample with the new violation pattern ────────────


def test_realistic_sst_passage_with_period_violations():
    """A realistic SST/SWT-style answer with the bug pattern the user
    reported — multiple sentences run together with no spaces."""
    text = (
        "The author argues that technology improves productivity.However, "
        "it also creates new problems.Workers need to adapt quickly!Schools "
        "must teach new skills early."
    )
    score, f = grammar_heuristic(text)
    # 3 violations: .H .W !S
    assert len(f["missing_space_after_punct"]) == 3
    # First is the period-after-productivity into "However"
    assert f["missing_space_after_punct"][0][2] == ".H"
    # Score: 2 - 3 = -1 → floored at 0
    assert score == 0


def test_realistic_we_essay_with_full_spacing_scores_full_marks():
    """A WE-style essay with proper spacing should NOT be penalised by
    Rule 6 — pins the no-false-positive promise."""
    text = (
        "Education is the foundation of progress. Schools, colleges, and "
        "universities all play a vital role. However, online learning is "
        "now equally important: it offers flexibility; it reaches remote "
        "students; and it lowers cost. We must invest in both formats."
    )
    score, f = grammar_heuristic(text)
    assert len(f["missing_space_after_punct"]) == 0, (
        f"False positives on clean essay: {f['missing_space_after_punct']}"
    )
    assert score == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

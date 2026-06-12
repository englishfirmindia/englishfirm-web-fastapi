"""Regression tests for services.scoring_health.is_row_failed.

Background: on 2026-06-12 Anish (user 108) finished sectional speaking
attempt 5630 with 32 questions scored. He saw the banner "12 of 32
questions couldn't be scored due to a temporary system issue." All 32
rows had scoring_status='complete' and a valid score — the false alarm
came from 12 rows carrying `scoring_warnings` with successful-fallback
codes (`pronunciation_fallback_azure`, `stimulus_whisper_fallback_used`)
that weren't in the allowlist. Those codes mean the primary backbone
was unavailable but a designed-in backup completed scoring normally
— same shape as `gpt4o_unavailable_used_claude` (which WAS allowlisted).

These tests pin:
  1. The three successful-fallback codes never trigger failed=True
     (the bug we just fixed).
  2. Genuine failure paths still trigger failed=True
     (scoring_status='failed', non-allowlisted warning, mix of allowed
     + non-allowed warnings).
  3. User-skip path (form-gate-floor) doesn't trigger failed=True even
     when warnings are present — gate did its job.
  4. Allowlist additions are wired through end-to-end via
     `collect_failed_question_ids` and `build_scoring_health`.
  5. Counterfactual capping (excl >= with) holds.
"""
from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret")

from types import SimpleNamespace

import pytest

from services.scoring_health import (
    build_scoring_health,
    collect_failed_question_ids,
    is_row_failed,
)


def _row(*, status="complete", scorer=None, warnings=None, question_id=1):
    """Build a fake attempt_answers row with just the fields is_row_failed reads."""
    result_json: dict = {}
    if scorer is not None:
        result_json["scorer"] = scorer
    if warnings is not None:
        result_json["scoring_warnings"] = warnings
    return SimpleNamespace(
        scoring_status=status,
        result_json=result_json,
        question_id=question_id,
    )


# ── Successful-fallback warnings must NOT count as failed ──────────────────


@pytest.mark.parametrize(
    "warning",
    [
        "gpt4o_unavailable_used_claude",
        "pronunciation_fallback_azure",
        "stimulus_whisper_fallback_used",
    ],
)
def test_successful_fallback_warnings_are_not_failures(warning):
    """The three designed-in fallback codes all mean scoring succeeded via
    a backup pathway. None should trip the "couldn't be scored" banner."""
    row = _row(warnings=[warning])
    assert is_row_failed(row) is False, (
        f"{warning!r} is a successful fallback, not a failure"
    )


def test_anish_rts_row_with_both_fallbacks_is_not_failure():
    """Aid 5463/5464 in the Anish report carried BOTH fallback warnings on
    the same row. Both being allowlisted means the row counts as scored."""
    row = _row(warnings=[
        "stimulus_whisper_fallback_used",
        "pronunciation_fallback_azure",
    ])
    assert is_row_failed(row) is False


# ── Genuine failures must still count ──────────────────────────────────────


def test_scoring_status_failed_is_always_failure():
    """Even with no warnings, status='failed' (reaper-flipped) means we
    couldn't score the row."""
    row = _row(status="failed", warnings=None)
    assert is_row_failed(row) is True


def test_unknown_warning_code_counts_as_failure():
    """Any warning string not in the allowlist is treated as a real failure
    — keeps us honest about novel error codes shipping unflagged."""
    row = _row(warnings=["transcription_failed"])
    assert is_row_failed(row) is True


def test_mixed_warnings_failure_wins():
    """If a row carries both an allowlisted fallback AND a genuine error
    code, the failure side wins — we can't score it."""
    row = _row(warnings=[
        "pronunciation_fallback_azure",   # ok
        "transcription_failed",            # real failure
    ])
    assert is_row_failed(row) is True


# ── User-fault & user-skip paths ───────────────────────────────────────────


@pytest.mark.parametrize(
    "warning",
    [
        "Empty response.",
        "content_off_topic",
        "essay_paragraph_count_cap",
        "sst_paragraph_count_cap",
    ],
)
def test_user_fault_warnings_are_not_failures(warning):
    """User-fault codes (legacy behaviour) must still pass through unchanged."""
    assert is_row_failed(_row(warnings=[warning])) is False


def test_form_zero_prefix_is_user_fault():
    """`Form-zero — 31 words is below the band` is per-attempt user-fault
    text and matched by prefix."""
    row = _row(warnings=["Form-zero — 31 words is below the band"])
    assert is_row_failed(row) is False


def test_form_gate_floor_scorer_overrides_warnings():
    """When the rubric gate caught bad input pre-scoring (`scorer=form-gate-floor`),
    we never consider the row failed — the gate did its job, the user gets
    their legitimate 0."""
    row = _row(scorer="form-gate-floor", warnings=["transcription_failed"])
    assert is_row_failed(row) is False


# ── Empty / defensive paths ────────────────────────────────────────────────


def test_none_row_is_not_failure():
    assert is_row_failed(None) is False


def test_empty_warnings_list_is_not_failure():
    assert is_row_failed(_row(warnings=[])) is False


def test_missing_warnings_field_is_not_failure():
    assert is_row_failed(_row()) is False


def test_non_string_warning_is_treated_as_failure():
    """Defensive: a non-string entry slips through the allowlist (it can't
    be matched) — treat it as failure so we surface the issue rather than
    silently dropping it."""
    row = _row(warnings=[{"odd": "shape"}])
    assert is_row_failed(row) is True


# ── End-to-end through collect_failed_question_ids + build_scoring_health ──


def test_anish_pattern_no_failures_after_fix():
    """Reproduce the exact warning-mix from Anish's attempt 5630 and verify
    the corrected behaviour: zero failures, no banner triggered."""
    rows = [
        _row(question_id=6162, warnings=["pronunciation_fallback_azure"]),
        _row(question_id=4019, warnings=["pronunciation_fallback_azure"]),
        _row(question_id=8080, warnings=["pronunciation_fallback_azure"]),
        _row(question_id=7487, warnings=["pronunciation_fallback_azure"]),
        _row(question_id=8131, warnings=["pronunciation_fallback_azure"]),
        _row(question_id=21276, warnings=["stimulus_whisper_fallback_used"]),
        _row(question_id=21246, warnings=["stimulus_whisper_fallback_used"]),
        _row(question_id=9054, warnings=["pronunciation_fallback_azure"]),
        _row(question_id=9680, warnings=["pronunciation_fallback_azure"]),
        _row(question_id=10921, warnings=["pronunciation_fallback_azure"]),
        _row(question_id=21510, warnings=[
            "stimulus_whisper_fallback_used",
            "pronunciation_fallback_azure",
        ]),
        _row(question_id=21714, warnings=[
            "stimulus_whisper_fallback_used",
            "pronunciation_fallback_azure",
        ]),
    ]
    assert collect_failed_question_ids(rows) == []


def test_mix_of_fallback_and_genuine_failure_surfaces_real_failure():
    """Sanity: when a row is genuinely failed, it bubbles up."""
    rows = [
        _row(question_id=1, warnings=["pronunciation_fallback_azure"]),       # ok
        _row(question_id=2, status="failed"),                                  # real
        _row(question_id=3, warnings=["transcription_failed"]),                # real
        _row(question_id=4, warnings=["stimulus_whisper_fallback_used"]),     # ok
    ]
    assert collect_failed_question_ids(rows) == [2, 3]


def test_build_scoring_health_caps_counterfactual_at_with_failures():
    """`score_excluding_failures` must never go below `score_with_failures`
    — protects against weighted-aggregation quirks telling the user "you
    would have scored worse without the failures."
    """
    out = build_scoring_health(
        total_questions=10,
        failed_question_ids=[3, 7],
        score_with_failures=65,
        score_excluding_failures=55,   # backend produced a worse number
    )
    assert out["score_excluding_failures"] == 65   # capped up to with-failures
    assert out["score_with_failures"] == 65
    assert out["failed_count"] == 2
    assert out["failed_question_ids"] == [3, 7]


def test_build_scoring_health_passes_through_question_numbers():
    """1-based attempt positions are propagated so the banner can render
    'Affected questions: Q3, Q7' instead of internal qids."""
    out = build_scoring_health(
        total_questions=10,
        failed_question_ids=[101, 102],
        score_with_failures=50,
        score_excluding_failures=60,
        failed_question_numbers=[3, 7],
    )
    assert out["failed_question_numbers"] == [3, 7]
    assert out["score_excluding_failures"] == 60

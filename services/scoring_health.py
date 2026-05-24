"""Helpers for the post-attempt 'scoring health' summary surfaced on the
student feedback screen.

The student-visible message:
  "N of M questions couldn't be scored due to a temporary system issue.
   Affected questions: Q3, Q7, Q12.
   Your score is X. Without the N unscored question(s), you would have
   scored ~Y."

Rules for what counts as "our-fault failed":
  - attempt_answers.scoring_status == 'failed'                     (reaper-flipped)
  - OR attempt_answers.result_json.scoring_warnings contains at least
    one SYSTEM-fault code (LLM unavailable, transcription failed, etc.)

What does NOT count as failed:
  - Empty/blank submissions (scorer == 'form-gate-floor')
  - Form-zero rejections from the rubric gate (word-count out of band,
    paragraph-count out of band)
  - Off-topic content rejections (LLM said content=0)
  - "gpt4o_unavailable_used_claude" — the Claude fallback succeeded
  - Anything else that's the user's choice or a designed-in scoring decision

Those are the user's own input (or the rubric's correct verdict on it)
and we don't bail them out.

The counterfactual `score_excluding_failures` is always >= `score_with_failures`
because excluding zero-contribution rows from the denominator can only help.
"""
from typing import Iterable, List, Optional


# Scorer labels that indicate the scoring pipeline never actually ran
# because the input failed the rubric gate. These are USER-fault, not
# system-fault — the gate did its job correctly.
_USER_SKIP_SCORERS = {"form-gate-floor"}


# Warning codes / messages that are user-fault or designed-in scoring
# decisions, not system failures. Anything not in this list is treated
# as a real failure.
_USER_FAULT_WARNINGS = {
    "Empty response.",
    "content_off_topic",
    "essay_paragraph_count_cap",
    "sst_paragraph_count_cap",
    # Claude fallback path succeeded — scoring completed normally
    "gpt4o_unavailable_used_claude",
}


def _is_user_fault_warning(w) -> bool:
    """True if a `scoring_warnings` entry indicates user-fault (not a
    system failure). Form-zero strings are matched by prefix because
    they carry per-attempt context (`"Form-zero — 31 words is below…"`)."""
    if not isinstance(w, str):
        return False
    if w in _USER_FAULT_WARNINGS:
        return True
    if w.startswith("Form-zero —"):
        return True
    return False


def is_row_failed(row) -> bool:
    """True if an attempt_answers row counts as our-fault failed.

    Accepts a row object with `.scoring_status` and `.result_json` attrs.
    Defensive against None / missing fields.
    """
    if row is None:
        return False
    if (getattr(row, "scoring_status", None) or "").lower() == "failed":
        return True
    result = getattr(row, "result_json", None) or {}
    if not isinstance(result, dict):
        return False
    # User-skip path: gate caught empty / out-of-band input before
    # scoring ran. Not a system failure — they get a legitimate 0.
    if (result.get("scorer") or "") in _USER_SKIP_SCORERS:
        return False
    warnings = result.get("scoring_warnings")
    if not isinstance(warnings, list) or not warnings:
        return False
    # Has warnings — counts as system failure only if at least one
    # warning is NOT user-fault.
    return any(not _is_user_fault_warning(w) for w in warnings)


def collect_failed_question_ids(rows: Iterable) -> List[int]:
    """Pick out the question_ids of failed rows. Stable ordering preserved."""
    out: List[int] = []
    for r in rows or ():
        if is_row_failed(r) and getattr(r, "question_id", None) is not None:
            out.append(int(r.question_id))
    return out


def build_scoring_health(
    total_questions: int,
    failed_question_ids: List[int],
    score_with_failures: int,
    score_excluding_failures: int,
    per_section_failed: Optional[dict] = None,
    failed_question_numbers: Optional[List[int]] = None,
) -> dict:
    """Standard envelope returned to the frontend. Caps the counterfactual
    at >= score_with_failures so a quirk of weighted aggregation never tells
    the user "you would have scored worse.

    `failed_question_numbers` (optional) is the 1-based position of each
    failed question within the attempt — surfaced so the banner can render
    "Affected questions: Q3, Q7, Q12" instead of opaque internal q_ids.
    Same length and ordering as `failed_question_ids` when provided.
    """
    capped_excl = max(int(score_excluding_failures), int(score_with_failures))
    return {
        "total_questions":          int(total_questions),
        "failed_count":             len(failed_question_ids),
        "failed_question_ids":      list(failed_question_ids),
        "failed_question_numbers":  list(failed_question_numbers or []),
        "score_with_failures":      int(score_with_failures),
        "score_excluding_failures": capped_excl,
        "failed_per_section":       per_section_failed or {},
    }

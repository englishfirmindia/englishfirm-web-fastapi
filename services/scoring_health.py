"""Helpers for the post-attempt 'scoring health' summary surfaced on the
student feedback screen.

The student-visible message:
  "N of M questions couldn't be scored due to a temporary system issue.
   Your displayed score is conservative; the affected questions are marked
   with a yellow icon.
   Your score is X. Without the N unscored question(s), you would have
   scored ~Y."

Rules for what counts as "our-fault failed":
  - attempt_answers.scoring_status == 'failed'                     (reaper-flipped)
  - OR attempt_answers.result_json.scoring_warnings is non-empty   (scorer-degraded)

User-skipped questions (no attempt_answers row at all) are NOT counted as
failed — those are the user's own choice and we don't bail them out.

The counterfactual `score_excluding_failures` is always >= `score_with_failures`
because excluding zero-contribution rows from the denominator can only help.
"""
from typing import Iterable, List, Optional


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
    warnings = result.get("scoring_warnings") if isinstance(result, dict) else None
    return isinstance(warnings, list) and len(warnings) > 0


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
) -> dict:
    """Standard envelope returned to the frontend. Caps the counterfactual
    at >= score_with_failures so a quirk of weighted aggregation never tells
    the user "you would have scored worse."""
    capped_excl = max(int(score_excluding_failures), int(score_with_failures))
    return {
        "total_questions":          int(total_questions),
        "failed_count":             len(failed_question_ids),
        "failed_question_ids":      list(failed_question_ids),
        "score_with_failures":      int(score_with_failures),
        "score_excluding_failures": capped_excl,
        "failed_per_section":       per_section_failed or {},
    }

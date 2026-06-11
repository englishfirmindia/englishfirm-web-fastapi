"""Shared paging / ordering helper for the 22 practice question list endpoints.

The 22 routers (read_aloud, repeat_sentence, …, listening_wfd) each ship a
near-identical `GET /list` endpoint that filters `questions_from_apeuni`
rows, paginates them, and returns the page. Historically each router
ordered by `question_id ASC|DESC`. As of 2026-06-02 we add a third sort
mode — `recent` — that orders practiced questions first (most-recently
practiced at the top) followed by never-practiced questions in id order.

Practice-mode only: sectional and mock attempts do NOT bubble a question
up the recency list. We pull recency from `attempt_answers JOIN
practice_attempts` with `pa.module != 'mock' AND pa.question_type !=
'sectional'`. `user_question_attempts` is not used here because that
table mixes all three modes.

This helper is the single place that "recent" sort is implemented. Each
router calls `paginate_by_practice_recency()` when `sort == 'recent'`
and falls back to its existing id-ordered SQL pagination otherwise.

Public API:
  fetch_practice_recency_map(db, user_id, question_type) → {qid: ts}
  paginate_by_practice_recency(db, query, user_id, question_type,
                               page, limit) → (rows, total, recency_map)
"""
from __future__ import annotations

from datetime import datetime
from typing import Tuple, List, Dict

from sqlalchemy import func
from sqlalchemy.orm import Session, Query

from db.models import AttemptAnswer, PracticeAttempt, QuestionFromApeuni


# Source-of-truth alias map for question_type strings. Different layers
# of the app historically named the same question type differently:
# the submit handler stores one canonical (e.g. "listening_sst" in
# `attempt_answers.question_type`) while the matching `/list` endpoint
# passes a legacy descriptive name (e.g. "summarize_spoken_text"). When
# the strings don't match, the "Done" / "practiced" badge silently
# returns no rows — questions stay flagged "New" forever on the practice
# list even after the user submits them. Same root cause as the 7-type
# audit on 2026-06-09.
#
# Strategy: every drift pair below maps BOTH directions to the same
# tuple. The helpers below expand the lookup IN-clause via this map, so
# the badge lookup matches whichever string the submit OR list side
# happens to pass. Identity-fallback for any unlisted type.
#
# This is the SINGLE SOURCE OF TRUTH for these aliases. routers/user.py
# imports from here — do NOT duplicate the table there.
QUESTION_TYPE_ALIASES: Dict[str, Tuple[str, ...]] = {
    # Reading FIB drag-drop — original alias (2026-04 era)
    "reading_fib":            ("reading_fib", "reading_drag_and_drop"),
    "reading_drag_and_drop":  ("reading_fib", "reading_drag_and_drop"),
    # Speaking — Respond To a Situation (legacy "ptea_" prefix from
    # the mobile app naming)
    "respond_to_situation":   ("respond_to_situation", "ptea_respond_situation"),
    "ptea_respond_situation": ("respond_to_situation", "ptea_respond_situation"),
    # Reading MCQs — legacy generic names without module prefix
    "reading_mcs":            ("reading_mcs", "mcq_single"),
    "mcq_single":             ("reading_mcs", "mcq_single"),
    "reading_mcm":            ("reading_mcm", "mcq_multiple"),
    "mcq_multiple":           ("reading_mcm", "mcq_multiple"),
    # Listening SST — descriptive name used by /list, shortened by submit
    "listening_sst":          ("listening_sst", "summarize_spoken_text"),
    "summarize_spoken_text":  ("listening_sst", "summarize_spoken_text"),
    # Listening MCQs — same shortening-drift as reading MCQs
    "listening_mcs":          ("listening_mcs", "listening_mcq_single"),
    "listening_mcq_single":   ("listening_mcs", "listening_mcq_single"),
    "listening_mcm":          ("listening_mcm", "listening_mcq_multiple"),
    "listening_mcq_multiple": ("listening_mcm", "listening_mcq_multiple"),
    # Listening HIW — descriptive name vs shortened
    "listening_hiw":          ("listening_hiw", "highlight_incorrect_words"),
    "highlight_incorrect_words": ("listening_hiw", "highlight_incorrect_words"),
}

# Internal alias for backwards-compat with prior private name. New code
# should import QUESTION_TYPE_ALIASES (no underscore) since it's now a
# public, shared constant.
_QUESTION_TYPE_ALIASES = QUESTION_TYPE_ALIASES


def practiced_questions_subq(
    db: Session, user_id: int, question_type: str
):
    """SQL subquery of qids the user has practised IN PRACTICE MODE.

    Use with `.in_()` / `.not_in_()` on QuestionFromApeuni.question_id when
    applying the "Done" / "New" filter to a list endpoint.

    Replaces the prior pattern of selecting from `user_question_attempts`
    (which counts all modes, including mock and sectional). UQA-based Done
    diverged from the recency map (practice-mode only) — questions only
    attempted via mock/sectional appeared in Done with a "Never practiced"
    header, and the last-answer endpoint correctly returned null for them.
    Aligning Done to practice-mode-only fixes that mismatch AND implicitly
    hides any orphan UQA rows (UQA without matching attempt_answers).
    """
    types = _QUESTION_TYPE_ALIASES.get(question_type, (question_type,))
    return (
        db.query(AttemptAnswer.question_id)
        .join(PracticeAttempt, AttemptAnswer.attempt_id == PracticeAttempt.id)
        .filter(
            PracticeAttempt.user_id == user_id,
            PracticeAttempt.filter_type == "practice",
            AttemptAnswer.scoring_status != "cleared",
            AttemptAnswer.question_type.in_(types),
        )
        .distinct()
        .subquery()
    )


def practiced_question_ids_in(
    db: Session, user_id: int, question_type: str, restrict_to: List[int]
) -> set:
    """Return the subset of `restrict_to` qids the user has practised
    in practice mode. Used for per-page enrichment so each returned row
    can be tagged `practiced: true/false` without re-querying UQA.
    """
    if not restrict_to:
        return set()
    types = _QUESTION_TYPE_ALIASES.get(question_type, (question_type,))
    rows = (
        db.query(AttemptAnswer.question_id)
        .join(PracticeAttempt, AttemptAnswer.attempt_id == PracticeAttempt.id)
        .filter(
            PracticeAttempt.user_id == user_id,
            PracticeAttempt.filter_type == "practice",
            AttemptAnswer.scoring_status != "cleared",
            AttemptAnswer.question_type.in_(types),
            AttemptAnswer.question_id.in_(restrict_to),
        )
        .distinct()
        .all()
    )
    return {r[0] for r in rows}


def fetch_practice_recency_map(
    db: Session, user_id: int, question_type: str
) -> Dict[int, datetime]:
    """Returns {question_id: most_recent_practice_submit_at} for this user
    in PRACTICE mode only.

    Excludes mock + sectional attempts so the recency list reflects only
    deliberate practice activity, not graded-context exposure.
    """
    types = _QUESTION_TYPE_ALIASES.get(question_type, (question_type,))
    rows = (
        db.query(
            AttemptAnswer.question_id,
            func.max(AttemptAnswer.submitted_at).label("ts"),
        )
        .join(PracticeAttempt, AttemptAnswer.attempt_id == PracticeAttempt.id)
        .filter(
            PracticeAttempt.user_id == user_id,
            PracticeAttempt.module != "mock",
            PracticeAttempt.question_type != "sectional",
            AttemptAnswer.question_type.in_(types),
        )
        .group_by(AttemptAnswer.question_id)
        .all()
    )
    return {r.question_id: r.ts for r in rows}


def paginate_by_practice_recency(
    db: Session,
    filtered_query: Query,
    user_id: int,
    question_type: str,
    page: int,
    limit: int,
) -> Tuple[List[QuestionFromApeuni], int, Dict[int, datetime]]:
    """Sort the post-filter result by (recency DESC NULLS LAST, id ASC)
    and return the requested page.

    Strategy:
      1. Fetch the matching question_ids ID-only (cheap; avoids loading
         large content_json fields for rows we'll discard).
      2. Sort the ID list in Python using the recency map: practiced
         first (DESC), then never-practiced (ASC by id).
      3. Slice the page.
      4. Load the full QuestionFromApeuni rows for the page only.
      5. Return rows in the sorted order.

    For 22 question types with ≤~600 questions each, the ID-only fetch
    is sub-50ms and the Python sort is trivial. Adding SQL-side
    LEFT JOIN ordering would be marginally faster but would require
    every router to know about the recency subquery shape — not worth
    the per-router complexity at our scale.

    Returns:
      page_rows: ordered list of QuestionFromApeuni for this page
      total: count of all matching questions (across all pages)
      recency_map: {qid: ts} — passed back so the caller can populate
                   `last_practiced_at` in the response without re-querying
    """
    recency = fetch_practice_recency_map(db, user_id, question_type)

    all_ids: List[int] = [
        r[0] for r in filtered_query.with_entities(
            QuestionFromApeuni.question_id
        ).all()
    ]

    def _sort_key(qid: int):
        ts = recency.get(qid)
        if ts is not None:
            # Practiced: bucket 0 → sorts before unpracticed.
            # Negative microsecond timestamp → most-recent first.
            # Tie-break by qid for determinism when two attempts share a ms.
            return (0, -int(ts.timestamp() * 1_000_000), qid)
        # Never practiced: bucket 1 → sorts after all practiced. id asc.
        return (1, 0, qid)

    all_ids.sort(key=_sort_key)
    total = len(all_ids)

    start = (page - 1) * limit
    end = start + limit
    page_ids = all_ids[start:end]
    if not page_ids:
        return [], total, recency

    rows_by_id = {
        q.question_id: q
        for q in db.query(QuestionFromApeuni)
        .filter(QuestionFromApeuni.question_id.in_(page_ids))
        .all()
    }
    page_rows = [rows_by_id[qid] for qid in page_ids if qid in rows_by_id]
    return page_rows, total, recency


def iso(ts: datetime | None) -> str | None:
    """Render a datetime as an ISO 8601 UTC string for the response payload.
    Frontend uses this to compute Today/Yesterday/This week/Earlier
    bucket headers in the user's local timezone.
    """
    if ts is None:
        return None
    if ts.tzinfo is None:
        # Assume naive datetimes from RDS are UTC.
        return ts.isoformat() + "Z"
    return ts.isoformat()

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


# Mapping from the question_type stored in `questions_from_apeuni` to the
# set of question_types that may appear in `attempt_answers` for that
# question. Defaults to identity when not aliased. Mirrors the table in
# routers/user.py:_QUESTION_TYPE_ALIASES — keep these two in sync.
_QUESTION_TYPE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "reading_fib":           ("reading_fib", "reading_drag_and_drop"),
    "reading_drag_and_drop": ("reading_fib", "reading_drag_and_drop"),
}


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

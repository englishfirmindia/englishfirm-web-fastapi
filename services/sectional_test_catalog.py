"""
Sectional Test Catalog
======================
Reads the locked question set for each (module, test_number) slot from
`sectional_test_questions`. Source of truth — once seeded, every user
gets the same canonical questions per test, every redo included.

Replaces the per-call `random.sample` selection that previously lived
inside each of the four sectional services. The actual seed lives in
scripts/migrations/2026-05-12_seed_sectional_test_questions.py.

Canonical ordering
------------------
PTE Academic groups all instances of a given task type together
(SWT × 2 first, then FIB-DD × 6, etc.). A subsequent reseed once
appended top-up questions to the END of the row — Nimisha saw a
second SWT pop up at question 23 instead of question 2. To make the
read path resilient against any past or future ordering drift in the
stored row, `get_locked_question_ids` now sorts by canonical task
order before returning. The stored row is also rewritten in-place by
the one-off `2026-05-20_reorder_sectional_seeds.py` migration so the
DB is consistent with what callers see.
"""

from fastapi import HTTPException
from sqlalchemy.orm import Session

from db.models import SectionalTestQuestions, QuestionFromApeuni


# Canonical task order per module — every test of `module=M` must serve
# questions in this order (all of task[0], then all of task[1], …). Mirrors
# the per-module STRUCTURE constants in the *_sectional_service files; kept
# local here to avoid a cyclic import. If those structures change, update
# the list below in lockstep.
_TASK_ORDER: dict[str, tuple[str, ...]] = {
    "reading": (
        "summarize_written_text",
        "reading_fib_drop_down",
        "mcq_multiple",
        "reorder_paragraphs",
        "reading_drag_and_drop",
        "mcq_single",
        "listening_hcs",
        "highlight_incorrect_words",
    ),
    "writing": (
        "summarize_written_text",
        "write_essay",
        "summarize_spoken_text",
        "listening_wfd",
    ),
    "speaking": (
        "read_aloud",
        "repeat_sentence",
        "describe_image",
        "retell_lecture",
        "answer_short_question",
        "summarize_group_discussion",
        "ptea_respond_situation",
    ),
    "listening": (
        "repeat_sentence",
        "retell_lecture",
        "summarize_group_discussion",
        "answer_short_question",
        "summarize_spoken_text",
        "listening_mcq_multiple",
        "listening_fib",
        "listening_hcs",
        "listening_smw",
        "highlight_incorrect_words",
        "listening_mcq_single",
        "listening_wfd",
    ),
}


def canonical_order(db: Session, module: str, question_ids: list[int]) -> list[int]:
    """Return `question_ids` reordered so all questions of each task type
    are grouped, with task-type groups appearing in `_TASK_ORDER[module]`.

    Within each group the original relative order is preserved (stable
    sort) so a test that had Q1=swt-A, Q2=swt-B before keeps that
    SWT-pair order after reordering. Unknown task types (e.g. a slug
    that's no longer in `_TASK_ORDER`) sort to the end stably, so they
    never disappear silently.
    """
    if not question_ids:
        return []
    order = _TASK_ORDER.get(module)
    if not order:
        return list(question_ids)
    rows = (
        db.query(QuestionFromApeuni.question_id, QuestionFromApeuni.question_type)
        .filter(QuestionFromApeuni.question_id.in_(question_ids))
        .all()
    )
    type_by_qid = {r.question_id: r.question_type for r in rows}
    rank = {t: i for i, t in enumerate(order)}
    # Stable sort by (task-rank, original index). Unknown / deleted types
    # land after the known ones rather than getting filtered out — the
    # service layer downstream already handles missing questions.
    indexed = list(enumerate(question_ids))
    indexed.sort(
        key=lambda pair: (
            rank.get(type_by_qid.get(pair[1], ""), len(order)),
            pair[0],
        )
    )
    return [qid for _, qid in indexed]


def get_locked_question_ids(db: Session, module: str, test_number: int) -> list[int]:
    """Return the canonical, ordered question_ids for (module, test_number).

    Raises 404 if the slot hasn't been seeded — the catalog is expected
    to cover test_number ∈ [1, 40] for each of the four modules after the
    one-off seed runs in production.

    Reorders the stored row defensively at read time so legacy rows
    (where a reseed appended top-ups at the end instead of merging in
    task-order position) still serve questions in the correct PTE order.
    """
    row = (
        db.query(SectionalTestQuestions)
        .filter_by(module=module, test_number=test_number)
        .first()
    )
    if not row or not row.question_ids:
        raise HTTPException(
            status_code=404,
            detail=f"Test {test_number} for {module} sectional has not been seeded yet.",
        )
    return canonical_order(db, module, list(row.question_ids))

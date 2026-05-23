"""
Mock Test Catalog
=================
Reads the locked question set for each `test_number` (1..40) from
`mock_test_questions`. Source of truth — every user starting mock
test N gets the same 65 questions.

Symmetric with `sectional_test_catalog`, with two differences:
  1. Single-key (test_number) — mock isn't per-module.
  2. Re-seedable — the seed script supports `--reseed` for refreshes
     when the question pool grows. Existing PracticeAttempt rows are
     unaffected because each attempt freezes its qids at start time.

Order in the stored array IS the in-test sequence (speaking → writing
→ reading → listening, per MOCK_STRUCTURE). The seed script writes it
that way; readers don't reorder.
"""

from fastapi import HTTPException
from sqlalchemy.orm import Session

from db.models import MockTestQuestions


def get_locked_mock_question_ids(db: Session, test_number: int) -> list[int]:
    """Return the canonical, ordered question_ids for mock `test_number`.

    Raises 404 if the slot hasn't been seeded. Production is expected to
    cover test_number ∈ [1, 40] after the seed migration runs.
    """
    row = (
        db.query(MockTestQuestions)
        .filter_by(test_number=test_number)
        .first()
    )
    if not row or not row.question_ids:
        raise HTTPException(
            status_code=404,
            detail=f"Mock test {test_number} has not been seeded yet.",
        )
    return list(row.question_ids)

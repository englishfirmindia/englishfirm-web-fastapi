"""
Sectional Test Catalog
======================
Reads the locked question set for each (module, test_number) slot from
`sectional_test_questions`. Source of truth — once seeded, every user
gets the same canonical questions per test, every redo included.

Replaces the per-call `random.sample` selection that previously lived
inside each of the four sectional services. The actual seed lives in
scripts/migrations/2026-05-12_seed_sectional_test_questions.py.
"""

from fastapi import HTTPException
from sqlalchemy.orm import Session

from db.models import SectionalTestQuestions


def get_locked_question_ids(db: Session, module: str, test_number: int) -> list[int]:
    """Return the canonical, ordered question_ids for (module, test_number).

    Raises 404 if the slot hasn't been seeded — the catalog is expected
    to cover test_number ∈ [1, 40] for each of the four modules after the
    one-off seed runs in production.
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
    return list(row.question_ids)

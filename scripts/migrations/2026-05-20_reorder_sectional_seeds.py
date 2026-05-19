"""One-off migration: reorder every `sectional_test_questions` row so the
`question_ids` array is sorted by canonical task order.

Background
----------
A previous reseed on 2026-05-19 fixed missing per-task counts by appending
top-up question IDs to the END of existing rows. The result: tests where
the second SWT (or second reorder paragraph, etc.) appeared at question
23 instead of question 2 — out of PTE task-grouping order. See the
investigation summary attached to the 2026-05-20 conversation.

This migration walks every row in `sectional_test_questions`, looks up
each question's `question_type` from `questions_from_apeuni`, and rewrites
the array in `services.sectional_test_catalog._TASK_ORDER` order.
Idempotent — running twice produces the same array. Safe to re-run.

The read path was also patched (see `get_locked_question_ids`) so any
row that drifts again in the future is reordered defensively at read
time. This migration just brings the stored data in line with what
callers will see.

Run
---
    python scripts/migrations/2026-05-20_reorder_sectional_seeds.py [--dry-run]
"""
import os
import sys

# Allow execution from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sqlalchemy.orm.attributes import flag_modified  # noqa: E402
from db.database import SessionLocal  # noqa: E402
from db.models import SectionalTestQuestions  # noqa: E402
from services.sectional_test_catalog import canonical_order  # noqa: E402


def main(dry_run: bool = False) -> None:
    db = SessionLocal()
    try:
        rows = (
            db.query(SectionalTestQuestions)
            .order_by(SectionalTestQuestions.module, SectionalTestQuestions.test_number)
            .all()
        )
        total = len(rows)
        rewritten = 0
        unchanged = 0
        for r in rows:
            before = list(r.question_ids or [])
            after = canonical_order(db, r.module, before)
            if before == after:
                unchanged += 1
                continue
            rewritten += 1
            print(
                f"  [{r.module}/test_{r.test_number}] reordering — "
                f"first 5 ids before: {before[:5]} → after: {after[:5]}"
            )
            if not dry_run:
                r.question_ids = after
                flag_modified(r, "question_ids")
        if not dry_run:
            db.commit()
        print()
        print(f"Total rows:      {total}")
        print(f"Rewritten:       {rewritten}")
        print(f"Already in order:{unchanged}")
        if dry_run:
            print("(dry-run — no commit)")
    finally:
        db.close()


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)

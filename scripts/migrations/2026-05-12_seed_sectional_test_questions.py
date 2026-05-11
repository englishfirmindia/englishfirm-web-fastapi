"""
One-off seed for sectional_test_questions.
==========================================

Picks the canonical question set for each (module, test_number) slot:
4 modules × 40 tests = 160 rows. After this runs, every user gets the
same questions for the same test, every redo included. Re-runs are
idempotent — existing rows are NOT overwritten (ON CONFLICT DO NOTHING).

Run once locally with the prod .env in place:

    PYTHONPATH=. python scripts/migrations/2026-05-12_seed_sectional_test_questions.py

Or with --module / --test-number flags to seed a single slot.

Sampling: uses each service's existing STRUCTURE constant — same module
+ task_type filter the live services used to use, minus the user-aware
`practiced_ids` exclusion (there is no user during seeding). If a task
type's pool has fewer questions than the structure asks for, takes the
whole pool (logged as a warning).
"""
import argparse
import random
import sys
from pathlib import Path
from typing import Optional

# Make sure we can import from the repo root regardless of cwd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sqlalchemy import text as _sql_text  # noqa: E402

from db.database import SessionLocal  # noqa: E402
from db.models import QuestionFromApeuni  # noqa: E402

from services.speaking_sectional_service import SPEAKING_STRUCTURE  # noqa: E402
from services.reading_sectional_service import READING_STRUCTURE  # noqa: E402
from services.writing_sectional_service import WRITING_STRUCTURE  # noqa: E402
from services.listening_sectional_service import LISTENING_STRUCTURE  # noqa: E402


TESTS_PER_MODULE = 40

# Each entry: (module_label, structure, default_db_module)
# default_db_module is the QuestionFromApeuni.module value used when a
# structure row has no explicit "module" override (speaking structure
# rows don't carry one — they're all speaking).
MODULES = [
    ("speaking",  SPEAKING_STRUCTURE,  "speaking"),
    ("reading",   READING_STRUCTURE,   "reading"),
    ("writing",   WRITING_STRUCTURE,   "writing"),
    ("listening", LISTENING_STRUCTURE, "listening"),
]


def _select_question_ids(db, structure, default_db_module) -> "list":
    """Walk the structure and pick `count` random question_ids per task."""
    selected: list[int] = []
    for task in structure:
        task_type = task["task"]
        db_module = task.get("module", default_db_module)
        count     = task["count"]
        pool = (
            db.query(QuestionFromApeuni.question_id)
            .filter(
                QuestionFromApeuni.module == db_module,
                QuestionFromApeuni.question_type == task_type,
            )
            .all()
        )
        ids = [row[0] for row in pool]
        if len(ids) < count:
            print(
                f"  ⚠️  pool for {db_module}/{task_type} has only {len(ids)} "
                f"(needed {count}); taking the whole pool"
            )
            picked = ids
        else:
            picked = random.sample(ids, count)
        selected.extend(picked)
    return selected


def seed(only_module: Optional[str] = None, only_test_number: Optional[int] = None) -> None:
    db = SessionLocal()
    try:
        total_inserted = 0
        total_skipped  = 0
        for module_label, structure, default_db_module in MODULES:
            if only_module and module_label != only_module:
                continue
            for test_number in range(1, TESTS_PER_MODULE + 1):
                if only_test_number is not None and test_number != only_test_number:
                    continue
                question_ids = _select_question_ids(db, structure, default_db_module)
                if not question_ids:
                    print(f"  ✗ {module_label}/test {test_number}: no questions, skipped")
                    continue
                result = db.execute(
                    _sql_text(
                        """
                        INSERT INTO sectional_test_questions
                            (module, test_number, question_ids)
                        VALUES
                            (:module, :test_number, :question_ids)
                        ON CONFLICT (module, test_number) DO NOTHING
                        """
                    ),
                    {
                        "module": module_label,
                        "test_number": test_number,
                        "question_ids": question_ids,
                    },
                )
                if result.rowcount:
                    total_inserted += 1
                    print(
                        f"  ✓ {module_label}/test {test_number}: "
                        f"{len(question_ids)} questions seeded"
                    )
                else:
                    total_skipped += 1
                    print(
                        f"  ↷ {module_label}/test {test_number}: already seeded, left as-is"
                    )
        db.commit()
        print(f"\nDone. Inserted: {total_inserted}  Skipped (already seeded): {total_skipped}")
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--module",
        choices=[m[0] for m in MODULES],
        help="Seed only this module (default: all four)",
    )
    parser.add_argument(
        "--test-number",
        type=int,
        help="Seed only this test_number (default: 1..40)",
    )
    args = parser.parse_args()
    seed(only_module=args.module, only_test_number=args.test_number)

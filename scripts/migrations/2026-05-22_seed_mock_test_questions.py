"""
2026-05-22 — Seed mock_test_questions (one-off + re-seedable)

Creates the `mock_test_questions` table if missing, then locks 40 sets of
65 questions — one per test_number (1..40). Each set follows MOCK_STRUCTURE:
6 RA + 10 RS + 5 DI + 2 RL + 5 ASQ + 2 SGD + 2 RTS + 2 SWT + 1 WE +
6 RFD + 3 RMCM + 2 RP + 4 RDD + 2 RMCS + 1 SST + 2 LMCM + 2 LFIB + 2 LHCS +
1 LMCS + 1 LSMW + 2 LHIW + 3 LWFD = 65.

Questions within each set are ordered in MOCK_STRUCTURE order so the in-test
sequence is consistent (speaking → writing → reading → listening, with
canonical PTE ordering within each section).

Usage:
  # First-time seeding (idempotent — skips test_numbers already present)
  python scripts/migrations/2026-05-22_seed_mock_test_questions.py

  # Refresh existing sets (REPLACES all 40)
  python scripts/migrations/2026-05-22_seed_mock_test_questions.py --reseed

  # Refresh just one set (e.g. test_number=5)
  python scripts/migrations/2026-05-22_seed_mock_test_questions.py --reseed-test 5

  # Dry run — print what would be seeded without committing
  python scripts/migrations/2026-05-22_seed_mock_test_questions.py --dry-run
"""

import argparse
import os
import random
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from db.database import SessionLocal
from db.models import MockTestQuestions, QuestionFromApeuni
from services.mock_service import (
    MOCK_STRUCTURE,
    _apply_mock_counts,
    _DB_TYPE_ALIASES,
    _NORMALIZE_TYPE,
)


TOTAL_TESTS = 40


def ensure_table(db) -> None:
    """Create the table if it doesn't exist. Idempotent."""
    db.execute(text("""
        CREATE TABLE IF NOT EXISTS mock_test_questions (
            test_number  integer       PRIMARY KEY,
            question_ids integer[]     NOT NULL,
            seeded_at    timestamptz   NOT NULL DEFAULT now()
        );
    """))
    db.commit()


def pick_one_set(db) -> list[int]:
    """Pick 65 question_ids following MOCK_STRUCTURE order. Mirrors the
    pre-fix logic from `start_mock_test` but with no per-user filtering
    (locked sets are global, not per-student)."""
    structure = _apply_mock_counts(db)
    selected: list[int] = []

    for t in structure:
        task_type = t["task_type"]
        module = t["module"]
        count = t["count"]
        if count == 0:
            continue

        db_type_candidates = _DB_TYPE_ALIASES.get(task_type, [task_type])
        pool = (
            db.query(QuestionFromApeuni)
            .filter(
                QuestionFromApeuni.module == module,
                QuestionFromApeuni.question_type.in_(db_type_candidates),
            )
            .all()
        )

        # HIW: only questions with passage + incorrectWords populated. Same
        # filter as `start_mock_test` so seeded sets aren't different in
        # this respect.
        if task_type == "highlight_incorrect_words":
            pool = [
                q for q in pool
                if (q.content_json or {}).get("passage")
                and (
                    (q.evaluation.evaluation_json if q.evaluation else {})
                    .get("correctAnswers", {})
                    .get("incorrectWords")
                )
            ]

        if len(pool) < count:
            print(f"  ⚠ only {len(pool)} questions available for {task_type} "
                  f"(needed {count}) — using all {len(pool)}")
            picks = pool
        else:
            picks = random.sample(pool, count)

        selected.extend(q.question_id for q in picks)

    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reseed", action="store_true",
                    help="Replace all 40 sets (default: skip ones already seeded)")
    ap.add_argument("--reseed-test", type=int, default=None,
                    help="Replace just this single test_number")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=None,
                    help="Optional RNG seed for reproducibility")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    db = SessionLocal()
    try:
        ensure_table(db)

        existing_nums = {
            r[0] for r in db.query(MockTestQuestions.test_number).all()
        }
        print(f"Existing locked sets: {sorted(existing_nums) or '(none)'}")

        if args.reseed_test is not None:
            targets = [args.reseed_test]
        else:
            targets = list(range(1, TOTAL_TESTS + 1))

        touched = 0
        for tn in targets:
            already = tn in existing_nums
            if already and not (args.reseed or args.reseed_test == tn):
                print(f"  test_number={tn}: already seeded, skipping (use --reseed to replace)")
                continue

            qids = pick_one_set(db)
            print(f"  test_number={tn}: {len(qids)} qids "
                  f"({'replace' if already else 'insert'})")

            if not args.dry_run:
                stmt = pg_insert(MockTestQuestions).values(
                    test_number=tn,
                    question_ids=qids,
                    seeded_at=datetime.now(timezone.utc),
                ).on_conflict_do_update(
                    index_elements=["test_number"],
                    set_={"question_ids": qids, "seeded_at": datetime.now(timezone.utc)},
                )
                db.execute(stmt)
                touched += 1

        if args.dry_run:
            print(f"\nDRY RUN — would touch {len(targets)} test(s). No changes committed.")
            db.rollback()
            return

        db.commit()
        print(f"\n✓ committed {touched} test set(s)")

        # Summary
        rows = db.query(MockTestQuestions).order_by(MockTestQuestions.test_number).all()
        print(f"\nTotal mock_test_questions rows: {len(rows)}")
        if rows:
            print(f"  test_numbers: {[r.test_number for r in rows]}")
            print(f"  qids/test:    {[len(r.question_ids) for r in rows]}")
    finally:
        db.close()


if __name__ == "__main__":
    main()

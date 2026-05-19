"""
Backfill: add 1 random Reorder Paragraphs question to each reading sectional
template that currently has fewer than 2.

Context: READING_STRUCTURE had count=1 for reorder_paragraphs; min table
(pte_min_question_count) says it should be 2. Code fixed in
services/reading_sectional_service.py, but the 40 templates already seeded
in sectional_test_questions have only 1 RP each.

This script appends 1 extra RP per template (random, not already in that
template). Idempotent — re-running on a template that already has >=2 RP
is a no-op for that template.

Run with prod .env in place:

    PYTHONPATH=. python scripts/migrations/2026-05-19_backfill_reading_rp.py
"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sqlalchemy import text as _sql_text  # noqa: E402

from db.database import SessionLocal  # noqa: E402


def backfill() -> None:
    db = SessionLocal()
    try:
        # Pull the full RP pool once
        rp_pool = [
            row[0]
            for row in db.execute(_sql_text(
                "SELECT question_id FROM questions_from_apeuni "
                "WHERE question_type = 'reorder_paragraphs'"
            )).all()
        ]
        print(f"RP pool size: {len(rp_pool)}")

        rows = db.execute(_sql_text(
            "SELECT test_number, question_ids FROM sectional_test_questions "
            "WHERE module = 'reading' ORDER BY test_number"
        )).all()

        updated = 0
        skipped = 0
        for r in rows:
            test_number = r.test_number
            qids = list(r.question_ids)
            rp_in_test = [q for q in qids if q in set(rp_pool)]
            if len(rp_in_test) >= 2:
                print(f"  ↷ test {test_number}: already has {len(rp_in_test)} RP, skipped")
                skipped += 1
                continue
            candidates = [q for q in rp_pool if q not in qids]
            if not candidates:
                print(f"  ✗ test {test_number}: no RP candidates left in pool, skipped")
                skipped += 1
                continue
            extra = random.choice(candidates)
            new_qids = qids + [extra]
            db.execute(
                _sql_text(
                    "UPDATE sectional_test_questions SET question_ids = :qids, "
                    "seeded_at = NOW() "
                    "WHERE module = 'reading' AND test_number = :tn"
                ),
                {"qids": new_qids, "tn": test_number},
            )
            print(f"  ✓ test {test_number}: appended RP qid {extra} (was {len(qids)} → {len(new_qids)})")
            updated += 1
        db.commit()
        print(f"\nDone. Updated: {updated}  Skipped: {skipped}")
    finally:
        db.close()


if __name__ == "__main__":
    backfill()

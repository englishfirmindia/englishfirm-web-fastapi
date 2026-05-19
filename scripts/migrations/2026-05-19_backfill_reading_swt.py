"""
Backfill: add 1 random Summarize Written Text question to each reading
sectional template that currently has fewer than 2.

Context: SWT count in READING_STRUCTURE was bumped from 1 to 2 in commit
7aa6c47, but the 40 templates in sectional_test_questions had already
been seeded with 1 SWT each. No re-seed ran, so /info advertises 23
questions / 2 SWT but the templates still serve 22 / 1 SWT.

Mirrors the RP backfill pattern (2026-05-19_backfill_reading_rp.py).
Idempotent — re-running on a template that already has >=2 SWT is a
no-op for that template.

Run with prod .env in place:

    PYTHONPATH=. python scripts/migrations/2026-05-19_backfill_reading_swt.py
"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

from sqlalchemy import text as _sql_text  # noqa: E402

from db.database import SessionLocal  # noqa: E402


def backfill() -> None:
    db = SessionLocal()
    try:
        swt_pool = [
            row[0]
            for row in db.execute(_sql_text(
                "SELECT question_id FROM questions_from_apeuni "
                "WHERE question_type = 'summarize_written_text'"
            )).all()
        ]
        print(f"SWT pool size: {len(swt_pool)}")
        swt_set = set(swt_pool)

        rows = db.execute(_sql_text(
            "SELECT test_number, question_ids FROM sectional_test_questions "
            "WHERE module = 'reading' ORDER BY test_number"
        )).all()

        updated = 0
        skipped = 0
        for r in rows:
            test_number = r.test_number
            qids = list(r.question_ids)
            swt_in_test = [q for q in qids if q in swt_set]
            if len(swt_in_test) >= 2:
                print(f"  ↷ test {test_number}: already has {len(swt_in_test)} SWT, skipped")
                skipped += 1
                continue
            candidates = [q for q in swt_pool if q not in qids]
            if not candidates:
                print(f"  ✗ test {test_number}: no SWT candidates left in pool, skipped")
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
            print(f"  ✓ test {test_number}: appended SWT qid {extra} (was {len(qids)} → {len(new_qids)})")
            updated += 1
        db.commit()
        print(f"\nDone. Updated: {updated}  Skipped: {skipped}")
    finally:
        db.close()


if __name__ == "__main__":
    backfill()

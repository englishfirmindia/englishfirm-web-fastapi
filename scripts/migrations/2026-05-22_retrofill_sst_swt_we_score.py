"""
2026-05-22 retrofill — SWT / WE / SST score column

Until today, mock_service._extract_score_and_max didn't recognise the
SWT/WE/SST scorer breakdown shape (uses `earned` + `max_pts` keys, not
`hits/total` like FIB or `score/max_possible` like MCQ). Result:
attempt_answers.score was persisted as 0 even when the LLM rubric gave
full marks. Review headers showed "0 / 12" everywhere and section-score
aggregation under-counted writing+SST.

This script:
  1. Finds attempt_answers rows for summarize_written_text / write_essay /
     summarize_spoken_text where score=0 and result_json.earned > 0.
  2. Updates row.score = result_json.earned.
  3. Re-aggregates each touched mock attempt's total_score via
     finish_mock_test (idempotency guard flipped first).

Usage:
  python scripts/migrations/2026-05-22_retrofill_sst_swt_we_score.py
      [--attempt-id 3719]   # default: just Nimisha's attempt
      [--all]               # all affected mock attempts
      [--dry-run]
"""

import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sqlalchemy.orm.attributes import flag_modified

from db.database import SessionLocal
from db.models import PracticeAttempt, AttemptAnswer
from services.mock_service import finish_mock_test


AFFECTED_TYPES = (
    "summarize_written_text",
    "write_essay",
    "summarize_spoken_text",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempt-id", type=int, default=3719)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    db = SessionLocal()
    try:
        q = db.query(AttemptAnswer).filter(
            AttemptAnswer.question_type.in_(AFFECTED_TYPES),
        )
        if not args.all:
            q = q.filter(AttemptAnswer.attempt_id == args.attempt_id)
        rows = q.all()

        affected_attempt_ids = set()
        for row in rows:
            rj = row.result_json or {}
            earned = rj.get("earned")
            if earned is None:
                continue
            try:
                earned_f = float(earned)
            except (TypeError, ValueError):
                continue
            if abs(float(row.score or 0) - earned_f) < 1e-6:
                continue  # already aligned
            old = row.score
            row.score = earned_f
            print(f"  attempt={row.attempt_id} qid={row.question_id} "
                  f"{row.question_type}  score {old} → {earned_f}")
            affected_attempt_ids.add(row.attempt_id)

        if args.dry_run:
            print(f"\nDRY RUN — no changes committed. Would touch "
                  f"{len(affected_attempt_ids)} attempt(s).")
            db.rollback()
            return

        db.commit()
        print(f"\n✓ committed score updates for {len(affected_attempt_ids)} attempt(s)")

        # Re-aggregate each touched mock attempt.
        for aid in affected_attempt_ids:
            attempt = db.query(PracticeAttempt).filter_by(id=aid).first()
            if not attempt or attempt.module != "mock":
                continue
            print(f"\n→ Re-aggregating mock attempt {aid} "
                  f"(was total_score={attempt.total_score})")
            attempt.status = "in_progress"
            attempt.scoring_status = "pending"
            db.commit()
            result = finish_mock_test(
                db=db, session_id=attempt.session_id, user_id=attempt.user_id,
            )
            print(f"  new total_score={result.get('overall_score')}  "
                  f"listening={result.get('listening_score')}  "
                  f"writing={result.get('writing_score')}")

    finally:
        db.close()


if __name__ == "__main__":
    main()

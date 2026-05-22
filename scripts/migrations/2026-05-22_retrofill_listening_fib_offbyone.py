"""
2026-05-22 retrofill — listening_fib off-by-one indexing

The mock frontend was submitting user_answers as 0-indexed dict keys
({"0": "...", "1": "..."}) while the FIB scorer keys on the 1-based blankId
from evaluation_json. Result: every blank's user value was compared against
the next blank's correct answer → perfectly correct answers scored 0.

This script:
  1. Finds attempt_answers rows where:
     - question_type = 'listening_fib'
     - user_answer_json.user_answers is a dict
     - keys are digit strings starting at "0"
  2. Re-runs FIBScorer with keys shifted +1 (so "0"→"1", etc.)
  3. Updates attempt_answers.score + result_json
  4. Re-aggregates the parent PracticeAttempt's total_score + task_breakdown
     via finish_mock_test (which is idempotent — we flip status back to
     pending first to force a fresh aggregation).

Usage:
  python scripts/migrations/2026-05-22_retrofill_listening_fib_offbyone.py
      [--attempt-id 3719]   # default: just Nimisha's attempt
      [--all]               # opt-in to fix every affected attempt
      [--dry-run]
"""

import os, sys, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sqlalchemy.orm.attributes import flag_modified

from db.database import SessionLocal
from db.models import (
    PracticeAttempt,
    AttemptAnswer,
    QuestionFromApeuni,
    QuestionEvaluationApeuni,
)
from services.scoring.registry import get_scorer
from services.mock_service import finish_mock_test


def needs_shift(ua: dict) -> bool:
    """Detects the 0-indexed dict shape produced by the buggy mock client."""
    if not isinstance(ua, dict) or not ua:
        return False
    keys = list(ua.keys())
    if not all(isinstance(k, str) and k.isdigit() for k in keys):
        return False
    return "0" in keys


def rescore_row(db, row: AttemptAnswer) -> tuple[int, int]:
    """Returns (old_score, new_score). Mutates row in-place."""
    ua = (row.user_answer_json or {}).get("user_answers")
    if not needs_shift(ua):
        return row.score, row.score

    shifted = {str(int(k) + 1): v for k, v in ua.items()}

    q = db.query(QuestionFromApeuni).filter_by(question_id=row.question_id).first()
    eval_row = db.query(QuestionEvaluationApeuni).filter_by(question_id=row.question_id).first()
    if not q or not eval_row or not eval_row.evaluation_json:
        print(f"  ⚠ skip qid={row.question_id}: no evaluation_json")
        return row.score, row.score

    scorer = get_scorer("listening_fib")
    result = scorer.score(
        question_id=row.question_id,
        session_id="retrofill",
        answer={"user_answers": shifted, "evaluation_json": eval_row.evaluation_json},
    )

    # Mock convention: score column stores raw hits (not PTE-scaled), and
    # result_json carries `maxScore` so the review screen renders "hits/total".
    bd = result.breakdown or {}
    new_score = int(bd.get("hits") or 0)
    max_raw = int(bd.get("total") or 0) or new_score

    old_score = row.score
    row.score = new_score
    new_ua = dict(row.user_answer_json or {})
    new_ua["user_answers"] = shifted
    new_ua["_retrofilled_2026_05_22"] = {"prior_keys": list(ua.keys())}
    row.user_answer_json = new_ua
    flag_modified(row, "user_answer_json")
    row.result_json = {
        **bd,
        "maxScore": max_raw,
        "pte_score": result.pte_score,
        "_retrofilled_2026_05_22": True,
    }
    flag_modified(row, "result_json")
    return old_score, new_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempt-id", type=int, default=3719,
                    help="Default 3719 (Nimisha's mock). Use --all for every affected attempt.")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    db = SessionLocal()
    try:
        q = db.query(AttemptAnswer).filter(AttemptAnswer.question_type == "listening_fib")
        if not args.all:
            q = q.filter(AttemptAnswer.attempt_id == args.attempt_id)
        rows = q.all()

        affected_attempt_ids = set()
        for row in rows:
            ua = (row.user_answer_json or {}).get("user_answers")
            if not needs_shift(ua):
                continue
            old, new = rescore_row(db, row)
            print(f"  attempt={row.attempt_id} qid={row.question_id}  "
                  f"score {old} → {new}  user_answers={ua}")
            affected_attempt_ids.add(row.attempt_id)

        if args.dry_run:
            print(f"\nDRY RUN — no changes committed. Would touch {len(affected_attempt_ids)} attempt(s).")
            db.rollback()
            return

        db.commit()
        print(f"\n✓ committed updates for {len(affected_attempt_ids)} attempt(s)")

        # Re-aggregate each touched mock attempt.
        for aid in affected_attempt_ids:
            attempt = db.query(PracticeAttempt).get(aid)
            if not attempt or attempt.module != "mock":
                continue
            # finish_mock_test is idempotent on (status, scoring_status) — flip
            # them back so it recomputes section + overall scores.
            print(f"\n→ Re-aggregating mock attempt {aid} (was total_score={attempt.total_score})")
            attempt.status = "in_progress"
            attempt.scoring_status = "pending"
            db.commit()
            result = finish_mock_test(db=db, session_id=attempt.session_id, user_id=attempt.user_id)
            print(f"  new total_score={result.get('overall_score')}  "
                  f"listening={result.get('listening_score')}")

    finally:
        db.close()


if __name__ == "__main__":
    main()

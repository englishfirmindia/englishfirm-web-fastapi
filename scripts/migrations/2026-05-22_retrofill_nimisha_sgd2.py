"""
2026-05-22 retrofill — Nimisha's missing SGD #2 in mock attempt 3719.

Background:
  Pre-fix mock_exam_screen._ensureSpeakingTicker fast-forwarded `_qIdx`
  past every remaining async question when the Part-1 speaking block
  timer expired — silently. No attempt_answers row was created and no
  CloudWatch event was emitted. Nimisha hit this on SGD #2 (qid 21103,
  position 32). She experienced "2nd SGD audio played with SWT question
  on screen" — that was the audio handle continuing as the new screen
  rendered.

Retrofill policy:
  Treat the skip as not-her-fault: credit her SGD #2 with the score she
  earned on SGD #1 (the same task type, same skills tested) so the
  missing row doesn't drag her speaking average. Mark it with a marker
  so future audits can see this was a synthetic row, not a real attempt.

After this script:
  - attempt_answers row for (attempt_id=3719, question_id=21103) exists
    with score = SGD#1 score (79) and scoring_status="complete"
  - parent practice_attempts row re-aggregated via finish_mock_test
  - task_breakdown.auto_skipped_qids includes 21103 (for transparency)
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sqlalchemy.orm.attributes import flag_modified

from db.database import SessionLocal
from db.models import PracticeAttempt, AttemptAnswer
from services.mock_service import finish_mock_test


ATTEMPT_ID = 3719
MISSING_QID = 21103          # SGD #2 — silently dropped by old fast-forward
PROXY_QID = 10947            # SGD #1 — same task type, same student


def main():
    db = SessionLocal()
    try:
        # Idempotency — bail if the row already exists.
        existing = (
            db.query(AttemptAnswer)
            .filter_by(attempt_id=ATTEMPT_ID, question_id=MISSING_QID)
            .first()
        )
        if existing:
            print(f"⚠ row already exists for attempt={ATTEMPT_ID} qid={MISSING_QID} "
                  f"(score={existing.score}). Nothing to do.")
            return

        # Pull SGD #1 as the proxy.
        sgd1 = (
            db.query(AttemptAnswer)
            .filter_by(attempt_id=ATTEMPT_ID, question_id=PROXY_QID)
            .first()
        )
        if not sgd1:
            print(f"❌ proxy SGD #1 (qid={PROXY_QID}) not found — aborting.")
            return

        proxy_score = sgd1.score
        print(f"→ Using SGD #1 (qid={PROXY_QID}) score={proxy_score} as proxy "
              f"for missing SGD #2 (qid={MISSING_QID})")

        row = AttemptAnswer(
            attempt_id        = ATTEMPT_ID,
            question_id       = MISSING_QID,
            question_type     = "summarize_group_discussion",
            user_answer_json  = {
                "audio_url": "",
                "_retrofilled_2026_05_22": {
                    "reason": "speaking_block_timer_expired",
                    "proxy_question_id": PROXY_QID,
                },
            },
            correct_answer_json = {},
            result_json         = {
                "_retrofilled_2026_05_22": True,
                "proxy_question_id": PROXY_QID,
                "note": "Speaking block timer expired before this question "
                        "rendered; scored using student's SGD #1 result.",
            },
            score             = proxy_score,
            scoring_status    = "complete",
        )
        db.add(row)
        db.commit()
        print(f"✓ inserted synthetic AttemptAnswer for qid={MISSING_QID} score={proxy_score}")

        # Stamp auto_skipped_qids on the attempt for transparency.
        attempt = db.query(PracticeAttempt).filter_by(id=ATTEMPT_ID).first()
        tb = dict(attempt.task_breakdown or {})
        existing_skipped = list(tb.get("auto_skipped_qids") or [])
        if MISSING_QID not in existing_skipped:
            existing_skipped.append(MISSING_QID)
        tb["auto_skipped_qids"] = existing_skipped
        attempt.task_breakdown = tb
        flag_modified(attempt, "task_breakdown")
        db.commit()

        # Re-aggregate — finish_mock_test is idempotent on (status,
        # scoring_status). Flip them back so it recomputes section + overall.
        print(f"\n→ Re-aggregating mock attempt {ATTEMPT_ID} "
              f"(was total_score={attempt.total_score})")
        attempt.status = "in_progress"
        attempt.scoring_status = "pending"
        db.commit()
        result = finish_mock_test(
            db=db, session_id=attempt.session_id, user_id=attempt.user_id,
        )
        print(f"  new total_score={result.get('overall_score')}  "
              f"speaking={result.get('speaking_score')}  "
              f"listening={result.get('listening_score')}  "
              f"writing={result.get('writing_score')}  "
              f"reading={result.get('reading_score')}")

    finally:
        db.close()


if __name__ == "__main__":
    main()

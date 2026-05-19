"""
One-off retrofill: re-score the 6 writing-sectional answers in Nimisha's
attempt 3672 and rewrite `result_json` with the rich breakdown the
frontend expects.

Why: writing sectional submit previously persisted only
{pte_score, maxScore, error} — the LLM breakdown (form / content / dsc /
grammar / glr / vocabulary / spelling / max_pts / task_type / mistakes
/ highlights) was thrown away. Both the student review screen
(_isRichSwtSectional) and the trainer review screen
(_isRichWritingBreakdown) gate on `result.content.score` + `max_pts`,
so without those fields they fall through to a bare "Score: X".

The current backend already persists the rich shape going forward —
this script just back-fills Nimisha's last attempt.

Preserves the original `pte_score` on each row (Claude isn't
deterministic; we don't want the headline number to wobble). Only adds
the breakdown fields.

Run with prod .env in place:

    PYTHONPATH=. python scripts/migrations/2026-05-19_retrofill_nimisha_writing_sectional.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

from sqlalchemy.orm.attributes import flag_modified  # noqa: E402

from db.database import SessionLocal  # noqa: E402
from db.models import AttemptAnswer, QuestionFromApeuni  # noqa: E402
from services.scoring import get_scorer  # noqa: E402


ATTEMPT_ID = 3672  # Nimisha's most recent writing sectional

_SCORER_ALIAS = {
    "summarize_spoken_text": "listening_sst",
}


def _build_answer(question: QuestionFromApeuni, user_text: str) -> dict:
    qt = question.question_type
    content_json = question.content_json or {}
    if qt == "summarize_written_text":
        prompt = content_json.get("passage", content_json.get("text", ""))
        return {"text": user_text, "prompt": prompt}
    if qt == "write_essay":
        prompt = (
            content_json.get("topic")
            or content_json.get("prompt")
            or content_json.get("text", "")
        )
        return {"text": user_text, "prompt": prompt}
    if qt == "summarize_spoken_text":
        from services.transcription_service import get_or_create_sst_transcript
        return {"text": user_text, "prompt": get_or_create_sst_transcript(question)}
    if qt == "listening_wfd":
        return {
            "user_text": user_text,
            "evaluation_json": question.evaluation.evaluation_json if question.evaluation else {},
        }
    return {}


def retrofill() -> None:
    db = SessionLocal()
    try:
        answers = (
            db.query(AttemptAnswer)
            .filter(AttemptAnswer.attempt_id == ATTEMPT_ID)
            .order_by(AttemptAnswer.id)
            .all()
        )
        print(f"Retrofilling {len(answers)} answers for attempt {ATTEMPT_ID}")
        for a in answers:
            question = (
                db.query(QuestionFromApeuni)
                .filter(QuestionFromApeuni.question_id == a.question_id)
                .one_or_none()
            )
            if question is None:
                print(f"  ✗ qid={a.question_id}: question not found, skipped")
                continue
            user_text = (a.user_answer_json or {}).get("text", "")
            if not user_text and a.question_type == "listening_wfd":
                # WFD stores under "text" too; fall through normally
                pass
            scorer_key = _SCORER_ALIAS.get(a.question_type, a.question_type)
            try:
                scorer = get_scorer(scorer_key)
                result = scorer.score(
                    question_id=a.question_id,
                    session_id=f"retrofill-{ATTEMPT_ID}-{a.id}",
                    answer=_build_answer(question, user_text),
                )
            except Exception as e:
                print(f"  ✗ qid={a.question_id} qt={a.question_type}: scorer raised {e!r}")
                continue

            breakdown = result.breakdown or {}
            original_pte = a.score  # preserve headline number
            original_max = (a.result_json or {}).get("maxScore")
            new_result = {
                **breakdown,
                "pte_score": original_pte,
                "maxScore": original_max if original_max is not None else breakdown.get("max_pts"),
                "error": result.error,
                "retrofilled_at": "2026-05-19",
            }
            a.result_json = new_result
            flag_modified(a, "result_json")
            keys = ",".join(sorted(k for k in new_result.keys() if k != "retrofilled_at"))
            print(
                f"  ✓ qid={a.question_id} qt={a.question_type:25s} "
                f"pte={original_pte} rich-keys=[{keys}]"
            )
        db.commit()
        print("\nDone.")
    finally:
        db.close()


if __name__ == "__main__":
    retrofill()

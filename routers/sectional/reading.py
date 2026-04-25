"""
Reading Sectional Router
========================
Endpoints:
  GET  /sectional/reading/info                → module structure
  POST /sectional/reading/exam                → start exam, returns session_id + questions
  POST /sectional/reading/submit              → submit one answer, scores and stores in session
  POST /sectional/reading/finish              → compute final weighted score, persist attempt
  GET  /sectional/reading/results/{sid}       → return scoring status + final score

Submit payload shapes per task_type:

  reading_fib / reading_fib_drop_down:
    { session_id, question_id, user_answers: {blank_id: value, ...} }

  reading_mcs:
    { session_id, question_id, selected_option: str }

  reading_mcm:
    { session_id, question_id, selected_options: [str, ...] }

  reorder_paragraphs:
    { session_id, question_id, user_sequence: [str, ...] }
"""

from fastapi import APIRouter, Depends, Body, HTTPException
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import User, AttemptAnswer
from core.dependencies import get_current_user
from services.session_service import ACTIVE_SESSIONS
from services.scoring import get_scorer
from services.reading_sectional_service import (
    get_reading_sectional_info,
    start_reading_sectional_exam,
    resume_reading_sectional_exam,
    finish_reading_sectional,
    get_reading_sectional_results,
    _question_max,
)

router = APIRouter(prefix="/sectional/reading", tags=["Sectional - Reading"])


_SCORER_ALIAS = {
    "mcq_single":   "reading_mcs",
    "mcq_multiple": "reading_mcm",
}


def _build_answer(question_type: str, payload: dict) -> dict:
    """Build the answer dict expected by each scorer, given the raw request payload."""
    if question_type == "reading_fib_drop_down":
        return {"user_answers": payload.get("user_answers", {})}
    if question_type == "mcq_single":
        return {"selected_option": payload.get("selected_option", "")}
    if question_type == "mcq_multiple":
        return {"selected_options": payload.get("selected_options", [])}
    if question_type == "reorder_paragraphs":
        return {"user_sequence": payload.get("user_sequence", [])}
    return {}


@router.get("/info")
def info():
    return get_reading_sectional_info()


@router.post("/exam")
def start_exam(
    payload: dict = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    test_number = int(payload.get("test_number", 1))
    return start_reading_sectional_exam(db=db, user_id=current_user.id, test_number=test_number)


@router.get("/resume/{session_id}")
def resume_exam(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return resume_reading_sectional_exam(session_id=session_id, user_id=current_user.id, db=db)


@router.post("/submit")
def submit_answer(
    payload: dict = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Score one reading answer and store result in session for weighted finish scoring.
    """
    session_id  = payload["session_id"]
    question_id = int(payload["question_id"])

    session = ACTIVE_SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=400, detail="Session not found or expired")
    if session.get("user_id") != current_user.id:
        raise HTTPException(status_code=403, detail="Forbidden")

    question = session["questions"].get(question_id)
    if not question:
        raise HTTPException(status_code=404, detail="Question not found in session")
    if not question.evaluation:
        raise HTTPException(status_code=422, detail="Question has no evaluation data")

    answer = _build_answer(question.question_type, payload)
    answer["evaluation_json"] = question.evaluation.evaluation_json

    scorer = get_scorer(_SCORER_ALIAS.get(question.question_type, question.question_type))
    result = scorer.score(
        question_id=question_id,
        session_id=session_id,
        answer=answer,
    )

    # Store raw score in rubric units for weighted finish
    q_max      = _question_max(question)
    earned_raw = result.raw_score * q_max

    session.setdefault("question_scores", {})[question_id]      = earned_raw
    session.setdefault("question_score_maxes", {})[question_id] = q_max
    session.setdefault("submitted_questions", set()).add(question_id)
    session["score"] = session.get("score", 0) + result.pte_score

    attempt_id = session.get("attempt_id")
    if attempt_id:
        existing = db.query(AttemptAnswer).filter_by(
            attempt_id=attempt_id, question_id=question_id
        ).first()
        if not existing:
            user_answer = {k: v for k, v in answer.items() if k != "evaluation_json"}
            result_json = {**(result.breakdown or {}), "pte_score": result.pte_score, "maxScore": q_max}
            db.add(AttemptAnswer(
                attempt_id          = attempt_id,
                question_id         = question_id,
                question_type       = question.question_type,
                user_answer_json    = user_answer,
                correct_answer_json = {},
                result_json         = result_json,
                score               = result.pte_score,
                scoring_status      = "complete",
            ))
            db.commit()

    return {
        "pte_score":  result.pte_score,
        "raw_score":  round(result.raw_score, 3),
        "earned_raw": round(earned_raw, 2),
        "max_raw":    q_max,
        "is_async":   result.is_async,
        "breakdown":  result.breakdown,
    }


@router.post("/finish")
def finish_exam(
    payload: dict = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session_id = payload["session_id"]
    return finish_reading_sectional(session_id=session_id, user_id=current_user.id, db=db)


@router.get("/results/{session_id}")
def get_results(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return get_reading_sectional_results(session_id=session_id, user_id=current_user.id, db=db)


@router.get("/latest")
def latest(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Most recent reading sectional attempt for this user.
    Ported from app-fastapi/routers/questions.py."""
    from db.models import PracticeAttempt
    attempt = (
        db.query(PracticeAttempt)
        .filter_by(user_id=current_user.id, module="reading", question_type="sectional")
        .order_by(PracticeAttempt.id.desc())
        .first()
    )
    if not attempt:
        return {"found": False}
    return {
        "found": True,
        "session_id": attempt.session_id,
        "attempt_id": attempt.id,
        "scoring_status": attempt.scoring_status or "pending",
        "reading_score": attempt.total_score,
    }

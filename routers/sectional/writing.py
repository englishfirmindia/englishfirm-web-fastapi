"""
Writing Sectional Router
========================
Endpoints:
  GET  /sectional/writing/info                → module structure
  POST /sectional/writing/exam                → start exam, returns session_id + questions
  POST /sectional/writing/submit              → submit one answer, scores and stores in session
  POST /sectional/writing/finish              → compute final weighted score, persist attempt
  GET  /sectional/writing/results/{sid}       → return scoring status + final score
"""

from fastapi import APIRouter, Depends, Body, HTTPException
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import User
from core.dependencies import get_current_user
from services.session_service import ACTIVE_SESSIONS
from services.scoring import get_scorer
from services.writing_sectional_service import (
    get_writing_sectional_info,
    start_writing_sectional_exam,
    finish_writing_sectional,
    get_writing_sectional_results,
    _question_max,
)

router = APIRouter(prefix="/sectional/writing", tags=["Sectional - Writing"])


@router.get("/info")
def info():
    return get_writing_sectional_info()


@router.post("/exam")
def start_exam(
    payload: dict = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    test_number = int(payload.get("test_number", 1))
    return start_writing_sectional_exam(db=db, user_id=current_user.id, test_number=test_number)


@router.post("/submit")
def submit_answer(
    payload: dict = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Score one writing answer and store result in session for weighted finish scoring.

    Payload:
      session_id:  str
      question_id: int
      user_answer: str   (the written response)
    """
    session_id  = payload["session_id"]
    question_id = int(payload["question_id"])
    user_answer = payload["user_answer"]

    session = ACTIVE_SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=400, detail="Session not found or expired")
    if session.get("user_id") != current_user.id:
        raise HTTPException(status_code=403, detail="Forbidden")

    question = session["questions"].get(question_id)
    if not question:
        raise HTTPException(status_code=404, detail="Question not found in session")

    content_json = question.content_json or {}
    prompt = content_json.get("passage", content_json.get("text", ""))

    try:
        scorer = get_scorer(question.question_type)
        result = scorer.score(
            question_id=question_id,
            session_id=session_id,
            answer={"text": user_answer, "prompt": prompt},
        )
        raw_score = result.raw_score
        pte_score = result.pte_score
        error     = result.error
    except NotImplementedError:
        # AI scorer not yet configured — assign zero score, continue exam
        raw_score = 0.0
        pte_score = 10
        error     = "ai_scorer_not_configured"
    except Exception as e:
        raw_score = 0.0
        pte_score = 10
        error     = str(e)

    # Compute raw score in rubric units (0..max) for weighted finish scoring
    q_max = _question_max(question)
    earned_raw = raw_score * q_max

    session.setdefault("question_scores", {})[question_id]      = earned_raw
    session.setdefault("question_score_maxes", {})[question_id] = q_max
    session.setdefault("submitted_questions", set()).add(question_id)
    session["score"] = session.get("score", 0) + pte_score

    return {
        "pte_score":  pte_score,
        "raw_score":  round(raw_score, 3),
        "earned_raw": round(earned_raw, 2),
        "max_raw":    q_max,
        "is_async":   False,
        "error":      error,
    }


@router.post("/finish")
def finish_exam(
    payload: dict = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session_id = payload["session_id"]
    return finish_writing_sectional(session_id=session_id, user_id=current_user.id, db=db)


@router.get("/results/{session_id}")
def get_results(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return get_writing_sectional_results(session_id=session_id, user_id=current_user.id, db=db)

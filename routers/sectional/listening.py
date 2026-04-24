"""
Listening Sectional Router
==========================
Endpoints:
  GET  /sectional/listening/info              → module structure
  POST /sectional/listening/exam              → start exam, returns session_id + questions
  POST /sectional/listening/submit            → submit one answer, scores and stores in session
  POST /sectional/listening/finish            → compute final weighted score, persist attempt
  GET  /sectional/listening/results/{sid}     → return scoring status + final score

Submit payload shapes per task_type:

  listening_wfd:
    { session_id, question_id, user_text: str }

  listening_fib:
    { session_id, question_id, user_answers: {blank_id: value, ...} }

  listening_mcs / listening_hcs / listening_smw:
    { session_id, question_id, selected_option: str }

  listening_mcm:
    { session_id, question_id, selected_options: [str, ...] }

  listening_hiw:
    { session_id, question_id, highlighted_words: [str, ...] }

  listening_sst:
    { session_id, question_id, user_answer: str }
"""

from fastapi import APIRouter, Depends, Body, HTTPException
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import User, AttemptAnswer
from core.dependencies import get_current_user
from services.session_service import ACTIVE_SESSIONS
from services.scoring import get_scorer
from services.listening_sectional_service import (
    get_listening_sectional_info,
    start_listening_sectional_exam,
    finish_listening_sectional,
    get_listening_sectional_results,
    _question_max,
)

router = APIRouter(prefix="/sectional/listening", tags=["Sectional - Listening"])


_SCORER_ALIAS = {
    "summarize_spoken_text":   "listening_sst",
    "listening_mcq_single":    "listening_mcs",
    "listening_mcq_multiple":  "listening_mcm",
    "highlight_incorrect_words": "listening_hiw",
}


def _build_answer(question_type: str, payload: dict) -> dict:
    """Build the answer dict expected by each scorer."""
    if question_type == "listening_wfd":
        return {"user_text": payload.get("user_text", "")}
    if question_type == "listening_fib":
        return {"user_answers": payload.get("user_answers", {})}
    if question_type in ("listening_mcq_single", "listening_mcs", "listening_hcs", "listening_smw"):
        return {"selected_option": payload.get("selected_option", "")}
    if question_type in ("listening_mcq_multiple", "listening_mcm"):
        return {"selected_options": payload.get("selected_options", [])}
    if question_type in ("highlight_incorrect_words", "listening_hiw"):
        return {"highlighted_words": payload.get("highlighted_words", [])}
    if question_type in ("summarize_spoken_text", "listening_sst"):
        return {"text": payload.get("user_answer", ""), "prompt": ""}
    return {}


@router.get("/info")
def info():
    return get_listening_sectional_info()


@router.post("/exam")
def start_exam(
    payload: dict = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    test_number = int(payload.get("test_number", 1))
    return start_listening_sectional_exam(db=db, user_id=current_user.id, test_number=test_number)


@router.post("/submit")
def submit_answer(
    payload: dict = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Score one listening answer and store result in session for weighted finish scoring.
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

    answer = _build_answer(question.question_type, payload)

    # SST uses AI scorer — needs no evaluation_json; others need it
    if question.question_type not in ("listening_sst", "summarize_spoken_text"):
        if not question.evaluation:
            raise HTTPException(status_code=422, detail="Question has no evaluation data")
        answer["evaluation_json"] = question.evaluation.evaluation_json

    try:
        scorer = get_scorer(_SCORER_ALIAS.get(question.question_type, question.question_type))
        result = scorer.score(
            question_id=question_id,
            session_id=session_id,
            answer=answer,
        )
        raw_score = result.raw_score
        pte_score = result.pte_score
        error     = result.error
        breakdown = result.breakdown
    except NotImplementedError:
        # AI scorer (SST) not yet configured — assign zero score, continue exam
        raw_score = 0.0
        pte_score = 10
        error     = "ai_scorer_not_configured"
        breakdown = {}
    except Exception as e:
        raw_score = 0.0
        pte_score = 10
        error     = str(e)
        breakdown = {}

    q_max      = _question_max(question)
    earned_raw = raw_score * q_max

    session.setdefault("question_scores", {})[question_id]      = earned_raw
    session.setdefault("question_score_maxes", {})[question_id] = q_max
    session.setdefault("submitted_questions", set()).add(question_id)
    session["score"] = session.get("score", 0) + pte_score

    attempt_id = session.get("attempt_id")
    if attempt_id:
        existing = db.query(AttemptAnswer).filter_by(
            attempt_id=attempt_id, question_id=question_id
        ).first()
        if not existing:
            result_json = {**(breakdown or {}), "pte_score": pte_score, "maxScore": q_max}
            db.add(AttemptAnswer(
                attempt_id          = attempt_id,
                question_id         = question_id,
                question_type       = question.question_type,
                user_answer_json    = {},
                correct_answer_json = {},
                result_json         = result_json,
                score               = pte_score,
                scoring_status      = "complete",
            ))
            db.commit()

    return {
        "pte_score":  pte_score,
        "raw_score":  round(raw_score, 3),
        "earned_raw": round(earned_raw, 2),
        "max_raw":    q_max,
        "is_async":   False,
        "breakdown":  breakdown,
        "error":      error,
    }


@router.post("/finish")
def finish_exam(
    payload: dict = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session_id = payload["session_id"]
    return finish_listening_sectional(session_id=session_id, user_id=current_user.id, db=db)


@router.get("/results/{session_id}")
def get_results(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return get_listening_sectional_results(session_id=session_id, user_id=current_user.id, db=db)


@router.get("/latest")
def latest(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Most recent listening sectional attempt for this user.
    Ported from app-fastapi/routers/questions.py."""
    from db.models import PracticeAttempt
    attempt = (
        db.query(PracticeAttempt)
        .filter_by(user_id=current_user.id, module="listening", question_type="sectional")
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
        "listening_score": attempt.total_score,
    }

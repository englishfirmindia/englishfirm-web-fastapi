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
from db.models import User, AttemptAnswer, PracticeAttempt
from core.dependencies import get_current_user
from services.session_service import ACTIVE_SESSIONS
from services.scoring import get_scorer
from services.writing_sectional_service import (
    get_writing_sectional_info,
    start_writing_sectional_exam,
    resume_writing_sectional_exam,
    finish_writing_sectional,
    get_writing_sectional_results,
    _question_max,
)

router = APIRouter(prefix="/sectional/writing", tags=["Sectional - Writing"])


# Map question_type → scorer key in the scoring registry.
# SWT/WE use AI scorers keyed by their own question_type. SST uses the
# AI scorer registered as 'listening_sst'. WFD uses the rule scorer
# registered as 'listening_wfd'.
_SCORER_ALIAS = {
    "summarize_spoken_text": "listening_sst",
}


def _build_answer(question, payload: dict) -> dict:
    """Build the answer dict expected by each scorer."""
    qt = question.question_type
    if qt in ("summarize_written_text", "write_essay"):
        content_json = question.content_json or {}
        prompt = content_json.get("passage", content_json.get("text", ""))
        return {"text": payload.get("user_answer", ""), "prompt": prompt}
    if qt == "summarize_spoken_text":
        # No content prompt available (audio source) — heuristic-only path
        return {"text": payload.get("user_answer", ""), "prompt": ""}
    if qt == "listening_wfd":
        return {
            "user_text": payload.get("user_text", ""),
            "evaluation_json": question.evaluation.evaluation_json if question.evaluation else {},
        }
    return {}


def _user_answer_json(question_type: str, payload: dict) -> dict:
    """Shape of the row stored in attempt_answers.user_answer_json."""
    if question_type == "listening_wfd":
        return {"text": payload.get("user_text", "")}
    return {"text": payload.get("user_answer", "")}


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
    already_done = (
        db.query(PracticeAttempt)
        .filter(
            PracticeAttempt.user_id == current_user.id,
            PracticeAttempt.module == "writing",
            PracticeAttempt.question_type == "sectional",
            PracticeAttempt.status == "complete",
        )
        .all()
    )
    for a in already_done:
        if (a.task_breakdown or {}).get("test_number") == test_number:
            raise HTTPException(
                status_code=409,
                detail=f"Test {test_number} has already been completed and cannot be retaken.",
            )
    return start_writing_sectional_exam(db=db, user_id=current_user.id, test_number=test_number)


@router.get("/resume/{session_id}")
def resume_exam(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return resume_writing_sectional_exam(session_id=session_id, user_id=current_user.id, db=db)


@router.post("/submit")
def submit_answer(
    payload: dict = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Score one writing answer and store result in session for weighted finish scoring.

    Payload (varies by question type):
      session_id:  str
      question_id: int
      user_answer: str   (SWT, WE, SST — written response)
      user_text:   str   (WFD — dictation transcription)
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

    answer = _build_answer(question, payload)

    try:
        scorer_key = _SCORER_ALIAS.get(question.question_type, question.question_type)
        scorer = get_scorer(scorer_key)
        result = scorer.score(
            question_id=question_id,
            session_id=session_id,
            answer=answer,
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

    attempt_id = session.get("attempt_id")
    if attempt_id:
        existing = db.query(AttemptAnswer).filter_by(
            attempt_id=attempt_id, question_id=question_id
        ).first()
        if not existing:
            result_json = {"pte_score": pte_score, "maxScore": q_max, "error": error}
            db.add(AttemptAnswer(
                attempt_id          = attempt_id,
                question_id         = question_id,
                question_type       = question.question_type,
                user_answer_json    = _user_answer_json(question.question_type, payload),
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


@router.get("/latest")
def latest(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Most recent writing sectional attempt for this user.
    Ported from app-fastapi/routers/questions.py."""
    from db.models import PracticeAttempt
    attempt = (
        db.query(PracticeAttempt)
        .filter_by(user_id=current_user.id, module="writing", question_type="sectional")
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
        "writing_score": attempt.total_score,
    }


@router.get("/attempted-tests")
def attempted_tests(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Returns completed writing sectional tests with most-recent completion date."""
    attempts = (
        db.query(PracticeAttempt)
        .filter(
            PracticeAttempt.user_id == current_user.id,
            PracticeAttempt.module == "writing",
            PracticeAttempt.question_type == "sectional",
            PracticeAttempt.status == "complete",
        )
        .order_by(PracticeAttempt.completed_at.desc())
        .all()
    )
    seen: dict[int, str] = {}
    for a in attempts:
        tb = a.task_breakdown or {}
        tn = tb.get("test_number")
        if isinstance(tn, int) and tn not in seen:
            seen[tn] = a.completed_at.isoformat() if a.completed_at else None
    return {
        "attempted_tests": [
            {"test_number": tn, "completed_at": dt}
            for tn, dt in sorted(seen.items())
        ]
    }

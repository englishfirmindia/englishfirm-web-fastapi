"""
Speaking Sectional Router
=========================
Endpoints:
  GET  /sectional/speaking/info              → module structure
  POST /sectional/speaking/exam              → start exam, returns session_id + questions
  POST /sectional/speaking/submit-audio      → record submitted audio_url into session
  POST /sectional/speaking/finish            → kick off Azure scoring, return pending status
  GET  /sectional/speaking/results/{sid}     → poll scoring status + final score
"""

from fastapi import APIRouter, Depends, Body
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import User
from core.dependencies import get_current_user
from services.session_service import ACTIVE_SESSIONS
from services.speaking_sectional_service import (
    get_speaking_sectional_info,
    start_speaking_sectional_exam,
    finish_speaking_sectional,
    get_speaking_sectional_results,
)

router = APIRouter(prefix="/sectional/speaking", tags=["Sectional - Speaking"])


@router.get("/info")
def info():
    return get_speaking_sectional_info()


@router.post("/exam")
def start_exam(
    payload: dict = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    test_number = int(payload.get("test_number", 1))
    return start_speaking_sectional_exam(db=db, user_id=current_user.id, test_number=test_number)


@router.post("/submit-audio")
def submit_audio(
    payload: dict = Body(...),
    current_user: User = Depends(get_current_user),
):
    """
    Records that a question's audio has been uploaded to S3.
    Called by the Flutter client after each successful S3 PUT.
    Does NOT kick off scoring — that happens at /finish.
    """
    session_id  = payload["session_id"]
    question_id = int(payload["question_id"])
    audio_url   = payload["audio_url"]

    session = ACTIVE_SESSIONS.get(session_id)
    if not session:
        return {"status": "session_not_found"}
    if session.get("user_id") != current_user.id:
        return {"status": "forbidden"}

    session.setdefault("submitted_audio", {})[question_id] = audio_url
    session.setdefault("submitted_questions", set()).add(question_id)

    return {"status": "recorded", "question_id": question_id}


@router.post("/finish")
def finish_exam(
    payload: dict = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session_id = payload["session_id"]
    return finish_speaking_sectional(session_id=session_id, user_id=current_user.id, db=db)


@router.get("/results/{session_id}")
def get_results(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return get_speaking_sectional_results(session_id=session_id, user_id=current_user.id, db=db)

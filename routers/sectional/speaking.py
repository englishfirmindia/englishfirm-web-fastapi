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

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import User, PracticeAttempt
from core.dependencies import get_current_user
from services.session_service import ACTIVE_SESSIONS, persist_speaking_answer_pending
from services.speaking_scorer import kick_off_scoring
from core.security_helpers import safe_question_id, assert_audio_url_owned
from services.speaking_sectional_service import (

    get_speaking_sectional_info,
    start_speaking_sectional_exam,
    resume_speaking_sectional_exam,
    finish_speaking_sectional,
    get_speaking_sectional_results,
)

from core.logging_config import get_logger

log = get_logger(__name__)


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
    already_done = (
        db.query(PracticeAttempt)
        .filter(
            PracticeAttempt.user_id == current_user.id,
            PracticeAttempt.module == "speaking",
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
    return start_speaking_sectional_exam(db=db, user_id=current_user.id, test_number=test_number)


@router.get("/resume/{session_id}")
def resume_exam(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return resume_speaking_sectional_exam(session_id=session_id, user_id=current_user.id, db=db)


@router.post("/submit-audio")
def submit_audio(
    payload: dict = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Records that a question's audio has been uploaded to S3.
    Called by the Flutter client after each successful S3 PUT.
    Does NOT kick off scoring — that happens at /finish.
    """
    session_id  = payload["session_id"]
    question_id = safe_question_id(payload, db)
    audio_url   = payload["audio_url"]
    assert_audio_url_owned(audio_url, current_user.id)

    session = ACTIVE_SESSIONS.get(session_id)
    if not session:
        return {"status": "session_not_found"}
    if session.get("user_id") != current_user.id:
        return {"status": "forbidden"}

    session.setdefault("submitted_audio", {})[question_id] = audio_url
    session.setdefault("submitted_questions", set()).add(question_id)

    # Write pending AttemptAnswer to RDS + kick off Azure scoring immediately
    q = session.get("questions", {}).get(question_id)
    if q:
        persist_speaking_answer_pending(session, question_id, q.question_type, audio_url)
        cj = q.content_json or {}
        if q.question_type == "repeat_sentence":
            reference_text = cj.get("transcript", "")
        else:
            reference_text = cj.get("passage", "")
        kick_off_scoring(current_user.id, question_id, q.question_type, audio_url, reference_text)

    return {"status": "recorded", "question_id": question_id}


@router.post("/finish")
def finish_exam(
    background_tasks: BackgroundTasks,
    payload: dict = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session_id = payload["session_id"]

    # Drain any locally-cached audio submissions before kicking off final scoring.
    pending = payload.get("pending_submits") or []
    pending_failed = []
    for item in pending:
        if not isinstance(item, dict):
            continue
        qid = item.get("question_id")
        inner = item.get("payload") or {}
        if qid is None:
            continue
        try:
            submit_audio(
                payload={
                    **inner,
                    "session_id": session_id,
                    "question_id": qid,
                },
                db=db,
                current_user=current_user,
            )
        except Exception as e:
            log.error(f"[Speaking Sectional] pending_submit failed q={qid}: {e}")
            pending_failed.append(
                {"question_id": qid, "error": str(e)[:200]}
            )

    result = finish_speaking_sectional(
        session_id=session_id,
        user_id=current_user.id,
        db=db,
        background_tasks=background_tasks,
    )
    if pending_failed:
        result["pending_questions"] = pending_failed
        if result.get("scoring_status") == "complete":
            result["scoring_status"] = "partial"
    return result


@router.get("/results/{session_id}")
def get_results(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return get_speaking_sectional_results(session_id=session_id, user_id=current_user.id, db=db)


@router.get("/attempted-tests")
def attempted_tests(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Returns completed speaking sectional tests with most-recent completion date."""
    attempts = (
        db.query(PracticeAttempt)
        .filter(
            PracticeAttempt.user_id == current_user.id,
            PracticeAttempt.module == "speaking",
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


@router.get("/latest")
def latest(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Most recent speaking sectional attempt for this user.
    Ported from app-fastapi/routers/questions.py::get_latest_speaking_sectional."""
    from db.models import PracticeAttempt
    attempt = (
        db.query(PracticeAttempt)
        .filter_by(user_id=current_user.id, module="speaking", question_type="sectional")
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
        "speaking_score": attempt.total_score,
    }

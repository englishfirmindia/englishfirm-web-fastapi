"""
Mock Test router — Full PTE Academic exam (all 4 sections, ~2.5 hours).

Ported from englishfirm-app-fastapi/routers/questions.py:1632-1721
(same request/response contracts as mobile).

Endpoints (mounted under /api/v1/questions):
  GET  /mock/info
  POST /mock/start
  GET  /mock/part/{session_id}/{part}
  POST /mock/progress
  POST /mock/finish
  GET  /mock/results/{session_id}
  GET  /mock/review/{session_id}
  GET  /mock/resume/{session_id}
"""

from fastapi import APIRouter, Depends, Body, HTTPException
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import User
from core.dependencies import get_current_user
from services.mock_service import (
    get_mock_info,
    start_mock_test,
    get_mock_part,
    update_mock_progress,
    finish_mock_test,
    get_mock_results,
    get_mock_review,
    resume_mock_test,
    submit_mock_answer,
)

router = APIRouter(tags=["Mock Test"])


@router.get("/mock/info")
def mock_info(db: Session = Depends(get_db)):
    """Returns Part/Content/Time table for the mock intro screen."""
    return get_mock_info(db)


@router.post("/mock/start")
def mock_start(
    payload: dict = Body(default={}),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Picks 65 questions and creates a PracticeAttempt(module='mock')."""
    test_number = int(payload.get("test_number", 1))
    return start_mock_test(db=db, user_id=current_user.id, test_number=test_number)


@router.get("/mock/part/{session_id}/{part}")
def mock_part(
    session_id: str,
    part: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Returns questions + timing for the requested part (1/2/3)."""
    return get_mock_part(db=db, session_id=session_id, part=part)


@router.post("/mock/progress")
def mock_progress(
    payload: dict = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Saves current_part + remaining-time snapshot for resume."""
    session_id = payload.get("session_id", "")
    current_part = int(payload.get("current_part", 1))
    timer_remaining = payload.get("timer_remaining")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")
    return update_mock_progress(
        db=db,
        session_id=session_id,
        current_part=current_part,
        timer_remaining=int(timer_remaining) if timer_remaining is not None else None,
    )


@router.post("/mock/finish")
def mock_finish(
    payload: dict = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Computes 4 section scores + overall."""
    session_id = payload.get("session_id", "")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")

    # Drain locally-cached submits the client couldn't send mid-test.
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
            submit_mock_answer(
                session_id=session_id,
                question_id=int(qid),
                payload={**inner, "session_id": session_id, "question_id": qid},
            )
        except Exception as e:
            print(
                f"[Mock] pending_submit failed q={qid}: {e}",
                flush=True,
            )
            pending_failed.append(
                {"question_id": qid, "error": str(e)[:200]}
            )

    result = finish_mock_test(db=db, session_id=session_id, user_id=current_user.id)
    if pending_failed and isinstance(result, dict):
        result["pending_questions"] = pending_failed
        if result.get("scoring_status") == "complete":
            result["scoring_status"] = "partial"
    return result


@router.get("/mock/results/{session_id}")
def mock_results(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Polls mock results. Re-computes if async speaking was pending."""
    return get_mock_results(session_id=session_id, user_id=current_user.id, db=db)


@router.get("/mock/review/{session_id}")
def mock_review(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Returns all answers grouped by section for the review screen."""
    return get_mock_review(session_id=session_id, user_id=current_user.id, db=db)


@router.post("/mock/submit")
def mock_submit(
    payload: dict = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Unified submit for all non-speaking mock question types."""
    session_id = payload.get("session_id", "")
    question_id = payload.get("question_id")
    if not session_id or question_id is None:
        raise HTTPException(status_code=400, detail="session_id and question_id required")
    return submit_mock_answer(session_id=session_id, question_id=int(question_id), payload=payload)


@router.get("/mock/resume/{session_id}")
def mock_resume(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Restores a pending mock session with saved part + remaining timer."""
    return resume_mock_test(session_id=session_id, user_id=current_user.id, db=db)

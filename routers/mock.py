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
from db.models import PracticeAttempt, User
from core.dependencies import get_current_user
from services.billing.enforce_limit import EnforceLimit
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

from core.logging_config import get_logger

log = get_logger(__name__)


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
    _gate=Depends(EnforceLimit("mocks")),
):
    """Picks 65 questions and creates a PracticeAttempt(module='mock').

    test_number identifies which of the 40 numbered mock slots the user
    chose; 0 is reserved for debug (re-uses already-submitted questions).
    """
    test_number = int(payload.get("test_number", 1))
    if test_number != 0 and not (1 <= test_number <= 40):
        raise HTTPException(
            status_code=400,
            detail="test_number must be between 1 and 40",
        )
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
    _score_gate=Depends(EnforceLimit("mock_score")),
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
            log.error(f"[Mock] pending_submit failed q={qid}: {e}")
            pending_failed.append(
                {"question_id": qid, "error": str(e)[:200]}
            )

    result = finish_mock_test(db=db, session_id=session_id, user_id=current_user.id)
    if pending_failed and isinstance(result, dict):
        result["pending_questions"] = pending_failed
        if result.get("scoring_status") == "complete":
            result["scoring_status"] = "partial"

    # auto_skipped_qids: questions silently dropped when the Part-1 speaking
    # block timer expired mid-attempt. Pre-2026-05-22 these vanished from
    # attempt_answers entirely; now the client builds pending_submits for
    # them AND ships the qid list here so we can stamp it on the attempt
    # row for the feedback screen ("we ran out of time and skipped N").
    auto_skipped = payload.get("auto_skipped_qids") or []
    if auto_skipped:
        try:
            from db.models import PracticeAttempt
            from sqlalchemy.orm.attributes import flag_modified
            attempt = (
                db.query(PracticeAttempt)
                .filter_by(session_id=session_id, user_id=current_user.id, module="mock")
                .first()
            )
            if attempt is not None:
                tb = dict(attempt.task_breakdown or {})
                existing = list(tb.get("auto_skipped_qids") or [])
                # De-duplicate but preserve order.
                seen = set(existing)
                for qid in auto_skipped:
                    if qid not in seen:
                        existing.append(qid)
                        seen.add(qid)
                tb["auto_skipped_qids"] = existing
                attempt.task_breakdown = tb
                flag_modified(attempt, "task_breakdown")
                db.commit()
                log.info(
                    f"[MOCK_FF] session={session_id} skipped_count={len(auto_skipped)} qids={auto_skipped}"
                )
        except Exception as e:
            log.error(f"[Mock] failed to record auto_skipped_qids: {e}")
        if isinstance(result, dict):
            result["auto_skipped_qids"] = auto_skipped

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


@router.get("/mock/pending")
def mock_pending(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List pending mock attempts the user can resume.

    Joins `practice_attempts` (status='pending') with
    `practice_session_states` (saved state blob, must not be expired) so
    the client can surface a "Resume" row without relying on a
    browser-local cache that may have been cleared.
    """
    from sqlalchemy import text as _sql_text
    rows = db.execute(_sql_text(
        """
        SELECT pa.id            AS attempt_id,
               pa.session_id    AS session_id,
               pa.started_at    AS started_at,
               COUNT(aa.id)     AS submitted_count,
               pss.updated_at   AS state_updated_at,
               pss.expires_at   AS state_expires_at
        FROM practice_attempts pa
        JOIN practice_session_states pss ON pss.session_id = pa.session_id
        LEFT JOIN attempt_answers aa ON aa.attempt_id = pa.id
        WHERE pa.user_id = :uid
          AND pa.module = 'mock'
          AND pa.status = 'pending'
          AND pss.expires_at > NOW()
        GROUP BY pa.id, pa.session_id, pa.started_at, pss.updated_at, pss.expires_at
        ORDER BY pss.updated_at DESC
        """
    ), {"uid": current_user.id}).all()
    return {
        "pending": [
            {
                "attempt_id":      r.attempt_id,
                "session_id":      r.session_id,
                "started_at":      r.started_at.isoformat() if r.started_at else None,
                "submitted_count": int(r.submitted_count or 0),
                "last_saved_at":   r.state_updated_at.isoformat() if r.state_updated_at else None,
                "expires_at":      r.state_expires_at.isoformat() if r.state_expires_at else None,
            }
            for r in rows
        ]
    }


@router.get("/mock/attempted-tests")
def mock_attempted_tests(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Returns completed mocks for the current user, deduped by test_number,
    keeping the most-recent completion date and the best overall_score seen.
    """
    attempts = (
        db.query(PracticeAttempt)
        .filter(
            PracticeAttempt.user_id == current_user.id,
            PracticeAttempt.module == "mock",
            PracticeAttempt.status == "complete",
        )
        .order_by(PracticeAttempt.completed_at.desc())
        .all()
    )
    seen: dict[int, dict] = {}
    for a in attempts:
        tb = a.task_breakdown or {}
        tn = tb.get("test_number")
        if not isinstance(tn, int) or not (1 <= tn <= 40):
            continue
        completed_at = a.completed_at.isoformat() if a.completed_at else None
        overall = tb.get("overall_score", a.total_score)
        existing = seen.get(tn)
        if existing is None:
            seen[tn] = {
                "test_number": tn,
                "completed_at": completed_at,
                "best_score": overall,
            }
        else:
            if overall is not None and (
                existing["best_score"] is None or overall > existing["best_score"]
            ):
                existing["best_score"] = overall
    return {
        "attempted_tests": [seen[tn] for tn in sorted(seen.keys())]
    }

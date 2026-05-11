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
from sqlalchemy import text as _sql_text
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import User, PracticeAttempt
from core.dependencies import get_current_user
from services.session_service import ACTIVE_SESSIONS, persist_speaking_answer_pending
from services.speaking_scorer import kick_off_scoring
from core.security_helpers import safe_question_id, assert_audio_url_owned, resolve_question_with_retry
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
    # Redos of completed tests are allowed — each is a new versioned row.
    # Only block if there's an existing INCOMPLETE attempt for this same
    # test slot; the user should Resume that one instead of starting a
    # parallel attempt.
    in_progress = (
        db.query(PracticeAttempt)
        .filter(
            PracticeAttempt.user_id == current_user.id,
            PracticeAttempt.module == "speaking",
            PracticeAttempt.question_type == "sectional",
            PracticeAttempt.status != "complete",
        )
        .all()
    )
    for a in in_progress:
        if (a.task_breakdown or {}).get("test_number") == test_number:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "in_progress",
                    "session_id": a.session_id,
                    "message": f"Test {test_number} is still in progress — resume it before starting a new attempt.",
                },
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
    ACTIVE_SESSIONS.save(session_id)

    # Write pending AttemptAnswer to RDS + kick off Azure scoring immediately.
    # Retry the question lookup before giving up — a cold task right after
    # deploy can miss the in-process cache and pick up a transient DB blip,
    # which is exactly how Nimisha's q=22542 silently scored 0.
    q = resolve_question_with_retry(question_id, db, session=session)
    if q is None:
        raise HTTPException(
            status_code=503,
            detail="question lookup failed",
            headers={"Retry-After": "5"},
        )
    persist_speaking_answer_pending(session, question_id, q.question_type, audio_url)
    cj = q.content_json or {}
    if q.question_type == "repeat_sentence":
        reference_text = (cj.get("transcript") or "").strip()
    else:
        reference_text = (cj.get("passage") or "").strip()
    extra_warnings: list = []
    # Only RA / RS use reference_text downstream; other types pass empty by
    # design (their scorer ignores it). Flag missing only for RA / RS.
    if q.question_type in ("read_aloud", "repeat_sentence") and not reference_text:
        log.error(
            "[Sectional submit-audio] q=%d type=%s reference missing — scoring with warning",
            question_id, q.question_type,
        )
        extra_warnings.append("reference_missing")
    kick_off_scoring(
        current_user.id, question_id, q.question_type, audio_url, reference_text,
        extra_warnings=extra_warnings,
    )

    return {"status": "recorded", "question_id": question_id}


@router.post("/timer")
def update_timer(
    payload: dict = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Persist the current on-screen sectional countdown into
    practice_attempts.time_remaining_seconds. Called by the frontend after
    every speaking-sectional submit (via session_id) so a resume picks up
    at the exact second the user last saw.

    LEAST(COALESCE(existing, :s), :s) guarantees out-of-order arrivals
    (older payload landing after a newer one due to network reorder) can
    never raise the value — the smaller wins. NULL existing → take the
    new value as-is.
    """
    session_id = payload.get("session_id")
    trs = payload.get("time_remaining_seconds")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")
    if not isinstance(trs, (int, float)) or trs < 0:
        raise HTTPException(status_code=400, detail="time_remaining_seconds must be a non-negative number")

    session = ACTIVE_SESSIONS.get(session_id)
    if not session or session.get("user_id") != current_user.id:
        # Don't 404 — the timer is best-effort. Caller shouldn't have to
        # branch on session-not-found errors.
        return {"status": "ignored"}
    attempt_id = session.get("attempt_id")
    if attempt_id is None:
        return {"status": "ignored"}

    db.execute(
        _sql_text("""
            UPDATE practice_attempts
            SET time_remaining_seconds = LEAST(
                COALESCE(time_remaining_seconds, :s), :s
            )
            WHERE id = :attempt_id
        """),
        {"s": int(trs), "attempt_id": attempt_id},
    )
    db.commit()
    return {"status": "saved", "time_remaining_seconds": int(trs)}


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
    """Returns every completed speaking sectional attempt with derived version.

    Response:
      attempted_tests: [{test_number, completed_at}]   ← back-compat: latest per test
      attempts:        [{test_number, version, score, session_id, completed_at}]
                       ← every completed attempt, oldest = v1 within each test_number
    """
    rows = (
        db.query(PracticeAttempt)
        .filter(
            PracticeAttempt.user_id == current_user.id,
            PracticeAttempt.module == "speaking",
            PracticeAttempt.question_type == "sectional",
            PracticeAttempt.status == "complete",
        )
        .order_by(PracticeAttempt.completed_at.asc())
        .all()
    )
    # Bucket by test_number (oldest first) to derive version 1..N
    by_test: dict[int, list[PracticeAttempt]] = {}
    for a in rows:
        tn = (a.task_breakdown or {}).get("test_number")
        if isinstance(tn, int):
            by_test.setdefault(tn, []).append(a)
    attempts_payload = []
    latest_per_test: dict[int, str] = {}
    for tn in sorted(by_test):
        for version, a in enumerate(by_test[tn], start=1):
            completed_at_iso = a.completed_at.isoformat() if a.completed_at else None
            attempts_payload.append({
                "test_number": tn,
                "version":     version,
                "score":       a.total_score,
                "session_id":  a.session_id,
                "completed_at": completed_at_iso,
            })
            latest_per_test[tn] = completed_at_iso
    return {
        "attempted_tests": [
            {"test_number": tn, "completed_at": latest_per_test[tn]}
            for tn in sorted(latest_per_test)
        ],
        "attempts": attempts_payload,
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

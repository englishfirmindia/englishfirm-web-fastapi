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

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import User, AttemptAnswer, PracticeAttempt
from core.dependencies import get_current_user
from services.session_service import ACTIVE_SESSIONS
from services.scoring import get_scorer
from services.listening_sectional_service import (

    get_listening_sectional_info,
    start_listening_sectional_exam,
    resume_listening_sectional_exam,
    finish_listening_sectional,
    get_listening_sectional_results,
    _question_max,
)

from core.logging_config import get_logger

log = get_logger(__name__)


router = APIRouter(prefix="/sectional/listening", tags=["Sectional - Listening"])


def _extract_user_answer(question_type: str, payload: dict) -> dict:
    if question_type == "listening_wfd":
        return {"text": payload.get("user_text", "")}
    if question_type in ("listening_sst", "summarize_spoken_text"):
        return {"text": payload.get("user_answer", "")}
    if question_type == "listening_fib":
        return {"user_answers": payload.get("user_answers", {})}
    if question_type in ("listening_mcs", "listening_hcs", "listening_smw", "listening_mcq_single"):
        return {"selected_option": payload.get("selected_option", "")}
    if question_type in ("listening_mcm", "listening_mcq_multiple"):
        return {"selected_options": payload.get("selected_options", [])}
    if question_type in ("listening_hiw", "highlight_incorrect_words"):
        return {"highlighted_words": payload.get("highlighted_words", [])}
    return {}


_SCORER_ALIAS = {
    "summarize_spoken_text":   "listening_sst",
    "listening_mcq_single":    "listening_mcs",
    "listening_mcq_multiple":  "listening_mcm",
    "highlight_incorrect_words": "listening_hiw",
}


def _build_answer(question, payload: dict) -> dict:
    """Build the answer dict expected by each scorer."""
    question_type = question.question_type
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
        from services.transcription_service import get_or_create_sst_transcript
        return {
            "text": payload.get("user_answer", ""),
            "prompt": get_or_create_sst_transcript(question),
        }
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
    # Redos of completed tests are allowed — each is a new versioned row.
    # Only block when an INCOMPLETE attempt exists for this test slot.
    in_progress = (
        db.query(PracticeAttempt)
        .filter(
            PracticeAttempt.user_id == current_user.id,
            PracticeAttempt.module == "listening",
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
    return start_listening_sectional_exam(db=db, user_id=current_user.id, test_number=test_number)


@router.get("/resume/{session_id}")
def resume_exam(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return resume_listening_sectional_exam(session_id=session_id, user_id=current_user.id, db=db)


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

    answer = _build_answer(question, payload)

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
    ACTIVE_SESSIONS.save(session_id)

    attempt_id = session.get("attempt_id")
    if attempt_id:
        existing = db.query(AttemptAnswer).filter_by(
            attempt_id=attempt_id, question_id=question_id
        ).first()
        if not existing:
            result_json = {**(breakdown or {}), "pte_score": pte_score, "maxScore": q_max}
            user_answer_json = _extract_user_answer(question.question_type, payload)
            db.add(AttemptAnswer(
                attempt_id          = attempt_id,
                question_id         = question_id,
                question_type       = question.question_type,
                user_answer_json    = user_answer_json,
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
    background_tasks: BackgroundTasks,
    payload: dict = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session_id = payload["session_id"]

    # Drain any locally-cached submits before final scoring kicks off.
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
            submit_payload = {
                **inner,
                "session_id": session_id,
                "question_id": qid,
            }
            submit_answer(
                payload=submit_payload, db=db, current_user=current_user
            )
        except Exception as e:
            log.error(f"[Listening Sectional] pending_submit failed q={qid}: {e}")
            pending_failed.append(
                {"question_id": qid, "error": str(e)[:200]}
            )

    result = finish_listening_sectional(
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
    return get_listening_sectional_results(session_id=session_id, user_id=current_user.id, db=db)


@router.get("/attempted-tests")
def attempted_tests(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Returns every completed listening sectional attempt with derived version.

    Response:
      attempted_tests: [{test_number, completed_at}]   ← back-compat: latest per test
      attempts:        [{test_number, version, score, session_id, completed_at}]
    """
    rows = (
        db.query(PracticeAttempt)
        .filter(
            PracticeAttempt.user_id == current_user.id,
            PracticeAttempt.module == "listening",
            PracticeAttempt.question_type == "sectional",
            PracticeAttempt.status == "complete",
        )
        .order_by(PracticeAttempt.completed_at.asc())
        .all()
    )
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

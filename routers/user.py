from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func

from db.database import get_db
from db.models import (
    User,
    UserQuestionAttempt,
    AttemptAnswer,
    PracticeAttempt,
    QuestionFromApeuni,
    QuestionEvaluationApeuni,
)
from core.dependencies import get_current_user

router = APIRouter(prefix="/user", tags=["User"])


@router.get("/me")
def get_me(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "score_requirement": current_user.score_requirement,
        "exam_date": str(current_user.exam_date) if current_user.exam_date else None,
    }


@router.get("/dashboard")
def get_dashboard(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    total_answered = (
        db.query(func.count(UserQuestionAttempt.id))
        .filter(UserQuestionAttempt.user_id == current_user.id)
        .scalar() or 0
    )
    practice_days = (
        db.query(func.count(func.distinct(func.date(UserQuestionAttempt.attempted_at))))
        .filter(UserQuestionAttempt.user_id == current_user.id)
        .scalar() or 0
    )
    return {
        "username": current_user.username,
        "total_questions_answered": total_answered,
        "practice_days": practice_days,
    }


# Question-type aliases — the FE and BE historically diverged on the
# Reading FIB drag-drop name. Flutter ships `reading_fib` (route /practice/
# reading-fib, HttpGenericPracticeApiService('reading_fib')), but the
# fill_in_blanks.py submit handler persists AA rows with
# `question_type='reading_drag_and_drop'`. Map both names to the same
# bucket so /answered-questions returns the right set regardless of which
# name the caller sends.
_QUESTION_TYPE_ALIASES: dict[str, tuple[str, ...]] = {
    "reading_fib": ("reading_fib", "reading_drag_and_drop"),
    "reading_drag_and_drop": ("reading_fib", "reading_drag_and_drop"),
}


@router.get("/answered-questions")
def get_answered_questions(
    question_type: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Question IDs the user has an actual recorded answer for.

    Switched from user_question_attempts (UQA) → attempt_answers because
    UQA records "ever touched" but never gets cleaned up, so old AA rows
    that cascade-deleted with their parent PracticeAttempt leave UQA
    entries with nothing behind them. That divergence was making the
    practice list flag questions as "practiced ✓" — and the per-screen
    restore path treat them as already-answered — even when
    /user/last-answer/{qid} returned null on re-entry and the screen
    fell back to a blank fresh question. Reading practice felt this
    most (Nimisha had 69 UQA rows for reading_mcm but only 5 surviving
    AA rows); speaking / writing / listening get the same correctness
    benefit for any legacy ghosts.

    `question_type` accepts both the Flutter screen's type string and
    the backend's stored AA type — see _QUESTION_TYPE_ALIASES above.
    """
    query = (
        db.query(AttemptAnswer.question_id)
        .join(PracticeAttempt, AttemptAnswer.attempt_id == PracticeAttempt.id)
        .filter(PracticeAttempt.user_id == current_user.id)
    )
    if question_type:
        types = _QUESTION_TYPE_ALIASES.get(question_type, (question_type,))
        query = query.filter(AttemptAnswer.question_type.in_(types))
    ids = [row[0] for row in query.distinct().all()]
    return {"answered_question_ids": ids}


@router.get("/last-answer/{question_id}")
def get_last_answer(
    question_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    answer = (
        db.query(AttemptAnswer)
        .join(PracticeAttempt, AttemptAnswer.attempt_id == PracticeAttempt.id)
        .filter(
            PracticeAttempt.user_id == current_user.id,
            AttemptAnswer.question_id == question_id,
        )
        .order_by(AttemptAnswer.submitted_at.desc())
        .first()
    )
    if not answer:
        return None
    result_json = dict(answer.result_json or {})
    if "pte_score" not in result_json and answer.score is not None:
        result_json["pte_score"] = answer.score
    if "correct_answers" not in result_json:
        from db.models import QuestionEvaluationApeuni
        ev = db.query(QuestionEvaluationApeuni).filter_by(question_id=question_id).first()
        if ev and ev.evaluation_json:
            eval_json = ev.evaluation_json or {}
            raw = eval_json.get("correctAnswers", {}) or {}
            if isinstance(raw, dict) and raw.get("blanks") is not None:
                correct_answers = {
                    str(b.get("blankId")): b.get("answer")
                    for b in raw.get("blanks", [])
                }
            else:
                correct_answers = {str(k): v for k, v in raw.items()}
            if correct_answers:
                result_json["correct_answers"] = correct_answers
    return {
        "user_answer_json": answer.user_answer_json,
        "result_json": result_json,
    }


@router.get("/attempts/history")
def get_attempts_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    from sqlalchemy import func as _func
    from db.models import TrainerNote, TrainerShare

    attempts = (
        db.query(PracticeAttempt)
        .filter(
            PracticeAttempt.user_id == current_user.id,
            PracticeAttempt.status == "complete",
        )
        .order_by(PracticeAttempt.started_at.desc())
        .limit(50)
        .all()
    )
    attempt_ids = [a.id for a in attempts]

    # Per-attempt counts in two cheap aggregates (no n+1).
    notes_by_attempt: dict = {}
    shares_by_attempt: dict = {}
    if attempt_ids:
        for aid, cnt in (
            db.query(TrainerNote.attempt_id, _func.count(TrainerNote.id))
            .filter(
                TrainerNote.attempt_id.in_(attempt_ids),
                TrainerNote.deleted_at.is_(None),
            )
            .group_by(TrainerNote.attempt_id)
            .all()
        ):
            notes_by_attempt[aid] = cnt
        for aid, cnt in (
            db.query(TrainerShare.attempt_id, _func.count(TrainerShare.id))
            .filter(
                TrainerShare.attempt_id.in_(attempt_ids),
                TrainerShare.revoked_at.is_(None),
            )
            .group_by(TrainerShare.attempt_id)
            .all()
        ):
            shares_by_attempt[aid] = cnt

    return [
        {
            "id": a.id,
            "session_id": a.session_id,
            "module": a.module,
            "question_type": a.question_type,
            "filter_type": a.filter_type,
            "total_questions": a.total_questions,
            "questions_answered": a.questions_answered,
            "total_score": a.total_score,
            "status": a.status,
            "scoring_status": a.scoring_status,
            "task_breakdown": a.task_breakdown,
            "started_at": a.started_at.isoformat() if a.started_at else None,
            "completed_at": a.completed_at.isoformat() if a.completed_at else None,
            "notes_count": notes_by_attempt.get(a.id, 0),
            "active_shares_count": shares_by_attempt.get(a.id, 0),
        }
        for a in attempts
    ]


@router.get("/attempts/evaluation/{question_id}")
def get_evaluation(
    question_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    ev = db.query(QuestionEvaluationApeuni).filter_by(question_id=question_id).first()
    if not ev:
        return None
    return {"evaluation_json": ev.evaluation_json}


@router.get("/practice-attempts/by-session/{session_id}/answers")
def get_practice_session_answers(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Per-question results for a practice attempt, looked up by session_id.

    Powers the end-of-practice results screen — returns each answered
    question with its content, the user's submission, the stored result
    (score + breakdown + correct answer + time_on_question_seconds), and
    the correctAnswers payload from the evaluation row so the UI can
    render correct-answer overlays without a second round-trip per row.
    """
    attempt = (
        db.query(PracticeAttempt)
        .filter_by(session_id=session_id, user_id=current_user.id)
        .first()
    )
    if not attempt:
        raise HTTPException(status_code=404, detail="Practice attempt not found")

    answers = (
        db.query(AttemptAnswer)
        .filter_by(attempt_id=attempt.id)
        .order_by(AttemptAnswer.submitted_at.asc().nullslast())
        .all()
    )
    qids = [a.question_id for a in answers]
    questions_by_id: dict = {}
    evaluations_by_id: dict = {}
    if qids:
        for q in (
            db.query(QuestionFromApeuni)
            .filter(QuestionFromApeuni.question_id.in_(qids))
            .all()
        ):
            questions_by_id[q.question_id] = q
        for ev in (
            db.query(QuestionEvaluationApeuni)
            .filter(QuestionEvaluationApeuni.question_id.in_(qids))
            .all()
        ):
            evaluations_by_id[ev.question_id] = ev

    rows = []
    for idx, a in enumerate(answers):
        q = questions_by_id.get(a.question_id)
        ev = evaluations_by_id.get(a.question_id)
        rows.append(
            {
                "index": idx,
                "question_id": a.question_id,
                "question_type": a.question_type,
                "score": a.score,
                "scoring_status": a.scoring_status,
                "user_answer_json": a.user_answer_json or {},
                "result_json": a.result_json or {},
                "submitted_at": (
                    a.submitted_at.isoformat() if a.submitted_at else None
                ),
                "content_json": (q.content_json if q else {}) or {},
                "title": (q.title if q else None),
                "correct_answers": (
                    ((ev.evaluation_json or {}).get("correctAnswers") or {})
                    if ev else {}
                ),
            }
        )

    # Aggregate top-line stats. PTE average is over scored answers only;
    # time-total uses the stopwatch field when present.
    scored = [r["score"] for r in rows if isinstance(r["score"], int)]
    avg_pte = round(sum(scored) / len(scored)) if scored else 0
    total_time = sum(
        int(r["result_json"].get("time_on_question_seconds") or 0) for r in rows
    )

    return {
        "attempt_id": attempt.id,
        "session_id": session_id,
        "module": attempt.module,
        "question_type": attempt.question_type,
        "total_questions": attempt.total_questions,
        "questions_answered": len(rows),
        "started_at": (
            attempt.started_at.isoformat() if attempt.started_at else None
        ),
        "average_pte": avg_pte,
        "total_time_seconds": total_time,
        "answers": rows,
    }

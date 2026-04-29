from typing import Optional
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from db.database import get_db
from db.models import User, UserQuestionAttempt, AttemptAnswer, PracticeAttempt
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


@router.get("/answered-questions")
def get_answered_questions(
    question_type: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    query = db.query(UserQuestionAttempt.question_id).filter(
        UserQuestionAttempt.user_id == current_user.id
    )
    if question_type:
        query = query.filter(UserQuestionAttempt.question_type == question_type)
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
    from db.models import QuestionEvaluationApeuni
    ev = db.query(QuestionEvaluationApeuni).filter_by(question_id=question_id).first()
    if not ev:
        return None
    return {"evaluation_json": ev.evaluation_json}

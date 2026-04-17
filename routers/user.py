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
    return {
        "user_answer_json": answer.user_answer_json,
        "result_json": answer.result_json,
    }

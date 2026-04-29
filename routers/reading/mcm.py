import math
from typing import Optional
from fastapi import APIRouter, Depends, Body, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import asc, desc

from db.database import get_db
from db.models import User, QuestionFromApeuni, UserQuestionAttempt
from core.dependencies import get_current_user
from services.session_service import start_session, get_session, mark_submitted, persist_answer_to_db
from services.scoring import get_scorer

router = APIRouter(prefix="/reading/mcm", tags=["Reading - MCM"])


@router.get("/list")
def list_questions(
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
    difficulty: Optional[int] = Query(default=None),
    is_prediction: Optional[bool] = Query(default=None),
    practiced: Optional[bool] = Query(default=None),
    search: Optional[str] = Query(default=None),
    sort: str = Query(default='asc'),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    query = db.query(QuestionFromApeuni).filter(
        QuestionFromApeuni.module == "reading",
        QuestionFromApeuni.question_type == "mcq_multiple",
    )
    if difficulty is not None:
        query = query.filter(QuestionFromApeuni.difficulty_level == difficulty)
    if is_prediction is not None:
        query = query.filter(QuestionFromApeuni.is_prediction == is_prediction)
    if practiced is not None:
        practiced_subq = (
            db.query(UserQuestionAttempt.question_id)
            .filter(UserQuestionAttempt.user_id == current_user.id)
            .subquery()
        )
        if practiced:
            query = query.filter(QuestionFromApeuni.question_id.in_(practiced_subq))
        else:
            query = query.filter(~QuestionFromApeuni.question_id.in_(practiced_subq))
    if search:
        query = query.filter(QuestionFromApeuni.title.ilike(f'%{search}%'))

    total = query.count()
    total_pages = math.ceil(total / limit) if total > 0 else 1
    order_dir = desc if sort == 'desc' else asc
    questions = (
        query
        .order_by(order_dir(QuestionFromApeuni.question_id))
        .offset((page - 1) * limit)
        .limit(limit)
        .all()
    )

    page_qids = [q.question_id for q in questions]
    practiced_ids: set = set()
    if page_qids:
        rows = (
            db.query(UserQuestionAttempt.question_id)
            .filter(
                UserQuestionAttempt.user_id == current_user.id,
                UserQuestionAttempt.question_id.in_(page_qids),
            )
            .all()
        )
        practiced_ids = {r[0] for r in rows}

    return {
        "questions": [
            {
                "question_id": q.question_id,
                "question_number": q.question_number_from_apeuni,
                "title": q.title,
                "difficulty_level": q.difficulty_level,
                "is_prediction": bool(q.is_prediction),
                "practiced": q.question_id in practiced_ids,
            }
            for q in questions
        ],
        "total": total,
        "page": page,
        "total_pages": total_pages,
        "limit": limit,
    }


@router.post("/start")
def start(
    payload: dict = Body(default={}),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    raw_qid = payload.get("question_id")
    return start_session(
        db=db,
        user_id=current_user.id,
        module="reading",
        question_type="mcq_multiple",
        difficulty_level=payload.get("difficulty_level"),
        question_id=int(raw_qid) if raw_qid is not None else None,
    )


@router.post("/submit")
def submit(
    payload: dict = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session_id = payload["session_id"]
    question_id = int(payload["question_id"])
    selected_options = payload.get("selected_options") or payload.get("selected_option_ids", [])

    session = get_session(session_id)
    question = session["questions"].get(question_id)
    if not question or not question.evaluation:
        raise HTTPException(status_code=404, detail="Question not found")

    scorer = get_scorer("reading_mcm")
    result = scorer.score(
        question_id=question_id,
        session_id=session_id,
        answer={
            "selected_options": selected_options,
            "evaluation_json": question.evaluation.evaluation_json,
        },
    )
    mark_submitted(session_id, question_id, result.pte_score, question_type="reading_mcm")
    persist_answer_to_db(
        session=session, question_id=question_id, question_type="reading_mcm",
        user_answer_json={"selected_options": list(selected_options or [])},
        correct_answer_json={}, result_json=result.breakdown or {}, score=result.pte_score,
    )

    eval_json = question.evaluation.evaluation_json or {}
    correct_answers = eval_json.get("correctAnswers", {}) or {}
    correct_option_ids = list(correct_answers.get("correctOptions", []) or [])
    selected_option_ids = list(selected_options or [])
    is_correct = set(selected_option_ids) == set(correct_option_ids)
    total_score = session.get("score", 0)

    return {
        "pte_score": result.pte_score,
        "is_async": result.is_async,
        "breakdown": result.breakdown,
        "totalScore": total_score,
        # snake_case
        "correct_option_ids": correct_option_ids,
        "selected_option_ids": selected_option_ids,
        "is_correct": is_correct,
        "score_for_question": result.pte_score,
        # camelCase aliases for mobile parity
        "correctOptions": correct_option_ids,
        "selectedOptions": selected_option_ids,
        "isCorrect": is_correct,
        "scoreForQuestion": result.pte_score,
    }

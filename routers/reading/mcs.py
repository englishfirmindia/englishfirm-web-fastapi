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
from schemas.submit_requests import SingleOptionSubmitRequest
from core.logging_config import get_logger
from services.question_search import apply_search_filter

log = get_logger(__name__)

router = APIRouter(prefix="/reading/mcs", tags=["Reading - MCS"])


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
        QuestionFromApeuni.question_type == "mcq_single",
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
    query = apply_search_filter(query, search)

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
        question_type="mcq_single",
        difficulty_level=payload.get("difficulty_level"),
        question_id=int(raw_qid) if raw_qid is not None else None,
    )


@router.post("/submit")
def submit(
    req: SingleOptionSubmitRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session_id = req.session_id
    question_id = req.question_id
    selected_option = req.resolved_option()

    session = get_session(session_id)
    question = session["questions"].get(question_id)
    if not question or not question.evaluation:
        raise HTTPException(status_code=404, detail="Question not found")

    scorer = get_scorer("reading_mcs")
    try:
        result = scorer.score(
            question_id=question_id,
            session_id=session_id,
            answer={
                "selected_option": selected_option,
                "evaluation_json": question.evaluation.evaluation_json,
            },
        )
    except Exception as e:
        # Malformed evaluation_json or any other scorer crash. Surface a
        # clean 500 instead of leaking the traceback; the answer is NOT
        # persisted and NOT marked submitted, so the user can retry.
        log.error(
            "[Reading MCS] scoring failed q=%d sid=%s err=%s: %s",
            question_id, session_id, type(e).__name__, e,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "code": "scoring_failed",
                "message": "We couldn't score your answer. Please try again.",
            },
        )
    mark_submitted(session_id, question_id, result.pte_score, question_type="reading_mcs")

    breakdown = result.breakdown or {}
    correct_option = breakdown.get("correct_option")
    is_correct = bool(breakdown.get("is_correct", False))
    correct_option_ids = [correct_option] if correct_option is not None else []
    total_score = session.get("score", 0)

    response = {
        "pte_score": result.pte_score,
        "is_async": result.is_async,
        "breakdown": breakdown,
        "totalScore": total_score,
        # snake_case
        "correct_option_ids": correct_option_ids,
        "correct_option": correct_option,
        "selected_option": selected_option,
        "is_correct": is_correct,
        "score_for_question": result.pte_score,
        # camelCase aliases for mobile parity
        "correctOption": correct_option,
        "selectedOption": selected_option,
        "isCorrect": is_correct,
        "scoreForQuestion": result.pte_score,
    }

    if req.time_on_question_seconds is not None:
        response["time_on_question_seconds"] = req.time_on_question_seconds
    persist_answer_to_db(
        session=session, question_id=question_id, question_type="reading_mcs",
        user_answer_json={"selected_option": selected_option},
        correct_answer_json={"correct_option": correct_option},
        result_json=response, score=result.pte_score,
    )

    return response

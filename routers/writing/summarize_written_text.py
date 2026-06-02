import math
from typing import Optional
from fastapi import APIRouter, Depends, Body, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import asc, desc

from db.database import get_db
from db.models import User, QuestionFromApeuni
from core.dependencies import get_current_user
from services.billing.enforce_limit import EnforceLimit
from services.session_service import (
    start_session,
    get_session,
    mark_submitted,
    store_score,
    get_score_from_store,
    persist_answer_to_db,
)
from services.scoring import get_scorer
from services.question_search import apply_search_filter
from services.question_list_helper import paginate_by_practice_recency, iso, practiced_questions_subq, practiced_question_ids_in
from schemas.submit_requests import TextSubmitRequest

router = APIRouter(prefix="/writing/summarize-written-text", tags=["Writing - Summarize Written Text"])


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
        QuestionFromApeuni.module == "writing",
        QuestionFromApeuni.question_type == "summarize_written_text",
    )
    if difficulty is not None:
        query = query.filter(QuestionFromApeuni.difficulty_level == difficulty)
    if is_prediction is not None:
        query = query.filter(QuestionFromApeuni.is_prediction == is_prediction)
    if practiced is not None:
        practiced_subq = practiced_questions_subq(db, current_user.id, "summarize_written_text")
        if practiced:
            query = query.filter(QuestionFromApeuni.question_id.in_(practiced_subq))
        else:
            query = query.filter(~QuestionFromApeuni.question_id.in_(practiced_subq))
    query = apply_search_filter(query, search)

    if sort == 'recent':
        questions, total, recency = paginate_by_practice_recency(
            db=db,
            filtered_query=query,
            user_id=current_user.id,
            question_type="summarize_written_text",
            page=page,
            limit=limit,
        )
    else:
        total = query.count()
        order_dir = desc if sort == 'desc' else asc
        questions = (
            query
            .order_by(order_dir(QuestionFromApeuni.question_id))
            .offset((page - 1) * limit)
            .limit(limit)
            .all()
        )
        recency = {}
    total_pages = math.ceil(total / limit) if total > 0 else 1

    page_qids = [q.question_id for q in questions]
    practiced_ids = practiced_question_ids_in(db, current_user.id, "summarize_written_text", page_qids)

    return {
        "questions": [
            {
                "question_id": q.question_id,
                "question_number": q.question_number_from_apeuni,
                "title": q.title,
                "difficulty_level": q.difficulty_level,
                "is_prediction": bool(q.is_prediction),
                "practiced": q.question_id in practiced_ids,
                "last_practiced_at": iso(recency.get(q.question_id)),
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
        module="writing",
        question_type="summarize_written_text",
        difficulty_level=payload.get("difficulty_level"),
        question_id=int(raw_qid) if raw_qid is not None else None,
    )


@router.post("/submit")
def submit(
    req: TextSubmitRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    _gate=Depends(EnforceLimit("practice")),
):
    session_id = req.session_id
    question_id = req.question_id
    user_answer = req.resolved_text()

    session = get_session(session_id)
    question = session["questions"].get(question_id)
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")

    # Extract prompt text from content_json for AI context
    content_json = question.content_json or {}
    prompt = content_json.get("passage", content_json.get("text", ""))

    scorer = get_scorer("summarize_written_text")
    try:
        result = scorer.score(
            question_id=question_id,
            session_id=session_id,
            answer={
                "text": user_answer,
                "prompt": prompt,
            },
        )
        mark_submitted(session_id, question_id, result.pte_score)
        status_str = "failed" if result.error else "complete"
        persist_answer_to_db(
            session=session, question_id=question_id, question_type="summarize_written_text",
            user_answer_json={"text": user_answer},
            correct_answer_json={},
            result_json={
                "pte_score": result.pte_score,
                "breakdown": result.breakdown,
                "error": result.error,
                "maxScore": result.breakdown.get("max_pts", 10),
            },
            score=result.breakdown.get("earned", 0),
        )
        store_score(current_user.id, question_id, {
            "status": status_str,
            "scoring": status_str,
            "pte_score": result.pte_score,
            "scoreForQuestion": result.pte_score,
            "breakdown": result.breakdown,
            "subScores": result.breakdown,
            "error": result.error,
        })
        return {
            "pte_score": result.pte_score,
            "is_async": result.is_async,
            "breakdown": result.breakdown,
            "totalScore": session.get("score", 0),
            "error": result.error,
        }
    except Exception as e:
        store_score(current_user.id, question_id, {
            "status": "failed",
            "scoring": "error",
            "pte_score": 10,
            "scoreForQuestion": 10,
            "breakdown": {},
            "subScores": {},
            "error": str(e),
        })
        raise


@router.get("/score/{question_id}")
def poll_score(
    question_id: int,
    current_user: User = Depends(get_current_user),
):
    result = get_score_from_store(current_user.id, question_id)
    if not result:
        return {"status": "pending", "scoring": "pending"}
    return result

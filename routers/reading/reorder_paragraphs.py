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
from schemas.submit_requests import SequenceSubmitRequest
from core.logging_config import get_logger
from services.question_search import apply_search_filter

log = get_logger(__name__)

router = APIRouter(prefix="/reading/reorder-paragraphs", tags=["Reading - Reorder Paragraphs"])


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
        QuestionFromApeuni.question_type == "reorder_paragraphs",
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
        question_type="reorder_paragraphs",
        difficulty_level=payload.get("difficulty_level"),
        question_id=int(raw_qid) if raw_qid is not None else None,
    )


@router.post("/submit")
def submit(
    req: SequenceSubmitRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session_id = req.session_id
    question_id = req.question_id
    user_sequence = req.resolved_sequence()

    session = get_session(session_id)
    question = session["questions"].get(question_id)
    if not question or not question.evaluation:
        raise HTTPException(status_code=404, detail="Question not found")

    # The frontend ships the user's order as a list of paragraph TEXTS,
    # but correctSequence in evaluation_json is paragraph IDs (e.g. ["2",
    # "4", "3"]). The rule scorer compares element-by-element, so passing
    # texts produced pair_results=[False, ...] on every submission — 100%
    # of historic RP attempts sat at PTE floor 10 because of this.
    # Map text → id here so the scorer sees ID lists on both sides.
    paragraphs = ((question.content_json or {}).get("paragraphs") or [])
    text_to_id = {}
    for p in paragraphs:
        if isinstance(p, dict):
            pid = p.get("id") or p.get("paragraphId")
            ptext = (p.get("text") or "").strip()
            if pid is not None and ptext:
                text_to_id[ptext] = str(pid)
    user_sequence_ids = [
        text_to_id.get((s or "").strip(), s) for s in user_sequence
    ]

    scorer = get_scorer("reorder_paragraphs")
    try:
        result = scorer.score(
            question_id=question_id,
            session_id=session_id,
            answer={
                "user_sequence": user_sequence_ids,
                "evaluation_json": question.evaluation.evaluation_json,
            },
        )
    except Exception as e:
        log.error(
            "[Reading Reorder] scoring failed q=%d sid=%s err=%s: %s",
            question_id, session_id, type(e).__name__, e,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "code": "scoring_failed",
                "message": "We couldn't score your answer. Please try again.",
            },
        )
    eval_json = question.evaluation.evaluation_json or {}
    correct_answers = eval_json.get("correctAnswers", {}) or {}
    correct_sequence = list(correct_answers.get("correctSequence", []) or [])

    breakdown = result.breakdown or {}
    pair_results = list(breakdown.get("pair_results", []) or [])
    is_correct = bool(pair_results) and all(pair_results)
    total_score = session.get("score", 0)

    mark_submitted(session_id, question_id, result.pte_score)
    persisted_result = {
        **breakdown,
        "correct_sequence": correct_sequence,
        "pair_results": pair_results,
        "is_correct": is_correct,
    }
    if req.time_on_question_seconds is not None:
        persisted_result["time_on_question_seconds"] = req.time_on_question_seconds
    persist_answer_to_db(
        session=session, question_id=question_id, question_type="reorder_paragraphs",
        user_answer_json={"user_sequence": list(user_sequence or [])},
        correct_answer_json={"correct_sequence": correct_sequence},
        result_json=persisted_result,
        score=result.pte_score,
    )

    return {
        "pte_score": result.pte_score,
        "is_async": result.is_async,
        "breakdown": breakdown,
        "totalScore": total_score,
        # snake_case
        "correct_sequence": correct_sequence,
        "user_sequence": list(user_sequence or []),
        "pair_results": pair_results,
        "is_correct": is_correct,
        "score_for_question": result.pte_score,
        # camelCase aliases for mobile parity
        "correctSequence": correct_sequence,
        "pairResults": pair_results,
        "isCorrect": is_correct,
        "scoreForQuestion": result.pte_score,
    }

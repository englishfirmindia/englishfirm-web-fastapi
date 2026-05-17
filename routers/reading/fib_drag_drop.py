"""
Reading FIB Dropdown router.

NAMING NOTE — historical: the file/URL/tag say "fib-drag-drop" but this
router actually serves the dropdown variant (`reading_fib_drop_down`).
The web frontend's `app_config.dart` is wired to POST dropdown practice
answers here, so the URL is locked in as a public contract.
True drag-and-drop questions (`reading_drag_and_drop`) are served by
`fill_in_blanks.py`.
"""

import math
from typing import Optional
from fastapi import APIRouter, Depends, Body, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import asc, desc

from db.database import get_db
from db.models import User, QuestionFromApeuni, UserQuestionAttempt, QuestionExplanation
from core.dependencies import get_current_user
from services.session_service import start_session, get_session, mark_submitted, persist_answer_to_db
from services.scoring import get_scorer
from schemas.submit_requests import AnswersDictSubmitRequest
from core.logging_config import get_logger
from services.question_search import apply_search_filter

log = get_logger(__name__)


def _get_or_generate_explanations(
    db: Session,
    question,
    question_type: str,
    correct_answers: dict,
    user_answers: dict,
    blank_results: dict,
) -> list:
    """Cache-aside lookup for FIB explanations.
      1. SELECT from `question_explanations` by question_id — hit returns cached.
      2. Miss → call Haiku (GPT-4o fallback) via fib_explainer.
      3. Non-empty result → INSERT into the cache table and return.
      4. Empty result (both LLMs failed) → return [] without caching so the
         next submit retries; never lets transient failures stick.
    """
    qid = int(question.question_id)
    try:
        row = db.query(QuestionExplanation).filter_by(question_id=qid).one_or_none()
        if row is not None and row.explanations:
            return row.explanations
    except Exception as e:
        log.warning("[FIB explanations] cache lookup failed q=%s: %s", qid, e)

    try:
        from services.fib_explainer import build_passage, generate_fib_explanations
        passage_text = build_passage(question.content_json or {})
        blanks = [
            {
                "blank_id": str(bid),
                "correct": correct_answers.get(str(bid)),
                "user_answer": user_answers.get(str(bid)) or user_answers.get(f"blank_{bid}"),
                "is_correct": bool(blank_results.get(str(bid))),
            }
            for bid in correct_answers.keys()
        ]
        if not passage_text or not blanks:
            return []
        result = generate_fib_explanations(passage_text, blanks)
    except Exception as e:
        log.warning("[FIB explanations] generation failed q=%s: %s", qid, e)
        return []

    # Only cache when we got at least one non-empty explanation back.
    has_content = any(
        isinstance(r, dict) and str(r.get("explanation") or "").strip()
        for r in result
    )
    if has_content:
        try:
            scorer = next((r.get("scorer") for r in result if isinstance(r, dict) and r.get("scorer")), None)
            db.add(QuestionExplanation(
                question_id=qid,
                question_type=question_type,
                explanations=result,
                scorer=scorer,
            ))
            db.commit()
        except Exception as e:
            db.rollback()
            log.warning("[FIB explanations] cache insert failed q=%s: %s", qid, e)
    return result


router = APIRouter(prefix="/reading/fib-drag-drop", tags=["Reading - FIB Drag & Drop"])


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
        QuestionFromApeuni.question_type == "reading_fib_drop_down",
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
        question_type="reading_fib_drop_down",
        difficulty_level=payload.get("difficulty_level"),
        question_id=int(raw_qid) if raw_qid is not None else None,
    )


@router.post("/submit")
def submit(
    req: AnswersDictSubmitRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session_id = req.session_id
    question_id = req.question_id
    raw = req.resolved_raw()
    if isinstance(raw, list):
        user_answers = {f"blank_{i + 1}": v for i, v in enumerate(raw)}
    elif isinstance(raw, dict):
        # Normalise plain numeric keys ("1", "2") to "blank_N" to match evaluation_json
        user_answers = {}
        for k, v in raw.items():
            key = str(k)
            user_answers[key if key.startswith("blank_") else f"blank_{key}"] = v
    else:
        user_answers = {}

    session = get_session(session_id)
    question = session["questions"].get(question_id)
    if not question or not question.evaluation:
        raise HTTPException(status_code=404, detail="Question not found")

    scorer = get_scorer("reading_fib_drop_down")
    try:
        result = scorer.score(
            question_id=question_id,
            session_id=session_id,
            answer={
                "user_answers": user_answers,
                "evaluation_json": question.evaluation.evaluation_json,
            },
        )
    except Exception as e:
        log.error(
            "[Reading FIB-DD] scoring failed q=%d sid=%s err=%s: %s",
            question_id, session_id, type(e).__name__, e,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "code": "scoring_failed",
                "message": "We couldn't score your answer. Please try again.",
            },
        )
    mark_submitted(session_id, question_id, result.pte_score)

    eval_json = question.evaluation.evaluation_json or {}
    correct_answers_raw = eval_json.get("correctAnswers", {}) or {}
    if isinstance(correct_answers_raw, dict) and correct_answers_raw.get("blanks") is not None:
        correct_answers = {
            str(b.get("blankId")): b.get("answer")
            for b in correct_answers_raw.get("blanks", [])
        }
    else:
        correct_answers = {str(k): v for k, v in correct_answers_raw.items()}

    breakdown = result.breakdown or {}
    blank_results = breakdown.get("blank_results", {}) or {}
    is_correct = bool(blank_results) and all(blank_results.values())

    # ── AI explanations (lazy cache: RDS → LLM on miss → write-through) ──
    explanations = _get_or_generate_explanations(
        db=db,
        question=question,
        question_type="reading_fib_drop_down",
        correct_answers=correct_answers,
        user_answers=user_answers,
        blank_results=blank_results,
    )

    persisted_result = {
        **breakdown,
        "correct_answers": correct_answers,
        "pte_score": result.pte_score,
        "is_correct": is_correct,
        "explanations": explanations,
    }
    if req.time_on_question_seconds is not None:
        persisted_result["time_on_question_seconds"] = req.time_on_question_seconds
    persist_answer_to_db(
        session=session, question_id=question_id, question_type="reading_fib_drop_down",
        user_answer_json={"user_answers": user_answers},
        correct_answer_json={},
        result_json=persisted_result,
        score=result.pte_score,
    )
    total_score = session.get("score", 0)

    return {
        "pte_score": result.pte_score,
        "is_async": result.is_async,
        "breakdown": breakdown,
        "totalScore": total_score,
        # snake_case
        "correct_answers": correct_answers,
        "user_answers": user_answers,
        "blank_results": blank_results,
        "is_correct": is_correct,
        "score_for_question": result.pte_score,
        "explanations": explanations,
        # camelCase aliases for mobile parity
        "correctAnswers": correct_answers,
        "blankResults": blank_results,
        "isCorrect": is_correct,
        "scoreForQuestion": result.pte_score,
    }

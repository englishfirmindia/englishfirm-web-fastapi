import math
import re
from typing import Optional
from fastapi import APIRouter, Depends, Body, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import asc, desc

from db.database import get_db
from db.models import User, QuestionFromApeuni
from core.dependencies import get_current_user
from services.billing.enforce_limit import EnforceLimit
from services.session_service import start_session, get_session, mark_submitted, persist_answer_to_db, ACTIVE_SESSIONS
from services.scoring import get_scorer
from services.s3_service import generate_presigned_url
from schemas.submit_requests import HIWSubmitRequest
from core.logging_config import get_logger
from services.question_search import apply_search_filter
from services.question_list_helper import paginate_by_practice_recency, iso

log = get_logger(__name__)

router = APIRouter(prefix="/listening/hiw", tags=["Listening - Highlight Incorrect Words"])


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
        QuestionFromApeuni.module == "listening",
        QuestionFromApeuni.question_type == "highlight_incorrect_words",
    )
    if difficulty is not None:
        query = query.filter(QuestionFromApeuni.difficulty_level == difficulty)
    if is_prediction is not None:
        query = query.filter(QuestionFromApeuni.is_prediction == is_prediction)
    if practiced is not None:
        practiced_subq = practiced_questions_subq(db, current_user.id, "highlight_incorrect_words")
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
            question_type="highlight_incorrect_words",
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
    practiced_ids = practiced_question_ids_in(db, current_user.id, "highlight_incorrect_words", page_qids)

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
    result = start_session(
        db=db,
        user_id=current_user.id,
        module="listening",
        question_type="highlight_incorrect_words",
        difficulty_level=payload.get("difficulty_level"),
        question_id=int(raw_qid) if raw_qid is not None else None,
    )
    # Override stored question_type so attempt_answers uses "listening_hiw"
    # (the value that fetchAnsweredQuestionIds queries for)
    ACTIVE_SESSIONS[result["session_id"]]["question_type"] = "listening_hiw"
    ACTIVE_SESSIONS.save(result["session_id"])
    return result


@router.post("/submit")
def submit(
    req: HIWSubmitRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    _gate=Depends(EnforceLimit("practice")),
):
    session_id = req.session_id
    question_id = req.question_id
    session = get_session(session_id)
    question = session["questions"].get(question_id)
    if not question or not question.evaluation:
        raise HTTPException(status_code=404, detail="Question not found")

    # Accept word strings or convert from 0-based word indices.
    # We persist BOTH below so on resume/review the frontend can restore
    # the exact positions the user clicked — string-only is ambiguous when
    # the passage has duplicate words.
    highlighted_words = req.highlighted_words
    highlighted_indices = req.highlighted_indices
    content = question.content_json or {}
    words = content.get("words") or (content.get("transcript") or content.get("passage") or content.get("text") or "").split()

    if highlighted_words is None:
        indices = highlighted_indices or []
        highlighted_words = [words[i] for i in indices if isinstance(i, int) and i < len(words)]
    if highlighted_indices is None and highlighted_words:
        # Words-only payload (legacy clients): best-effort leftmost-match.
        # Still vulnerable to duplicates here, but at least we freeze the
        # positions at submit time so resume/review stay consistent.
        derived: list[int] = []
        search_from = 0
        for w in highlighted_words:
            for i in range(search_from, len(words)):
                if words[i] == w:
                    derived.append(i)
                    search_from = i + 1
                    break
        highlighted_indices = derived

    eval_json = question.evaluation.evaluation_json or {}
    correct_answers = eval_json.get("correctAnswers", {}) or {}
    incorrect_words_raw = correct_answers.get("incorrectWords", []) or []
    # Support both plain string list and dict list {wrong, correct, index}.
    # When the dict shape is present we also capture wrong → correct mappings
    # so the UI can render APEUni-style inline corrections.
    incorrect_words = [
        (w["wrong"] if isinstance(w, dict) else w)
        for w in incorrect_words_raw
    ]
    wrong_to_correct: dict = {}
    for w in incorrect_words_raw:
        if isinstance(w, dict) and w.get("wrong") and w.get("correct"):
            wrong_to_correct[str(w["wrong"]).strip().lower()] = str(w["correct"]).strip()

    scorer = get_scorer("listening_hiw")
    try:
        result = scorer.score(
            question_id=question_id,
            session_id=session_id,
            answer={
                "highlighted_words": highlighted_words,
                "evaluation_json": {**eval_json, "correctAnswers": {**correct_answers, "incorrectWords": incorrect_words}},
            },
        )
    except Exception as e:
        log.error(
            "[Listening HIW] scoring failed q=%s sid=%s err=%s: %s",
            question_id, session_id, type(e).__name__, e,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "code": "scoring_failed",
                "message": "We couldn't score your answer. Please try again.",
            },
        )

    breakdown = result.breakdown or {}
    correct_clicks = list(breakdown.get("correct_clicks", []) or [])
    incorrect_clicks = list(breakdown.get("incorrect_clicks", []) or [])
    missed_words = list(breakdown.get("missed_words", []) or [])
    is_correct = len(incorrect_clicks) == 0 and len(missed_words) == 0

    # Compute which indices in the passage word array are the actual incorrect words
    def _norm(w: str) -> str:
        return re.sub(r'[^\w]', '', w).lower()

    incorrect_words_set = {_norm(w) for w in incorrect_words}
    incorrect_word_indices = [i for i, w in enumerate(words) if _norm(w) in incorrect_words_set]

    # Build index → correct-word map for inline corrections, when available.
    # Keyed by string index for JSON safety (Flutter parses as Map<String,String>).
    corrections: dict = {}
    if wrong_to_correct:
        for i, w in enumerate(words):
            cw = wrong_to_correct.get(_norm(w))
            if cw:
                corrections[str(i)] = cw

    mark_submitted(session_id, question_id, result.pte_score)
    persist_answer_to_db(
        session=session, question_id=question_id, question_type="listening_hiw",
        user_answer_json={
            "highlighted_words":   list(highlighted_words or []),
            "highlighted_indices": list(highlighted_indices or []),
        },
        correct_answer_json={},
        result_json={
            **breakdown,
            "incorrect_words": incorrect_words,
            "incorrect_word_indices": incorrect_word_indices,
            "corrections": corrections,
            "is_correct": is_correct,
        },
        score=result.pte_score,
    )

    total_score = session.get("score", 0)

    return {
        "pte_score": result.pte_score,
        "is_async": result.is_async,
        "breakdown": breakdown,
        "totalScore": total_score,
        # snake_case
        "incorrect_words": incorrect_words,
        "incorrect_word_indices": incorrect_word_indices,
        "corrections": corrections,
        "highlighted_words": highlighted_words,
        "correct_clicks": correct_clicks,
        "incorrect_clicks": incorrect_clicks,
        "missed_words": missed_words,
        "is_correct": is_correct,
        "score_for_question": result.pte_score,
        # camelCase aliases for mobile parity
        "incorrectWords": incorrect_words,
        "correctClicks": correct_clicks,
        "incorrectClicks": incorrect_clicks,
        "missedWords": missed_words,
        "isCorrect": is_correct,
        "scoreForQuestion": result.pte_score,
    }


@router.get("/audio-url")
def audio_url(
    s3_url: str,
    current_user: User = Depends(get_current_user),
):
    return {"presigned_url": generate_presigned_url(s3_url)}

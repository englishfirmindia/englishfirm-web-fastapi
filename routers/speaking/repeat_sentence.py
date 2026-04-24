import math
from typing import Optional
import uuid
import logging
from fastapi import APIRouter, Depends, Body, Query
from sqlalchemy.orm import Session
from sqlalchemy import asc, desc

logger = logging.getLogger(__name__)

from db.database import get_db
from db.models import User, QuestionFromApeuni, UserQuestionAttempt
from core.dependencies import get_current_user
from services.session_service import start_session, get_session, mark_submitted, get_score_from_store, store_score, persist_speaking_answer_pending
from services.scoring import get_scorer
from services.s3_service import generate_presigned_url, generate_presigned_upload_url

router = APIRouter(prefix="/speaking/repeat-sentence", tags=["Speaking - Repeat Sentence"])


def _kick_off_azure(task_type: str, question_id: int, audio_url: str, user_id: int, reference_text: str = "") -> None:
    from services.speaking_scorer import kick_off_scoring
    kick_off_scoring(user_id, question_id, task_type, audio_url, reference_text)



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
        QuestionFromApeuni.module == "speaking",
        QuestionFromApeuni.question_type == "repeat_sentence",
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
        module="speaking",
        question_type="repeat_sentence",
        difficulty_level=payload.get("difficulty_level"),
        question_id=int(raw_qid) if raw_qid is not None else None,
    )


@router.post("/get-upload-url")
def get_upload_url(
    payload: dict = Body(...),
    current_user: User = Depends(get_current_user),
):
    question_id = payload.get("question_id", "unknown")
    key = f"recordings/{current_user.id}/repeat_sentence/{question_id}/{uuid.uuid4()}.aac"
    return generate_presigned_upload_url(key)


@router.post("/submit")
def submit(
    payload: dict = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    logger.info("[RS submit] payload keys=%s session_id=%s question_id=%s",
                list(payload.keys()), payload.get("session_id"), payload.get("question_id"))
    try:
        session_id = payload["session_id"]
        question_id = int(payload["question_id"])
        audio_url = payload["audio_url"]
    except Exception as e:
        logger.error("[RS submit] payload parse error: %s", e)
        raise

    try:
        session = get_session(session_id)
    except Exception as e:
        logger.error("[RS submit] get_session failed: %s", e)
        raise
    try:
        q_obj = session["questions"].get(question_id)
        reference_text = (q_obj.content_json or {}).get("transcript", "") if q_obj else ""
        scorer = get_scorer("repeat_sentence")
        scorer.score(
            question_id=question_id,
            session_id=session_id,
            answer={
                "audio_url": audio_url,
                "kick_off_fn": lambda t, q, u: _kick_off_azure(t, q, u, current_user.id, reference_text),
            },
        )
        mark_submitted(session_id, question_id, 0)
        persist_speaking_answer_pending(session, question_id, "repeat_sentence", audio_url)
    except Exception as e:
        logger.error("[RS submit] scoring/mark error: %s", e, exc_info=True)
        raise

    return {"message": "submitted", "scoring_status": "pending"}


@router.get("/score/{question_id}")
def poll_score(
    question_id: int,
    current_user: User = Depends(get_current_user),
):
    result = get_score_from_store(current_user.id, question_id)
    if not result:
        return {"scoring": "pending"}
    return result


@router.get("/audio-url")
def audio_url(
    s3_url: str,
    current_user: User = Depends(get_current_user),
):
    return {"presigned_url": generate_presigned_url(s3_url)}

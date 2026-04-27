import math
from typing import Optional
import uuid
from fastapi import APIRouter, Depends, Body, Query
from sqlalchemy.orm import Session
from sqlalchemy import asc, desc

from db.database import get_db
from db.models import User, QuestionFromApeuni, UserQuestionAttempt
from core.dependencies import get_current_user
from services.session_service import start_session, get_session, mark_submitted, get_score_from_store, persist_speaking_answer_pending, store_score
from services.scoring import get_scorer
from services.s3_service import generate_presigned_url, generate_presigned_upload_url

router = APIRouter(prefix="/speaking/respond-to-situation", tags=["Speaking - Respond to Situation"])


def _kick_off_azure(task_type: str, question_id: int, audio_url: str, user_id: int,
                    key_points: list = None, stimulus_audio_url: str = "") -> None:
    from services.speaking_scorer import kick_off_scoring
    kick_off_scoring(user_id, question_id, task_type, audio_url,
                     key_points=key_points or [], stimulus_audio_url=stimulus_audio_url)



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
        QuestionFromApeuni.question_type == "ptea_respond_situation",
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
        question_type="ptea_respond_situation",
        difficulty_level=payload.get("difficulty_level"),
        question_id=int(raw_qid) if raw_qid is not None else None,
    )


@router.post("/get-upload-url")
def get_upload_url(
    payload: dict = Body(...),
    current_user: User = Depends(get_current_user),
):
    question_id = payload.get("question_id", "unknown")
    key = f"recordings/{current_user.id}/respond_to_situation/{question_id}/{uuid.uuid4()}.aac"
    return generate_presigned_upload_url(key)


@router.post("/submit")
def submit(
    payload: dict = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session_id = payload["session_id"]
    question_id = int(payload["question_id"])
    audio_url = payload["audio_url"]

    session = get_session(session_id)
    q_obj = session["questions"].get(question_id)
    key_points = []
    stimulus_audio_url = ""
    if q_obj:
        stimulus_audio_url = (q_obj.content_json or {}).get("audio_url", "")
        if q_obj.evaluation:
            ca = (q_obj.evaluation.evaluation_json or {}).get("correctAnswers", {})
            key_points = ca.get("keyPoints") or []

    scorer = get_scorer("respond_to_situation")
    scorer.score(
        question_id=question_id,
        session_id=session_id,
        answer={
            "audio_url": audio_url,
            "kick_off_fn": lambda t, q, u: _kick_off_azure(
                t, q, u, current_user.id, key_points, stimulus_audio_url),
        },
    )
    mark_submitted(session_id, question_id, 0)
    persist_speaking_answer_pending(session, question_id, "respond_to_situation", audio_url)

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

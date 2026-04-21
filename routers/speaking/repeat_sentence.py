import uuid
import logging
from fastapi import APIRouter, Depends, Body
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

from db.database import get_db
from db.models import User
from core.dependencies import get_current_user
from services.session_service import start_session, get_session, mark_submitted, get_score_from_store, store_score
from services.scoring import get_scorer
from services.s3_service import generate_presigned_url, generate_presigned_upload_url

router = APIRouter(prefix="/speaking/repeat-sentence", tags=["Speaking - Repeat Sentence"])


def _kick_off_azure(task_type: str, question_id: int, audio_url: str, user_id: int, reference_text: str = "") -> None:
    from services.speaking_scorer import kick_off_scoring
    kick_off_scoring(user_id, question_id, task_type, audio_url, reference_text)



@router.post("/start")
def start(
    payload: dict = Body(default={}),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return start_session(
        db=db,
        user_id=current_user.id,
        module="speaking",
        question_type="repeat_sentence",
        difficulty_level=payload.get("difficulty_level"),
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

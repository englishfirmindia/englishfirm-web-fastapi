import uuid
from fastapi import APIRouter, Depends, Body
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import User
from core.dependencies import get_current_user
from services.session_service import start_session, get_session, mark_submitted, get_score_from_store, store_score
from services.scoring import get_scorer
from services.s3_service import generate_presigned_url, generate_presigned_upload_url

router = APIRouter(prefix="/speaking/respond-to-situation", tags=["Speaking - Respond to Situation"])


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
        question_type="respond_to_situation",
        difficulty_level=payload.get("difficulty_level"),
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

    scorer = get_scorer("respond_to_situation")
    scorer.score(
        question_id=question_id,
        session_id=session_id,
        answer={
            "audio_url": audio_url,
            "kick_off_fn": lambda t, q, u: _kick_off_azure(t, q, u, current_user.id),
        },
    )
    mark_submitted(session_id, question_id, 0)

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

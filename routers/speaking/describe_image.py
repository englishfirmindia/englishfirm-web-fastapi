import threading
import uuid
from fastapi import APIRouter, Depends, Body
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import User
from core.dependencies import get_current_user
from services.session_service import start_session, get_session, mark_submitted, get_score_from_store, store_score
from services.scoring import get_scorer
from services.s3_service import generate_presigned_upload_url

router = APIRouter(prefix="/speaking/describe-image", tags=["Speaking - Describe Image"])


def _kick_off_azure(task_type: str, question_id: int, audio_url: str, user_id: int) -> None:
    def _run():
        try:
            store_score(user_id, question_id, {
                "scoring": "complete",
                "pte_score": 0,
                "content": 0,
                "fluency": 0,
                "pronunciation": 0,
            })
        except Exception as e:
            store_score(user_id, question_id, {"scoring": "error", "error": str(e)})

    t = threading.Thread(target=_run, daemon=True)
    t.start()


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
        question_type="describe_image",
        difficulty_level=payload.get("difficulty_level"),
    )


@router.post("/get-upload-url")
def get_upload_url(
    payload: dict = Body(...),
    current_user: User = Depends(get_current_user),
):
    question_id = payload.get("question_id", "unknown")
    key = f"recordings/{current_user.id}/describe_image/{question_id}/{uuid.uuid4()}.aac"
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

    scorer = get_scorer("describe_image")
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


@router.get("/image-url")
def image_url(
    s3_url: str,
    current_user: User = Depends(get_current_user),
):
    from services.s3_service import generate_presigned_url
    return {"presigned_url": generate_presigned_url(s3_url)}

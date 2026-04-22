import uuid
from fastapi import APIRouter, Depends, Body
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import User
from core.dependencies import get_current_user
from services.session_service import start_session, get_session, mark_submitted, get_score_from_store, persist_speaking_answer_pending, store_score
from services.scoring import get_scorer
from services.s3_service import generate_presigned_upload_url

router = APIRouter(prefix="/speaking/describe-image", tags=["Speaking - Describe Image"])


def _kick_off_azure(task_type: str, question_id: int, audio_url: str, user_id: int, key_points: list = None) -> None:
    from services.speaking_scorer import kick_off_scoring
    kick_off_scoring(user_id, question_id, task_type, audio_url, key_points=key_points or [])



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
    q_obj = session["questions"].get(question_id)
    key_points = []
    if q_obj and q_obj.evaluation:
        ca = (q_obj.evaluation.evaluation_json or {}).get("correctAnswers", {})
        key_points = ca.get("keyPoints") or []

    scorer = get_scorer("describe_image")
    scorer.score(
        question_id=question_id,
        session_id=session_id,
        answer={
            "audio_url": audio_url,
            "kick_off_fn": lambda t, q, u: _kick_off_azure(t, q, u, current_user.id, key_points),
        },
    )
    mark_submitted(session_id, question_id, 0)
    persist_speaking_answer_pending(session, question_id, "describe_image", audio_url)

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


@router.get("/image-url-by-id/{question_id}")
def image_url_by_id(
    question_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    from fastapi import HTTPException
    from services.s3_service import generate_presigned_url
    from db.models import QuestionFromApeuni

    q = (
        db.query(QuestionFromApeuni)
        .filter(QuestionFromApeuni.question_id == question_id)
        .first()
    )
    if not q:
        raise HTTPException(status_code=404, detail="Question not found")
    content = q.content_json or {}
    # Content shape may store the image path under "image_url" or "s3_url"
    s3_url = content.get("image_url") or content.get("s3_url") or ""
    if not s3_url:
        raise HTTPException(
            status_code=404,
            detail="No image URL on this question",
        )
    return {"presigned_url": generate_presigned_url(s3_url)}

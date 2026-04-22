import uuid
from fastapi import APIRouter, Depends, Body
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import User
from core.dependencies import get_current_user
from services.session_service import start_session, get_session, mark_submitted, get_score_from_store, persist_speaking_answer_pending, store_score
from services.scoring import get_scorer
from services.s3_service import generate_presigned_url, generate_presigned_upload_url

router = APIRouter(prefix="/speaking/summarize-group-discussion", tags=["Speaking - Summarize Group Discussion"])


def _kick_off_azure(task_type: str, question_id: int, audio_url: str, user_id: int,
                    key_points: list = None, stimulus_audio_url: str = "") -> None:
    from services.speaking_scorer import kick_off_scoring
    kick_off_scoring(user_id, question_id, task_type, audio_url,
                     key_points=key_points or [], stimulus_audio_url=stimulus_audio_url)



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
        question_type="summarize_group_discussion",
        difficulty_level=payload.get("difficulty_level"),
    )


@router.post("/get-upload-url")
def get_upload_url(
    payload: dict = Body(...),
    current_user: User = Depends(get_current_user),
):
    question_id = payload.get("question_id", "unknown")
    key = f"recordings/{current_user.id}/summarize_group_discussion/{question_id}/{uuid.uuid4()}.aac"
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

    scorer = get_scorer("summarize_group_discussion")
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
    persist_speaking_answer_pending(session, question_id, "summarize_group_discussion", audio_url)

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

from fastapi import APIRouter, Depends, Body, HTTPException
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import User
from core.dependencies import get_current_user
from services.session_service import (
    start_session,
    get_session,
    mark_submitted,
    store_score,
    get_score_from_store,
    persist_answer_to_db,
)
from services.scoring import get_scorer
from services.s3_service import generate_presigned_url

router = APIRouter(prefix="/listening/sst", tags=["Listening - Summarize Spoken Text"])


@router.post("/start")
def start(
    payload: dict = Body(default={}),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return start_session(
        db=db,
        user_id=current_user.id,
        module="listening",
        question_type="summarize_spoken_text",
        difficulty_level=payload.get("difficulty_level"),
    )


@router.post("/submit")
def submit(
    payload: dict = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session_id = payload["session_id"]
    question_id = int(payload["question_id"])
    user_answer = payload.get("user_answer") or payload.get("text", "")

    session = get_session(session_id)
    question = session["questions"].get(question_id)
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")

    content_json = question.content_json or {}
    prompt = content_json.get("transcript", content_json.get("audio_url", ""))

    scorer = get_scorer("listening_sst")
    try:
        result = scorer.score(
            question_id=question_id,
            session_id=session_id,
            answer={
                "text": user_answer,
                "prompt": prompt,
            },
        )
        mark_submitted(session_id, question_id, result.pte_score)
        status_str = "failed" if result.error else "complete"
        persist_answer_to_db(
            session=session, question_id=question_id, question_type="listening_sst",
            user_answer_json={"text": user_answer},
            correct_answer_json={},
            result_json={"pte_score": result.pte_score, "breakdown": result.breakdown, "error": result.error},
            score=result.pte_score,
        )
        store_score(current_user.id, question_id, {
            "status": status_str,
            "scoring": status_str,
            "pte_score": result.pte_score,
            "scoreForQuestion": result.pte_score,
            "breakdown": result.breakdown,
            "subScores": result.breakdown,
            "error": result.error,
        })
        return {
            "pte_score": result.pte_score,
            "is_async": result.is_async,
            "breakdown": result.breakdown,
            "totalScore": session.get("score", 0),
            "error": result.error,
        }
    except Exception as e:
        store_score(current_user.id, question_id, {
            "status": "failed",
            "scoring": "error",
            "pte_score": 10,
            "scoreForQuestion": 10,
            "breakdown": {},
            "subScores": {},
            "error": str(e),
        })
        raise


@router.get("/score/{question_id}")
def poll_score(
    question_id: int,
    current_user: User = Depends(get_current_user),
):
    result = get_score_from_store(current_user.id, question_id)
    if not result:
        return {"status": "pending", "scoring": "pending"}
    return result


@router.get("/audio-url")
def audio_url(
    s3_url: str,
    current_user: User = Depends(get_current_user),
):
    return {"presigned_url": generate_presigned_url(s3_url)}

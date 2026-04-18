from fastapi import APIRouter, Depends, Body, HTTPException
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import User
from core.dependencies import get_current_user
from services.session_service import start_session, get_session, mark_submitted
from services.scoring import get_scorer
from services.s3_service import generate_presigned_url

router = APIRouter(prefix="/listening/wfd", tags=["Listening - Write from Dictation"])


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
        question_type="listening_wfd",
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
    user_text = payload.get("user_text") or payload.get("text", "")

    session = get_session(session_id)
    question = session["questions"].get(question_id)
    if not question or not question.evaluation:
        raise HTTPException(status_code=404, detail="Question not found")

    scorer = get_scorer("listening_wfd")
    result = scorer.score(
        question_id=question_id,
        session_id=session_id,
        answer={
            "user_text": user_text,
            "evaluation_json": question.evaluation.evaluation_json,
        },
    )
    mark_submitted(session_id, question_id, result.pte_score)

    eval_json = question.evaluation.evaluation_json or {}
    correct_answers = eval_json.get("correctAnswers", {}) or {}
    transcript = correct_answers.get("transcript", "") or ""

    breakdown = result.breakdown or {}
    word_results = breakdown.get("word_results", {}) or {}
    hits = int(breakdown.get("hits", 0) or 0)
    total_words = int(breakdown.get("total", 0) or 0)
    is_correct = total_words > 0 and hits == total_words
    total_score = session.get("score", 0)

    return {
        "pte_score": result.pte_score,
        "is_async": result.is_async,
        "breakdown": breakdown,
        "totalScore": total_score,
        # snake_case
        "transcript": transcript,
        "user_text": user_text,
        "word_results": word_results,
        "hits": hits,
        "total_words": total_words,
        "is_correct": is_correct,
        "score_for_question": result.pte_score,
        # camelCase aliases for mobile parity
        "wordResults": word_results,
        "totalWords": total_words,
        "isCorrect": is_correct,
        "scoreForQuestion": result.pte_score,
    }


@router.get("/audio-url")
def audio_url(
    s3_url: str,
    current_user: User = Depends(get_current_user),
):
    return {"presigned_url": generate_presigned_url(s3_url)}

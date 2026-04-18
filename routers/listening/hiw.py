from fastapi import APIRouter, Depends, Body, HTTPException
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import User
from core.dependencies import get_current_user
from services.session_service import start_session, get_session, mark_submitted
from services.scoring import get_scorer
from services.s3_service import generate_presigned_url

router = APIRouter(prefix="/listening/hiw", tags=["Listening - Highlight Incorrect Words"])


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
        question_type="highlight_incorrect_words",
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
    session = get_session(session_id)
    question = session["questions"].get(question_id)
    if not question or not question.evaluation:
        raise HTTPException(status_code=404, detail="Question not found")

    # Accept word strings or convert from 0-based word indices
    highlighted_words = payload.get("highlighted_words")
    if highlighted_words is None:
        indices = payload.get("highlighted_indices", [])
        content = question.content_json or {}
        words = content.get("words") or content.get("transcript", "").split()
        highlighted_words = [words[i] for i in indices if isinstance(i, int) and i < len(words)]

    scorer = get_scorer("listening_hiw")
    result = scorer.score(
        question_id=question_id,
        session_id=session_id,
        answer={
            "highlighted_words": highlighted_words,
            "evaluation_json": question.evaluation.evaluation_json,
        },
    )
    mark_submitted(session_id, question_id, result.pte_score)

    eval_json = question.evaluation.evaluation_json or {}
    correct_answers = eval_json.get("correctAnswers", {}) or {}
    incorrect_words = list(correct_answers.get("incorrectWords", []) or [])

    breakdown = result.breakdown or {}
    correct_clicks = list(breakdown.get("correct_clicks", []) or [])
    incorrect_clicks = list(breakdown.get("incorrect_clicks", []) or [])
    missed_words = list(breakdown.get("missed_words", []) or [])
    is_correct = len(incorrect_clicks) == 0 and len(missed_words) == 0
    total_score = session.get("score", 0)

    return {
        "pte_score": result.pte_score,
        "is_async": result.is_async,
        "breakdown": breakdown,
        "totalScore": total_score,
        # snake_case
        "incorrect_words": incorrect_words,
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

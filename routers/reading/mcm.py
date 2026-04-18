from fastapi import APIRouter, Depends, Body, HTTPException
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import User
from core.dependencies import get_current_user
from services.session_service import start_session, get_session, mark_submitted
from services.scoring import get_scorer

router = APIRouter(prefix="/reading/mcm", tags=["Reading - MCM"])


@router.post("/start")
def start(
    payload: dict = Body(default={}),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return start_session(
        db=db,
        user_id=current_user.id,
        module="reading",
        question_type="mcq_multiple",
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
    selected_options = payload.get("selected_options") or payload.get("selected_option_ids", [])

    session = get_session(session_id)
    question = session["questions"].get(question_id)
    if not question or not question.evaluation:
        raise HTTPException(status_code=404, detail="Question not found")

    scorer = get_scorer("reading_mcm")
    result = scorer.score(
        question_id=question_id,
        session_id=session_id,
        answer={
            "selected_options": selected_options,
            "evaluation_json": question.evaluation.evaluation_json,
        },
    )
    mark_submitted(session_id, question_id, result.pte_score)

    eval_json = question.evaluation.evaluation_json or {}
    correct_answers = eval_json.get("correctAnswers", {}) or {}
    correct_option_ids = list(correct_answers.get("correctOptions", []) or [])
    selected_option_ids = list(selected_options or [])
    is_correct = set(selected_option_ids) == set(correct_option_ids)
    total_score = session.get("score", 0)

    return {
        "pte_score": result.pte_score,
        "is_async": result.is_async,
        "breakdown": result.breakdown,
        "totalScore": total_score,
        # snake_case
        "correct_option_ids": correct_option_ids,
        "selected_option_ids": selected_option_ids,
        "is_correct": is_correct,
        "score_for_question": result.pte_score,
        # camelCase aliases for mobile parity
        "correctOptions": correct_option_ids,
        "selectedOptions": selected_option_ids,
        "isCorrect": is_correct,
        "scoreForQuestion": result.pte_score,
    }

from fastapi import APIRouter, Depends, Body, HTTPException
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import User
from core.dependencies import get_current_user
from services.session_service import start_session, get_session, mark_submitted, persist_answer_to_db
from services.scoring import get_scorer

router = APIRouter(prefix="/reading/mcs", tags=["Reading - MCS"])


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
        question_type="mcq_single",
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
    ids = payload.get("selected_option_ids", [])
    selected_option = payload.get("selected_option") or (ids[0] if ids else "")

    session = get_session(session_id)
    question = session["questions"].get(question_id)
    if not question or not question.evaluation:
        raise HTTPException(status_code=404, detail="Question not found")

    scorer = get_scorer("reading_mcs")
    result = scorer.score(
        question_id=question_id,
        session_id=session_id,
        answer={
            "selected_option": selected_option,
            "evaluation_json": question.evaluation.evaluation_json,
        },
    )
    mark_submitted(session_id, question_id, result.pte_score)

    breakdown = result.breakdown or {}
    correct_option = breakdown.get("correct_option")
    persist_answer_to_db(
        session=session, question_id=question_id, question_type="reading_mcs",
        user_answer_json={"selected_option": selected_option},
        correct_answer_json={"correct_option": correct_option},
        result_json=breakdown, score=result.pte_score,
    )
    is_correct = bool(breakdown.get("is_correct", False))
    correct_option_ids = [correct_option] if correct_option is not None else []
    total_score = session.get("score", 0)

    return {
        "pte_score": result.pte_score,
        "is_async": result.is_async,
        "breakdown": breakdown,
        "totalScore": total_score,
        # snake_case
        "correct_option_ids": correct_option_ids,
        "correct_option": correct_option,
        "selected_option": selected_option,
        "is_correct": is_correct,
        "score_for_question": result.pte_score,
        # camelCase aliases for mobile parity
        "correctOption": correct_option,
        "selectedOption": selected_option,
        "isCorrect": is_correct,
        "scoreForQuestion": result.pte_score,
    }

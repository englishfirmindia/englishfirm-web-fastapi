from fastapi import APIRouter, Depends, Body, HTTPException
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import User
from core.dependencies import get_current_user
from services.session_service import start_session, get_session, mark_submitted
from services.scoring import get_scorer

router = APIRouter(prefix="/reading/reorder-paragraphs", tags=["Reading - Reorder Paragraphs"])


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
        question_type="reorder_paragraphs",
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
    user_sequence = payload["user_sequence"]

    session = get_session(session_id)
    question = session["questions"].get(question_id)
    if not question or not question.evaluation:
        raise HTTPException(status_code=404, detail="Question not found")

    scorer = get_scorer("reorder_paragraphs")
    result = scorer.score(
        question_id=question_id,
        session_id=session_id,
        answer={
            "user_sequence": user_sequence,
            "evaluation_json": question.evaluation.evaluation_json,
        },
    )
    mark_submitted(session_id, question_id, result.pte_score)

    return {
        "pte_score": result.pte_score,
        "is_async": result.is_async,
        "breakdown": result.breakdown,
        "totalScore": session.get("score", 0),
    }

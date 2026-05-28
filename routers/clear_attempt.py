"""Clear-attempt endpoint — soft-clears a user's prior speaking practice answer.

Sits next to the existing Redo flow (which overwrites the row in place).
Clear is a separate, opt-in action:
  * Frontend Clear button → POST /api/v1/questions/clear-attempt
  * Sets scoring_status="cleared" on the latest AttemptAnswer for this user
    + question, blanks the scoring fields, and deletes the matching
    UserQuestionAttempt row so the "Done" badge resets.

Doesn't touch Redo. Doesn't delete the row (audit trail kept). Doesn't
delete the S3 audio (lifecycle policy handles aging).
"""
import logging
from fastapi import APIRouter, Depends, Body, HTTPException
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import User
from core.dependencies import get_current_user
from services.session_service import clear_attempt_in_db

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Questions"])


@router.post("/clear-attempt")
def clear_attempt(
    payload: dict = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    qid_raw = payload.get("question_id")
    if qid_raw is None:
        raise HTTPException(status_code=400, detail="question_id required")
    try:
        question_id = int(qid_raw)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="question_id must be int")

    cleared = clear_attempt_in_db(
        user_id=current_user.id,
        question_id=question_id,
    )
    if not cleared:
        # Match get_last_answer semantics — silently 404 rather than leak
        # whether some other user has this question. Frontend treats 404
        # as "nothing to clear, already fresh" and surfaces a snackbar.
        raise HTTPException(status_code=404, detail="no attempt to clear")
    return {"cleared": True}

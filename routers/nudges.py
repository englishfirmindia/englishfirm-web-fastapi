"""In-app Coach nudges — funnel log endpoint.

Frontend fires a `POST /api/v1/nudges/event` at every step of a nudge
funnel (shown → clicked → booked). One row per event goes into
`nudge_events`. Weekly / monthly audits query this table to compute
conversion rate per `nudge_id` (e.g. 'gads_first_visit',
'post_submit_ra', 'post_submit_rs', 'post_submit_fib_dd').

Fire-and-forget from the client: errors are swallowed on the frontend
so a network blip during logging never breaks a user's flow.
"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import NudgeEvent, User
from core.dependencies import get_current_user

router = APIRouter(prefix="/nudges", tags=["Nudges"])


class NudgeEventIn(BaseModel):
    nudge_id: str = Field(..., max_length=64)
    event_type: str = Field(..., max_length=32)
    meta: Optional[dict] = None


@router.post("/event", status_code=204)
def log_event(
    req: NudgeEventIn,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Whitelist — reject unknown event types so the funnel stays clean.
    ALLOWED = {
        # bubble greeting
        "shown", "yes", "no", "dismissed",
        "chat_started", "maximised", "jumped_full",
        # post-submit feedback modal
        "feedback_shown", "feedback_dismissed", "feedback_clicked_book",
        "booking_started", "booking_confirmed", "booking_cancelled",
    }
    if req.event_type not in ALLOWED:
        raise HTTPException(status_code=400,
                            detail=f"unknown event_type '{req.event_type}'")
    db.add(NudgeEvent(
        user_id=current_user.id,
        nudge_id=req.nudge_id[:64],
        event_type=req.event_type[:32],
        meta=req.meta,
    ))
    db.commit()

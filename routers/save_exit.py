"""Telemetry ping for the "Save & Exit" action on sectional + mock exam
screens.

The exit flow itself is purely client-side (cancels timers, writes a
SharedPreferences breadcrumb, navigates back to the test list). No
server-side state changes — each in-flight answer was already persisted
by its own `POST /submit`. This endpoint exists only so we can SEE in
CloudWatch that the user clicked Save & Exit, with the session id and
test number, when diagnosing "the button did nothing" reports.

  POST /api/v1/save-exit
    body: {module: "reading"|"writing"|"speaking"|"listening"|"mock",
           session_id: str,
           test_number?: int,
           timer_remaining?: int,
           current_part?: int}
    → 200 {"ok": true}

Never raises. Never writes state. Fire-and-forget from the client.
"""
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import User
from core.dependencies import get_current_user
from core.logging_config import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/save-exit", tags=["Save & Exit telemetry"])


class SaveExitPing(BaseModel):
    module: str
    session_id: str
    test_number: Optional[int] = None
    timer_remaining: Optional[int] = None
    current_part: Optional[int] = None


@router.post("")
def save_exit_ping(
    req: SaveExitPing,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Log the Save & Exit event. Pure telemetry — does not mutate any
    state. The session_states / practice_attempts rows have already been
    persisted by the individual submits."""
    log.info(
        "[SAVE_EXIT] user=%d module=%s session=%s test_number=%s "
        "timer_remaining=%s current_part=%s",
        current_user.id,
        req.module,
        req.session_id,
        req.test_number,
        req.timer_remaining,
        req.current_part,
    )
    return {"ok": True}

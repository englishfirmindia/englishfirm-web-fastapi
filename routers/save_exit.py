"""Lightweight UI-click telemetry.

Logs button-press / nav events to CloudWatch so "the button did nothing"
reports become diagnosable in seconds. Pure logging — no state mutation,
no DB writes. Fire-and-forget from the client.

Two endpoints — historically `/save-exit` was Save & Exit only, but the
client now uses the more general `/ui-click` for ALL telemetry events
(Save & Exit clicks, logo clicks, etc.). The original `/save-exit` is
kept alive so older deployed clients still work.

  POST /api/v1/ui-click
    body: {button: str,            # e.g. "save_exit", "ef_logo"
           outcome?: str,          # e.g. "clicked" | "confirmed" | "cancelled" | "error"
           extras?: dict}          # free-form context (route, session_id, module, etc.)
    → 200 {"ok": true}

  POST /api/v1/save-exit  (legacy alias)
    body: {module, session_id, test_number?, timer_remaining?, current_part?}
    → 200 {"ok": true}
"""
from typing import Optional, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import User
from core.dependencies import get_current_user
from core.logging_config import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["UI click telemetry"])


class UiClickPing(BaseModel):
    button: str
    outcome: Optional[str] = "clicked"
    extras: Optional[dict[str, Any]] = None


@router.post("/ui-click")
def ui_click_ping(
    req: UiClickPing,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Log any UI click. Filter on `[BTN_CLICK]` in CloudWatch to see
    just these. `button` is a stable slug; `outcome` distinguishes
    multi-stage flows (click → confirmed/cancelled); `extras` carries
    any contextual fields the call site wants to surface."""
    extras_str = ""
    if req.extras:
        # Render extras as space-separated key=value pairs so CloudWatch
        # filter patterns can match individual keys cheaply.
        extras_str = " " + " ".join(
            f"{k}={v}" for k, v in req.extras.items() if v is not None
        )
    log.info(
        "[BTN_CLICK] user=%d button=%s outcome=%s%s",
        current_user.id, req.button, req.outcome or "clicked", extras_str,
    )
    return {"ok": True}


# ─────────────────────────────────────────────────────────────────────────────
# Legacy /save-exit endpoint — kept alive so any in-flight client builds that
# still POST here don't 404. The new client uses /ui-click instead.
# ─────────────────────────────────────────────────────────────────────────────

class SaveExitPing(BaseModel):
    module: str
    session_id: str
    test_number: Optional[int] = None
    timer_remaining: Optional[int] = None
    current_part: Optional[int] = None


@router.post("/save-exit")
def save_exit_ping(
    req: SaveExitPing,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    log.info(
        "[BTN_CLICK] user=%d button=save_exit outcome=legacy module=%s "
        "session=%s test_number=%s timer_remaining=%s current_part=%s",
        current_user.id,
        req.module,
        req.session_id,
        req.test_number,
        req.timer_remaining,
        req.current_part,
    )
    return {"ok": True}

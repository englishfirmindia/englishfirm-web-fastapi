"""Frontend observability — accepts batched telemetry events from the
Flutter web client.

Pure logging endpoint: each event is rendered as a single CloudWatch log
line prefixed `[FE_TELEMETRY]` so client-side errors, route-404s, recorder
hangs and network failures show up in the same log stream as the backend
logs and can be cross-correlated by `user_id` and `client_session`.

Event shape (one batch can carry many events to amortize the round-trip):

  POST /api/v1/telemetry/event
    {
      "client_session": "uuid",       # stable per browser tab
      "events": [
        {
          "type":  "error|network|route|recorder|audio|perf",
          "ts":    "ISO8601",
          "route": "/practice/describe-image",
          "ua":    "Mozilla/5.0 …",
          "data":  { … free-form, type-specific }
        },
        …
      ]
    }
    → 200 {"ok": true, "received": <count>}

No DB writes, no enrichment. Frontend errors must never be allowed to
fail loudly — the endpoint always returns 200 unless the payload is so
malformed pydantic itself rejects it.
"""
from typing import Optional, Any, List

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from db.models import User
from core.dependencies import get_current_user
from core.logging_config import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["Frontend telemetry"])


class TelemetryEvent(BaseModel):
    type: str = Field(..., max_length=32)
    ts: Optional[str] = Field(None, max_length=40)
    route: Optional[str] = Field(None, max_length=256)
    ua: Optional[str] = Field(None, max_length=512)
    data: Optional[dict[str, Any]] = None


class TelemetryBatch(BaseModel):
    client_session: Optional[str] = Field(None, max_length=64)
    events: List[TelemetryEvent] = Field(default_factory=list, max_length=50)


@router.post("/telemetry/event")
def telemetry_event(
    req: TelemetryBatch,
    current_user: User = Depends(get_current_user),
):
    """Log a batch of frontend telemetry events. Each event becomes one
    CloudWatch line so the `[FE_TELEMETRY]` filter pattern surfaces them
    all. Single inserts are fine; the batching is purely to amortize
    network from the client.
    """
    sess = req.client_session or "-"
    for ev in req.events:
        # Render `data` as space-separated key=value pairs so CloudWatch
        # filter patterns can match individual keys cheaply, matching
        # the pattern used by /ui-click.
        data_str = ""
        if ev.data:
            data_str = " " + " ".join(
                f"{k}={v}" for k, v in ev.data.items() if v is not None
            )
        log.info(
            "[FE_TELEMETRY] user=%d session=%s type=%s route=%s ua=%s ts=%s%s",
            current_user.id,
            sess,
            ev.type,
            ev.route or "-",
            (ev.ua or "-")[:120],
            ev.ts or "-",
            data_str,
        )
    return {"ok": True, "received": len(req.events)}

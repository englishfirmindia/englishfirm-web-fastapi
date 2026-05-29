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

Auth contract — DELIBERATELY SOFT
---------------------------------
The endpoint is the only route in the API that uses `try_get_user`
(returns Optional[User]) instead of `get_current_user`. Telemetry exists
to capture failures including:
  * Pre-login errors (boot crashes, login-page JS errors)
  * Expired-token edge cases (the token-refresh itself failing)
  * `navigator.sendBeacon` calls on tab close (sendBeacon cannot carry
    the Authorization header)
All three of those can't satisfy a hard-auth requirement. Before this
soft-auth shipped, 100% of telemetry POSTs were 401'd at the door and
nothing reached CloudWatch (the entire point of the system). Now
anonymous events log with `user=0`; authenticated events get the real
user_id attached for correlation.

Per-IP rate limit
-----------------
A naive in-memory token-bucket per remote IP caps the endpoint at
`_RATE_LIMIT_PER_MINUTE` POSTs per minute to keep an opportunistic
abuser from inflating CloudWatch ingest. Single-process state is fine:
ECS runs ~1 task today, and a brief over-limit on rollover is harmless
(telemetry just drops the batch, same as it does on any failure).

No DB writes, no enrichment. Frontend errors must never be allowed to
fail loudly — the endpoint returns 200 unless the payload is so
malformed pydantic itself rejects it, or the rate limit is breached
(429).
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Optional, Any, List

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from db.models import User
from core.dependencies import try_get_user
from core.logging_config import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["Frontend telemetry"])


# ── Rate limit: per-IP in-memory token bucket ─────────────────────────────
_RATE_LIMIT_PER_MINUTE = 60
_RATE_WINDOW_SECONDS = 60
_rate_buckets: dict[str, deque[float]] = defaultdict(deque)


def _check_rate_limit(ip: str) -> bool:
    """Return True if the request is within the per-IP rate budget.
    Slides a 60s window of timestamps; pops anything older than the
    window before checking the count."""
    now = time.monotonic()
    bucket = _rate_buckets[ip]
    while bucket and bucket[0] < now - _RATE_WINDOW_SECONDS:
        bucket.popleft()
    if len(bucket) >= _RATE_LIMIT_PER_MINUTE:
        return False
    bucket.append(now)
    return True


def _client_ip(request: Request) -> str:
    """Resolve the client's IP through ALB's X-Forwarded-For. Falls back
    to the direct peer when the header is missing (local dev / tests)."""
    xff = request.headers.get("x-forwarded-for")
    if xff:
        # XFF is a comma-separated list; the leftmost entry is the
        # original client.
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "?"


def _sanitize_for_log(value: Any) -> str:
    """Render a value for inclusion in a single CloudWatch log line.
    Strips newlines / tabs so an attacker can't inject fake-looking
    log lines via the `data` field. Caps length defensively."""
    s = str(value)
    # Collapse all whitespace control chars to a single space.
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return s[:200]


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
    request: Request,
    current_user: Optional[User] = Depends(try_get_user),
):
    """Log a batch of frontend telemetry events. Each event becomes one
    CloudWatch line so the `[FE_TELEMETRY]` filter pattern surfaces them
    all. Single inserts are fine; the batching is purely to amortize
    network from the client.

    Soft auth: anonymous events are accepted (user=0). See module docstring
    for why hard auth would defeat the purpose. Per-IP rate-limited.
    """
    ip = _client_ip(request)
    if not _check_rate_limit(ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"telemetry rate limit ({_RATE_LIMIT_PER_MINUTE}/min/ip)",
        )

    user_id = current_user.id if current_user else 0
    sess = _sanitize_for_log(req.client_session or "-")
    for ev in req.events:
        # Render `data` as space-separated key=value pairs so CloudWatch
        # filter patterns can match individual keys cheaply, matching
        # the pattern used by /ui-click. All values pass through
        # _sanitize_for_log to keep an attacker from injecting newlines
        # that would forge fake log lines.
        data_str = ""
        if ev.data:
            data_str = " " + " ".join(
                f"{_sanitize_for_log(k)}={_sanitize_for_log(v)}"
                for k, v in ev.data.items() if v is not None
            )
        log.info(
            "[FE_TELEMETRY] user=%d session=%s type=%s route=%s ua=%s ts=%s%s",
            user_id,
            sess,
            _sanitize_for_log(ev.type),
            _sanitize_for_log(ev.route or "-"),
            _sanitize_for_log(ev.ua or "-"),
            _sanitize_for_log(ev.ts or "-"),
            data_str,
        )
    return {"ok": True, "received": len(req.events)}

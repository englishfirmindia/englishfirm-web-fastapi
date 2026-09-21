"""
Zapier webhook integration for EnglishFirm web.

Two independent webhooks:
  - send_signup_webhook       → ZAPIER_WEBHOOK_URL       (signup CRM push)
  - send_demo_booking         → ZAPIER_DEMO_BOOKING_URL  (coach-bubble demo)

Both are fire-and-forget with retry/timeout; failures never crash the caller.
"""

import os
import time
import requests
from typing import Optional
from datetime import date, datetime, timedelta, timezone

from core.logging_config import get_logger

log = get_logger(__name__)


ZAPIER_WEBHOOK_URL       = os.getenv("ZAPIER_WEBHOOK_URL")
ZAPIER_DEMO_BOOKING_URL  = os.getenv("ZAPIER_DEMO_BOOKING_URL", "")


def send_signup_webhook(
    student_name: str,
    phone_number: Optional[str],
    exam_date: Optional[date],
) -> None:
    """
    Fire-and-forget: POST signup data to Zapier.
    Silently skips if ZAPIER_WEBHOOK_URL is not configured.
    Never raises — must not crash the signup flow.
    """
    if not ZAPIER_WEBHOOK_URL:
        return

    payload = {
        "student_name": student_name,
        "phone_number": phone_number,
        "exam_date": exam_date.isoformat() if exam_date else None,
    }

    try:
        response = requests.post(
            ZAPIER_WEBHOOK_URL,
            json=payload,
            timeout=5,
        )
        response.raise_for_status()
    except requests.RequestException as e:
        # Do NOT crash signup if webhook fails
        log.error(f"[Zapier webhook error] {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Demo-booking webhook — ported 2026-09-21 from englishfirm-app-fastapi.
# Called from the schedule_demo tool when Claude on the coach bubble
# collects preferred day+time and validates the user isn't already booked.
# ─────────────────────────────────────────────────────────────────────────────

# Australia/Sydney offset. Standard time (AEST) is UTC+10 (Apr-Oct);
# DST (AEDT) is UTC+11 (Oct-Apr). Zapier receives both offset variants
# so downstream steps can pick whichever matches Google Calendar's expectation.
def _sydney_offset_for(dt: datetime) -> timedelta:
    """Return the Sydney offset for a given date. Rough DST cutover for AU."""
    # DST: first Sunday of October → first Sunday of April
    y = dt.year
    oct_1 = datetime(y, 10, 1)
    oct_dst_start = oct_1 + timedelta(days=(6 - oct_1.weekday()) % 7)
    apr_1 = datetime(y, 4, 1)
    apr_dst_end = apr_1 + timedelta(days=(6 - apr_1.weekday()) % 7)
    naive = dt.replace(tzinfo=None) if dt.tzinfo else dt
    if apr_dst_end <= naive < oct_dst_start:
        return timedelta(hours=10)   # AEST
    return timedelta(hours=11)       # AEDT


def _build_gcal_datetime_fields(scheduled_date: str, scheduled_time: str,
                                 duration_minutes: int = 30) -> dict:
    """Parse 'YYYY-MM-DD' + 'HH:MM' as Sydney local time, expand to GCal fields.

    Returns empty dict on parse failure so the tool still sends the human
    preferences even if the LLM produced a malformed date.
    """
    try:
        start_naive = datetime.strptime(f"{scheduled_date} {scheduled_time}",
                                        "%Y-%m-%d %H:%M")
    except ValueError:
        log.warning("[ZAPIER_DEMO] bad scheduled_date/time: %r %r",
                    scheduled_date, scheduled_time)
        return {}

    offset = _sydney_offset_for(start_naive)
    tz = timezone(offset)
    start = start_naive.replace(tzinfo=tz)
    end   = start + timedelta(minutes=duration_minutes)
    offset_str = f"{'+' if offset >= timedelta(0) else '-'}{abs(offset).seconds//3600:02d}:00"

    return {
        "scheduled_start_datetime": start.strftime("%Y-%m-%dT%H:%M:%S"),
        "scheduled_end_datetime":   end.strftime("%Y-%m-%dT%H:%M:%S"),
        "scheduled_start_iso_tz":   start.isoformat(),
        "scheduled_end_iso_tz":     end.isoformat(),
        "scheduled_start_utc":      start.astimezone(timezone.utc).isoformat(),
        "scheduled_end_utc":        end.astimezone(timezone.utc).isoformat(),
        "scheduled_date":           start.strftime("%Y-%m-%d"),
        "scheduled_time":           start.strftime("%H:%M"),
        "scheduled_weekday":        start.strftime("%A"),
        "scheduled_pretty":         start.strftime("%A %d %B %Y at %I:%M %p"),
        "timezone":                 "Australia/Sydney",
        "timezone_offset":          offset_str,
        "duration_minutes":         duration_minutes,
    }


def send_demo_booking(
    *,
    user_id: int,
    username: str,
    email: Optional[str],
    target_score: Optional[int],
    preferred_day: str,
    preferred_time: str,
    phone: str,
    notes: str = "",
    scheduled_date: str = "",
    scheduled_time: str = "",
    duration_minutes: int = 30,
    timeout: float = 2.5,
    max_retries: int = 3,
) -> bool:
    """POST demo booking to Zapier. Returns True on success (2xx).

    scheduled_date + scheduled_time (Australia/Sydney local) are used to
    build Google-Calendar-ready fields (scheduled_start_datetime etc).
    Falls back gracefully if either is missing / unparseable.

    Never raises — the coach-bubble tool must degrade gracefully on Zapier
    failure (returns False, tool returns an apology string to the LLM).
    """
    if not ZAPIER_DEMO_BOOKING_URL:
        log.warning("[ZAPIER_DEMO] ZAPIER_DEMO_BOOKING_URL not configured — booking not sent")
        return False

    payload = {
        # Standard identifiers
        "user_id":         user_id,
        "username":        username,
        "user_email":      email,
        "target_score":    target_score,
        "phone":           phone,
        "notes":           notes,

        # Human-readable preferences (great for email bodies)
        "preferred_day":   preferred_day,
        "preferred_time":  preferred_time,

        # Meta — lets Nimisha's Zap tell "web coach bubble" from mobile chat.
        "source":          "web_coach",
        "timestamp":       datetime.now(timezone.utc).isoformat(),
    }

    payload.update(_build_gcal_datetime_fields(scheduled_date, scheduled_time,
                                               duration_minutes))

    for attempt in range(max_retries):
        try:
            r = requests.post(ZAPIER_DEMO_BOOKING_URL, json=payload, timeout=timeout)
            if 200 <= r.status_code < 300:
                log.info("[ZAPIER_DEMO] booked user_id=%s start=%s (Sydney)",
                         user_id, payload.get("scheduled_start_datetime") or "unset")
                return True
            log.warning("[ZAPIER_DEMO] attempt %d bad status: %s", attempt + 1, r.status_code)
        except requests.RequestException as e:
            log.warning("[ZAPIER_DEMO] attempt %d failed: %s", attempt + 1, e)
        time.sleep(0.5 * (attempt + 1))
    return False

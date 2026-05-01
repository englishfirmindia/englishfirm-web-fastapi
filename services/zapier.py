"""
Zapier webhook integration for EnglishFirm web.

Sends signup data to the configured ZAPIER_WEBHOOK_URL (non-blocking).
If the env var is not set the call is silently skipped.
"""

import os
import requests
from typing import Optional
from datetime import date

from core.logging_config import get_logger

log = get_logger(__name__)


ZAPIER_WEBHOOK_URL = os.getenv("ZAPIER_WEBHOOK_URL")


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

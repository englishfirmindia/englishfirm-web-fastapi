"""
Lightweight email-sending shim.

Two delivery paths, both fire-and-forget:
  1. Always print the message to stdout (so OTPs are visible in the
     backend log during development even when no provider is wired up).
  2. If EMAIL_WEBHOOK_URL is set, POST a JSON envelope to that URL —
     intended for a Zapier/Make.com webhook that relays to Gmail/SES.

Swap `_post_webhook` with a real provider (boto3 SES / SendGrid SDK)
when the project gets one.
"""

from typing import Optional

import requests

import core.config as config

from core.logging_config import get_logger

log = get_logger(__name__)


def _post_webhook(to: str, subject: str, body: str) -> None:
    if not config.EMAIL_WEBHOOK_URL:
        return
    try:
        requests.post(
            config.EMAIL_WEBHOOK_URL,
            json={
                "to": to,
                "from": config.EMAIL_FROM,
                "subject": subject,
                "body": body,
            },
            timeout=5,
        )
    except Exception as exc:
        log.warning("[email] webhook delivery failed: %s", exc)


def send_email(to: str, subject: str, body: str) -> None:
    """Fire-and-forget email send. Never raises."""
    log.info(f"\n[EMAIL] to={to}\n" f" from={config.EMAIL_FROM}\n" f" subject={subject}\n" f" body=\n{body}\n")
    _post_webhook(to, subject, body)


def send_password_reset(to: str, link: str, expires_in_minutes: int) -> None:
    subject = "Reset your EnglishFirm password"
    body = (
        f"We received a request to reset your password.\n\n"
        f"Open this link to set a new one (expires in {expires_in_minutes} minutes):\n"
        f"{link}\n\n"
        f"If you did not request this, you can ignore this email — your password "
        f"will stay the same.\n"
    )
    send_email(to, subject, body)


def send_trainer_otp(to: str, code: str, expires_in_minutes: int) -> None:
    subject = "Your EnglishFirm trainer sign-in code"
    body = (
        f"Your sign-in code is: {code}\n\n"
        f"It expires in {expires_in_minutes} minutes. "
        f"If you did not request this, you can ignore this email."
    )
    send_email(to, subject, body)


def send_trainer_share_received(
    to: str, student_name: str, test_label: str, share_id: int
) -> None:
    subject = f"{student_name} shared a {test_label} test with you"
    next_path = f"/trainer/shared/{share_id}"
    link = f"{config.FRONTEND_URL.rstrip('/')}/trainer/login?next={next_path}"
    body = (
        f"{student_name} just shared a {test_label} test with you on EnglishFirm.\n\n"
        f"Open the test: {link}\n\n"
        f"You'll be asked to enter your email to receive a one-time sign-in code.\n"
    )
    send_email(to, subject, body)


def send_student_note_posted(
    to: str,
    trainer_name: str,
    test_label: str,
    student_name: Optional[str] = None,
) -> None:
    greeting = f"Hi {student_name}," if student_name else "Hi,"
    subject = f"{trainer_name} left a note on your {test_label} test"
    body = (
        f"{greeting}\n\n"
        f"{trainer_name} left a note on your {test_label} test. "
        f"Open the Feedback tab to read it.\n"
    )
    send_email(to, subject, body)


def send_trainer_share_revoked(to: str, student_name: str, test_label: str) -> None:
    subject = f"{student_name} revoked access to a shared test"
    body = (
        f"{student_name} revoked your access to a previously shared "
        f"{test_label} test.\n"
    )
    send_email(to, subject, body)

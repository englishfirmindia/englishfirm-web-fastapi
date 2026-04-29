"""
Trainer authentication endpoints.

POST /api/v1/trainer/auth/request-otp  → always 200 (no enumeration)
POST /api/v1/trainer/auth/verify-otp   → JWT on success
GET  /api/v1/trainer/auth/me           → current trainer profile
POST /api/v1/trainer/auth/logout       → 204 (client discards token)
"""

import random
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException, Request
from pydantic import BaseModel, field_validator
from sqlalchemy import func
from sqlalchemy.orm import Session

import core.config as config
from db.database import get_db
from db.models import Trainer, TrainerOtp
from services.email import send_trainer_otp
from services.trainer_auth import get_current_trainer, make_trainer_token


router = APIRouter(prefix="/trainer/auth", tags=["Trainer - Auth"])


# ── Request models ────────────────────────────────────────────────────────────

def _validate_email_shape(value: str) -> str:
    """Lightweight format check — full validation lives in the trainers
    table (whitelist), so we just need to reject obviously broken input."""
    s = (value or "").strip()
    if "@" not in s or "." not in s.split("@")[-1] or len(s) > 254:
        raise ValueError("invalid email format")
    return s


class RequestOtpBody(BaseModel):
    email: str

    @field_validator("email")
    @classmethod
    def _check_email(cls, v: str) -> str:
        return _validate_email_shape(v)


class VerifyOtpBody(BaseModel):
    email: str
    code: str

    @field_validator("email")
    @classmethod
    def _check_email(cls, v: str) -> str:
        return _validate_email_shape(v)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize_email(raw: str) -> str:
    return raw.strip().lower()


def _generate_code() -> str:
    """Random N-digit numeric string with leading zeros preserved."""
    upper = 10 ** config.TRAINER_OTP_LENGTH
    return str(random.randint(0, upper - 1)).zfill(config.TRAINER_OTP_LENGTH)


def _client_ip(request: Request) -> Optional[str]:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/request-otp")
def request_otp(
    body: RequestOtpBody,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Always returns 200 to avoid email enumeration.
    Behind the scenes: if the email is whitelisted and active, generate a
    fresh 6-digit code, invalidate any earlier unconsumed codes for the
    same email, and email the new code.
    """
    email = _normalize_email(body.email)
    generic_response = {
        "message": "If your email is registered as a trainer, a code is on its way.",
        "expires_in_minutes": config.TRAINER_OTP_EXPIRY_MINUTES,
    }

    trainer = (
        db.query(Trainer)
        .filter(func.lower(Trainer.email) == email, Trainer.is_active.is_(True))
        .first()
    )
    if not trainer:
        return generic_response

    # Rate limit: max N codes per hour per email.
    one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
    recent_count = (
        db.query(TrainerOtp)
        .filter(
            func.lower(TrainerOtp.email) == email,
            TrainerOtp.created_at >= one_hour_ago,
        )
        .count()
    )
    if recent_count >= config.TRAINER_OTP_RATE_LIMIT_PER_HOUR:
        # Silent — no leak to the caller.
        return generic_response

    # Invalidate earlier unconsumed codes for this email.
    now = datetime.now(timezone.utc)
    db.query(TrainerOtp).filter(
        func.lower(TrainerOtp.email) == email,
        TrainerOtp.consumed_at.is_(None),
    ).update({TrainerOtp.consumed_at: now}, synchronize_session=False)

    code = _generate_code()
    otp = TrainerOtp(
        email=email,
        code=code,
        expires_at=now + timedelta(minutes=config.TRAINER_OTP_EXPIRY_MINUTES),
        attempts_left=config.TRAINER_OTP_MAX_ATTEMPTS,
        ip=_client_ip(request),
        user_agent=request.headers.get("user-agent"),
    )
    db.add(otp)
    db.commit()

    # Send out-of-band — backend log always shows it; webhook delivers if configured.
    background_tasks.add_task(
        send_trainer_otp,
        to=email,
        code=code,
        expires_in_minutes=config.TRAINER_OTP_EXPIRY_MINUTES,
    )

    return generic_response


@router.post("/verify-otp")
def verify_otp(
    body: VerifyOtpBody,
    db: Session = Depends(get_db),
):
    """
    Validate the OTP and mint a trainer JWT on success.
    Decrements `attempts_left` on bad code; invalidates the code when it
    hits zero.
    """
    email = _normalize_email(body.email)
    code = body.code.strip()

    trainer = (
        db.query(Trainer)
        .filter(func.lower(Trainer.email) == email, Trainer.is_active.is_(True))
        .first()
    )
    if not trainer:
        # Same opaque error as a wrong code to stop enumeration.
        raise HTTPException(status_code=400, detail="Invalid or expired code")

    now = datetime.now(timezone.utc)
    otp = (
        db.query(TrainerOtp)
        .filter(
            func.lower(TrainerOtp.email) == email,
            TrainerOtp.consumed_at.is_(None),
            TrainerOtp.expires_at > now,
        )
        .order_by(TrainerOtp.created_at.desc())
        .first()
    )
    if not otp:
        raise HTTPException(status_code=400, detail="Invalid or expired code")

    if otp.attempts_left <= 0:
        otp.consumed_at = now
        db.commit()
        raise HTTPException(status_code=400, detail="Code locked. Request a new one.")

    if otp.code != code:
        otp.attempts_left = max(0, otp.attempts_left - 1)
        attempts_left = otp.attempts_left
        if attempts_left == 0:
            otp.consumed_at = now
        db.commit()
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Invalid or expired code",
                "attempts_left": attempts_left,
            },
        )

    # Success — burn the code, mint a token.
    otp.consumed_at = now
    db.commit()

    token = make_trainer_token(trainer)
    return {
        "access_token": token,
        "token_type": "bearer",
        "trainer": {
            "id": trainer.id,
            "email": trainer.email,
            "display_name": trainer.display_name,
        },
        "expires_in_seconds": config.TRAINER_JWT_EXPIRY_HOURS * 3600,
    }


@router.get("/me")
def me(trainer: Trainer = Depends(get_current_trainer)):
    return {
        "id": trainer.id,
        "email": trainer.email,
        "display_name": trainer.display_name,
        "is_active": trainer.is_active,
    }


@router.post("/logout", status_code=204)
def logout(_: Trainer = Depends(get_current_trainer)):
    """
    Stateless JWT — the client discards the token. This endpoint exists
    purely so the client can call it for symmetry / audit logging.
    """
    return None

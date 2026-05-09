from datetime import datetime, timedelta, date
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response, status
from fastapi.security import OAuth2PasswordRequestForm
from passlib.context import CryptContext
from jose import jwt, jwk as jose_jwk, JWTError
from pydantic import BaseModel
from sqlalchemy.orm import Session
import requests as _requests

import core.config as config
from core.dependencies import get_current_user
from core.rate_limit import limiter
from db.database import get_db
from db.models import User
from services.email import send_password_reset
from services.zapier import send_signup_webhook
from services.auth_token_service import (
    issue_access_token,
    issue_token_pair,
    rotate_refresh_token,
    revoke_refresh_token,
)

router = APIRouter(prefix="/auth", tags=["Auth"])
_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")


class SignupRequest(BaseModel):
    username: str
    email: str
    password: str
    phone: Optional[str] = None
    score_requirement: Optional[int] = None
    exam_date: Optional[str] = None


class GoogleAuthRequest(BaseModel):
    id_token: str


class AppleAuthRequest(BaseModel):
    identity_token: str


class ForgotPasswordRequest(BaseModel):
    email: str


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str


def _make_token(user_id: int) -> str:
    """Short-lived (30 min) JWT access token. Pair this with a refresh
       token issued via services.auth_token_service.issue_token_pair."""
    return issue_access_token(user_id)


def _set_session_cookie(response: Response, token: str) -> None:
    """
    Set the httpOnly session cookie alongside the JSON `access_token`.
    Web clients (Flutter web with withCredentials) read this on subsequent
    requests; iOS ignores Set-Cookie and keeps using `Authorization: Bearer`.
    Now scoped to the access-token TTL — the refresh cookie carries the
    durable session.
    """
    response.set_cookie(
        key=config.SESSION_COOKIE_NAME,
        value=token,
        max_age=config.JWT_ACCESS_EXPIRY_MINUTES * 60,
        httponly=True,
        secure=config.SESSION_COOKIE_SECURE,
        samesite=config.SESSION_COOKIE_SAMESITE,
        domain=config.SESSION_COOKIE_DOMAIN,
        path="/",
    )


def _clear_session_cookie(response: Response) -> None:
    response.delete_cookie(
        key=config.SESSION_COOKIE_NAME,
        path="/",
        domain=config.SESSION_COOKIE_DOMAIN,
    )


def _set_refresh_cookie(response: Response, raw_refresh: str) -> None:
    """httpOnly cookie holding the raw refresh token. 30-day max-age.
       Web clients use this; iOS ignores Set-Cookie and uses the JSON
       refresh_token field instead."""
    response.set_cookie(
        key=config.REFRESH_COOKIE_NAME,
        value=raw_refresh,
        max_age=config.REFRESH_COOKIE_MAX_AGE_SECONDS,
        httponly=True,
        secure=config.SESSION_COOKIE_SECURE,
        samesite=config.SESSION_COOKIE_SAMESITE,
        domain=config.SESSION_COOKIE_DOMAIN,
        path="/",
    )


def _clear_refresh_cookie(response: Response) -> None:
    response.delete_cookie(
        key=config.REFRESH_COOKIE_NAME,
        path="/",
        domain=config.SESSION_COOKIE_DOMAIN,
    )


def _issue_pair_and_set_cookies(
    db: Session, user_id: int, request: Request, response: Response,
) -> dict:
    """Helper used by login / google / apple — issue access+refresh pair,
       set both cookies, return the JSON response body. db.commit() must
       be called by caller (or rely on dependency cleanup)."""
    access, refresh, _row = issue_token_pair(db, user_id=user_id, request=request)
    db.commit()
    _set_session_cookie(response, access)
    _set_refresh_cookie(response, refresh)
    return {
        "access_token":         access,
        "refresh_token":        refresh,
        "token_type":           "bearer",
        "access_expires_in":    config.JWT_ACCESS_EXPIRY_MINUTES * 60,
        "refresh_expires_in":   config.REFRESH_COOKIE_MAX_AGE_SECONDS,
    }


def _make_password_reset_token(user_id: int) -> str:
    payload = {
        "sub": str(user_id),
        "purpose": config.PASSWORD_RESET_TOKEN_PURPOSE,
        "exp": datetime.utcnow()
        + timedelta(minutes=config.PASSWORD_RESET_TOKEN_EXPIRY_MINUTES),
    }
    return jwt.encode(payload, config.JWT_SECRET_KEY, algorithm=config.JWT_ALGORITHM)


def _verify_password_reset_token(token: str) -> int:
    """Returns the user_id encoded in the token, or raises HTTPException 400."""
    try:
        claims = jwt.decode(
            token, config.JWT_SECRET_KEY, algorithms=[config.JWT_ALGORITHM]
        )
    except JWTError:
        raise HTTPException(status_code=400, detail="Invalid or expired reset link")
    if claims.get("purpose") != config.PASSWORD_RESET_TOKEN_PURPOSE:
        raise HTTPException(status_code=400, detail="Invalid or expired reset link")
    sub = claims.get("sub")
    try:
        return int(sub)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid or expired reset link")


def _parse_exam_date(exam_date_str: Optional[str]) -> Optional[date]:
    if not exam_date_str:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(exam_date_str, fmt).date()
        except ValueError:
            continue
    return None


@router.post("/signup", status_code=201)
@limiter.limit("5/minute")
def signup(request: Request, req: SignupRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == req.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    parsed_exam_date = _parse_exam_date(req.exam_date)
    user = User(
        username=req.username,
        email=req.email,
        hashed_password=_pwd.hash(req.password),
        phone=req.phone,
        score_requirement=req.score_requirement,
        exam_date=parsed_exam_date,
    )
    db.add(user)
    db.commit()
    background_tasks.add_task(
        send_signup_webhook,
        student_name=req.username,
        phone_number=req.phone,
        exam_date=parsed_exam_date,
    )
    return {"message": "Account created"}


@router.post("/login")
@limiter.limit("10/minute")
def login(
    request: Request,
    response: Response,
    form: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.email == form.username).first()
    if not user or not _pwd.verify(form.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    return _issue_pair_and_set_cookies(db, user.id, request, response)


@router.post("/forgot-password")
@limiter.limit("3/minute")
def forgot_password(
    request: Request,
    req: ForgotPasswordRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Always returns 200 to avoid email enumeration. If the email is registered,
    a short-lived reset link is sent out-of-band.
    """
    generic_response = {
        "message": "If that email is registered, a reset link is on its way.",
        "expires_in_minutes": config.PASSWORD_RESET_TOKEN_EXPIRY_MINUTES,
    }
    email = (req.email or "").strip().lower()
    if "@" not in email:
        return generic_response

    user = db.query(User).filter(User.email == email).first()
    if not user:
        return generic_response

    token = _make_password_reset_token(user.id)
    link = (
        f"{config.FRONTEND_URL.rstrip('/')}"
        f"/#/reset-password?token={token}"
    )
    background_tasks.add_task(
        send_password_reset,
        to=user.email,
        link=link,
        expires_in_minutes=config.PASSWORD_RESET_TOKEN_EXPIRY_MINUTES,
    )
    return generic_response


@router.post("/reset-password")
@limiter.limit("5/minute")
def reset_password(request: Request, req: ResetPasswordRequest, db: Session = Depends(get_db)):
    """Verify a password-reset JWT and overwrite the user's password."""
    if not req.new_password or len(req.new_password) < 6:
        raise HTTPException(
            status_code=400,
            detail="Password must be at least 6 characters",
        )
    user_id = _verify_password_reset_token(req.token)
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=400, detail="Invalid or expired reset link")
    user.hashed_password = _pwd.hash(req.new_password)
    db.commit()
    return {"message": "Password updated. You can now sign in."}


@router.post("/google")
@limiter.limit("10/minute")
def google_auth(
    request: Request,
    response: Response,
    req: GoogleAuthRequest,
    db: Session = Depends(get_db),
):
    """Verify a Google id_token and return a JWT. Creates the user if they don't exist."""
    try:
        resp = _requests.get(
            f"https://oauth2.googleapis.com/tokeninfo?id_token={req.id_token}",
            timeout=10,
        )
        resp.raise_for_status()
        token_info = resp.json()
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Google token verification failed: {exc}")

    if "error" in token_info:
        raise HTTPException(status_code=401, detail=token_info.get("error_description", "Invalid Google token"))

    email = token_info.get("email")
    if not email:
        raise HTTPException(status_code=401, detail="Google token missing email")

    if not token_info.get("email_verified", False):
        raise HTTPException(status_code=401, detail="Google email not verified")

    user = db.query(User).filter(User.email == email).first()
    if not user:
        name = token_info.get("name") or token_info.get("given_name") or email.split("@")[0]
        user = User(
            username=name,
            email=email,
            hashed_password=_pwd.hash(token_info.get("sub", "")),
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    return _issue_pair_and_set_cookies(db, user.id, request, response)


@router.post("/apple")
@limiter.limit("10/minute")
def apple_auth(
    request: Request,
    response: Response,
    req: AppleAuthRequest,
    db: Session = Depends(get_db),
):
    """
    Verify an Apple identity_token (RS256, signed by Apple) and return a JWT.
    Creates the user if they don't exist. Audience is whitelisted against
    APPLE_ALLOWED_AUDIENCES so only our own iOS + web apps are accepted.
    """
    try:
        keys_resp = _requests.get(
            "https://appleid.apple.com/auth/keys", timeout=10
        )
        keys_resp.raise_for_status()
        jwks = keys_resp.json()
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"Could not fetch Apple public keys: {exc}"
        )

    try:
        unverified_header = jwt.get_unverified_header(req.identity_token)
    except Exception as exc:
        raise HTTPException(
            status_code=401, detail=f"Malformed Apple identity token: {exc}"
        )

    kid = unverified_header.get("kid")
    matching_key = next(
        (k for k in jwks.get("keys", []) if k.get("kid") == kid), None
    )
    if not matching_key:
        raise HTTPException(status_code=401, detail="Apple signing key not found")

    try:
        public_key = jose_jwk.construct(matching_key)
        # Decode without aud check; we'll whitelist explicitly below.
        claims = jwt.decode(
            req.identity_token,
            public_key,
            algorithms=["RS256"],
            options={"verify_aud": False},
        )
    except Exception as exc:
        raise HTTPException(
            status_code=401, detail=f"Invalid Apple identity token: {exc}"
        )

    aud = claims.get("aud")
    if aud not in config.APPLE_ALLOWED_AUDIENCES:
        raise HTTPException(
            status_code=401, detail="Apple token audience not allowed"
        )

    email = claims.get("email")
    if not email:
        raise HTTPException(status_code=401, detail="Apple token missing email")

    user = db.query(User).filter(User.email == email).first()
    if not user:
        name = email.split("@")[0]
        # `sub` is Apple's stable user ID — fine as a placeholder password seed.
        user = User(
            username=name,
            email=email,
            hashed_password=_pwd.hash(claims.get("sub", "")),
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    return _issue_pair_and_set_cookies(db, user.id, request, response)


@router.get("/me")
def me(user: User = Depends(get_current_user)):
    """
    Lightweight session check used by the web client at boot to decide whether
    the user is logged in (cookie present + valid). Reused by iOS too.
    """
    return {
        "id": user.id,
        "email": user.email,
        "username": user.username,
    }


class RefreshRequest(BaseModel):
    refresh_token: Optional[str] = None


@router.post("/refresh")
@limiter.limit("60/minute")
def refresh(
    request: Request,
    response: Response,
    body: Optional[RefreshRequest] = None,
    db: Session = Depends(get_db),
):
    """
    Rotate a refresh token → new access + refresh pair. Reads the raw
    refresh token from the httpOnly cookie (web) or the JSON body (iOS).
    On any failure (expired / unknown / replayed) returns 401 with
    detail="refresh_invalid" so the frontend can distinguish from an
    expired access token and force re-login.
    """
    raw = (body.refresh_token if body else None) or request.cookies.get(config.REFRESH_COOKIE_NAME)
    if not raw:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="refresh_invalid",
        )
    result = rotate_refresh_token(db, raw, request=request)
    if result is None:
        db.commit()  # persist any family-revoke side effect
        _clear_refresh_cookie(response)
        _clear_session_cookie(response)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="refresh_invalid",
        )
    new_access, new_refresh = result
    db.commit()
    _set_session_cookie(response, new_access)
    _set_refresh_cookie(response, new_refresh)
    return {
        "access_token":       new_access,
        "refresh_token":      new_refresh,
        "token_type":         "bearer",
        "access_expires_in":  config.JWT_ACCESS_EXPIRY_MINUTES * 60,
        "refresh_expires_in": config.REFRESH_COOKIE_MAX_AGE_SECONDS,
    }


@router.post("/logout")
def logout(
    request: Request,
    response: Response,
    body: Optional[RefreshRequest] = None,
    db: Session = Depends(get_db),
):
    """
    Revoke the current refresh token (if any) and clear cookies. Safe to
    call without auth — bare cookie clear if no refresh token is found.
    """
    raw = (body.refresh_token if body else None) or request.cookies.get(config.REFRESH_COOKIE_NAME)
    if raw:
        try:
            revoke_refresh_token(db, raw)
            db.commit()
        except Exception:
            db.rollback()  # best-effort; never block logout on DB error
    _clear_session_cookie(response)
    _clear_refresh_cookie(response)
    return {"message": "Logged out"}

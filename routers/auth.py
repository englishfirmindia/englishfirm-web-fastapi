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
from core.logging_config import get_logger
from core.rate_limit import limiter
from db.database import SessionLocal, get_db
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
_log = get_logger(__name__)


def _client_ip(request: Request) -> Optional[str]:
    """Best-effort original client IP. ALB sets X-Forwarded-For; the first
    entry is the user, subsequent entries are intermediaries. Falls back
    to the direct connection IP for local/test environments."""
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip() or None
    return request.client.host if request.client else None


def _enrich_user_geoip(user_id: int, ip: Optional[str]) -> None:
    """Background task: look up country/region/city for `ip` via ipapi.co
    and UPDATE users for `user_id`. Never raises — failure leaves the
    columns NULL so signup is unaffected.

    Runs OUTSIDE the request lifecycle (BackgroundTasks) so signup
    latency is unaffected. 1.5s timeout; signups average <10/min in
    production so the free tier (1k/day) is never strained.
    """
    if not ip:
        return
    # Skip private / loopback ranges where geo lookup is meaningless.
    if ip.startswith(("10.", "127.", "192.168.", "172.16.", "172.17.",
                       "172.18.", "172.19.", "172.20.", "172.21.",
                       "172.22.", "172.23.", "172.24.", "172.25.",
                       "172.26.", "172.27.", "172.28.", "172.29.",
                       "172.30.", "172.31.", "::1", "fc", "fd")):
        return
    try:
        resp = _requests.get(f"https://ipapi.co/{ip}/json/", timeout=1.5)
        if resp.status_code != 200:
            _log.info("[GEOIP] non-200 status=%s ip=%s", resp.status_code, ip)
            return
        data = resp.json() or {}
        country = (data.get("country_code") or "").strip()[:2] or None
        region  = (data.get("region") or "").strip()[:64] or None
        city    = (data.get("city") or "").strip()[:128] or None
        if not any((country, region, city)):
            return
        db = SessionLocal()
        try:
            u = db.query(User).filter(User.id == user_id).first()
            if u is None:
                return
            u.signup_country = country
            u.signup_region  = region
            u.signup_city    = city
            db.commit()
        finally:
            db.close()
    except Exception as exc:
        # Network blip / ipapi outage / bad JSON — best-effort only.
        _log.info("[GEOIP] failed user=%s ip=%s err=%s",
                  user_id, ip, type(exc).__name__)


class SignupRequest(BaseModel):
    username: str
    email: str
    password: str
    phone: Optional[str] = None
    score_requirement: Optional[int] = None
    exam_date: Optional[str] = None
    # True when the frontend captured a ?gclid=... on first landing. False
    # for organic / direct / social / unknown. Frontend is authoritative.
    from_google_ads: bool = False
    # Acquisition detail — all optional, captured by the frontend before
    # submit. Backend persists as-is; missing fields stay NULL in the DB.
    device_class: Optional[str] = None   # mobile | tablet | desktop
    ads_keyword:  Optional[str] = None   # `utm_term` (Google's {keyword})
    ads_query:    Optional[str] = None   # `q` (Google's {query})


class GoogleAuthRequest(BaseModel):
    id_token: str
    # Same semantics as SignupRequest.from_google_ads. Only respected when
    # this Google OAuth call creates a NEW user; returning users keep their
    # existing flag (first-touch wins, even across login methods).
    from_google_ads: bool = False
    device_class: Optional[str] = None
    ads_keyword:  Optional[str] = None
    ads_query:    Optional[str] = None


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
        from_google_ads=req.from_google_ads,
        device_class=req.device_class,
        ads_keyword=req.ads_keyword,
        ads_query=req.ads_query,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    background_tasks.add_task(
        send_signup_webhook,
        student_name=req.username,
        phone_number=req.phone,
        exam_date=parsed_exam_date,
    )
    # Server-side GeoIP enrichment after response is sent — signup latency
    # unaffected. Best-effort; columns stay NULL if ipapi.co is slow/down.
    background_tasks.add_task(_enrich_user_geoip, user.id, _client_ip(request))
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
    background_tasks: BackgroundTasks,
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
    is_new_user = False
    if not user:
        name = token_info.get("name") or token_info.get("given_name") or email.split("@")[0]
        user = User(
            username=name,
            email=email,
            hashed_password=_pwd.hash(token_info.get("sub", "")),
            from_google_ads=req.from_google_ads,
            device_class=req.device_class,
            ads_keyword=req.ads_keyword,
            ads_query=req.ads_query,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        is_new_user = True
        # GeoIP enrichment for new OAuth signups only; returning users
        # already have their location from the original signup.
        background_tasks.add_task(_enrich_user_geoip, user.id, _client_ip(request))

    # _issue_pair_and_set_cookies returns the token dict; we merge in
    # is_new_user so the frontend can fire the Google Ads signup
    # conversion only on first-time signup, not returning logins.
    payload = _issue_pair_and_set_cookies(db, user.id, request, response)
    if isinstance(payload, dict):
        payload["is_new_user"] = is_new_user
    return payload


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
